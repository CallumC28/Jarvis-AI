from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Literal

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
    QTimer,
)
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from home_agent.voice import level

# ── Configuration ─────────────────────────────────────────────────────────────
SHOW_TEXT_INPUT: bool = False
ORBIT_COUNT: int = 4
PARTICLE_COUNT: int = 40
ORB_RADIUS: int = 70
ANIMATION_FPS: int = 60
SLIDE_DURATION_MS: int = 300
VOICE_SWELL: float = 0.45   # how much the orb grows at full loudness (0.45 = +45 % radius)

AgentState = Literal["idle", "listening", "thinking", "speaking"]

_PALETTE_JARVIS: dict[str, dict[str, QColor]] = {
    "listening": {"node": QColor(0, 150, 255),  "edge": QColor(0, 100, 200)},
    "thinking":  {"node": QColor(255, 180, 0),  "edge": QColor(180, 120, 0)},
    "speaking":  {"node": QColor(0, 220, 80),   "edge": QColor(0, 150, 50)},
    "idle":      {"node": QColor(40, 40, 80),   "edge": QColor(20, 20, 50)},
}

_PALETTE_OPENAI: dict[str, dict[str, QColor]] = {
    "listening": {"node": QColor(16, 163, 127),  "edge": QColor(10, 115, 90)},
    "thinking":  {"node": QColor(100, 200, 170), "edge": QColor(60, 148, 120)},
    "speaking":  {"node": QColor(0, 215, 165),   "edge": QColor(0, 158, 118)},
    "idle":      {"node": QColor(10, 52, 42),    "edge": QColor(5, 30, 24)},
}

_THEMES: dict[str, dict] = {
    "jarvis": {
        "palettes": _PALETTE_JARVIS,
        "bg": QColor(0, 0, 0, 220),
        "hud": QColor(58, 40, 0, 180),
        "label": "JARVIS",
        "label_color": QColor(120, 80, 0, 200),
    },
    "openai": {
        "palettes": _PALETTE_OPENAI,
        "bg": QColor(0, 12, 10, 220),
        "hud": QColor(0, 65, 50, 200),
        "label": "OPENAI",
        "label_color": QColor(16, 163, 127, 220),
    },
}

# Orbital ring radii (px) and their rotation speeds (rad/s) — inner rings faster
_ORBITAL_RADII: list[int] = [80, 130, 180, 230]
_ORBITAL_SPEEDS: list[float] = [0.8, 0.5, 0.35, 0.22]

_HUD_ARM = 20   # px length of each bracket arm


class _OrbCanvas(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._state: AgentState = "idle"
        self._t: float = 0.0
        self._level: float = 0.0   # smoothed loudness of whoever is talking, 0–1
        self._cx: float = 200.0
        self._cy: float = 150.0
        self._nc_rgb: tuple[int, int, int] = (40, 40, 80)
        self._ec_rgb: tuple[int, int, int] = (20, 20, 50)

        _t = _THEMES["jarvis"]
        self._palettes: dict[str, dict[str, QColor]] = _t["palettes"]
        self._bg_color: QColor = _t["bg"]
        self._hud_color: QColor = _t["hud"]
        self._provider_label: str = _t["label"]
        self._provider_label_color: QColor = _t["label_color"]
        # Each particle: (orbital_radius, angle_offset, speed_rad_per_s, clockwise)
        self._particles: list[tuple[float, float, float, bool]] = []

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._rebuild)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rebuild_timer.start(50)

    def _rebuild(self) -> None:
        self._cx = self.width() / 2.0
        self._cy = self.height() / 2.0
        rng = random.Random(42)
        self._particles = [
            (
                rng.uniform(60.0, 230.0),
                rng.uniform(0.0, 2.0 * math.pi),
                rng.uniform(0.3, 1.2),
                rng.choice([True, False]),
            )
            for _ in range(PARTICLE_COUNT)
        ]

    def set_state(self, state: AgentState) -> None:
        self._state = state
        palette = self._palettes.get(state, self._palettes["idle"])
        nc = palette["node"]
        ec = palette["edge"]
        self._nc_rgb = (nc.red(), nc.green(), nc.blue())
        self._ec_rgb = (ec.red(), ec.green(), ec.blue())

    def set_theme(self, provider: str) -> None:
        theme = _THEMES.get(provider, _THEMES["jarvis"])   # "ollama" → jarvis
        self._palettes = theme["palettes"]
        self._bg_color = theme["bg"]
        self._hud_color = theme["hud"]
        self._provider_label = theme["label"]
        self._provider_label_color = theme["label_color"]
        self.set_state(self._state)
        self.update()

    def _tick(self) -> None:
        self._t += 1.0 / ANIMATION_FPS
        target = {"listening": level.mic, "speaking": level.voice}.get(self._state, 0.0)
        # Rise fast, fall slowly — reads as a voice rather than a flicker.
        rate = 0.5 if target > self._level else 0.12
        self._level += (target - self._level) * rate
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self._bg_color)

        cx, cy = self._cx, self._cy
        t = self._t
        nr, ng, nb = self._nc_rgb
        er, eg, eb = self._ec_rgb
        pulse = (math.sin(t * 2.0) + 1.0) / 2.0   # 0.0 → 1.0
        lvl = self._level

        # ── 1. Central orb ────────────────────────────────────────────────────
        orb_r = ORB_RADIUS * (0.85 + 0.15 * pulse + VOICE_SWELL * lvl)
        grad = QRadialGradient(QPointF(cx, cy), orb_r)
        core_alpha = int(min(255, 200 + 55 * pulse + 55 * lvl))
        grad.setColorAt(0.0, QColor(
            min(nr + 180, 255), min(ng + 180, 255), min(nb + 180, 255), core_alpha,
        ))
        grad.setColorAt(0.4, QColor(nr, ng, nb, int(min(255, 60 + 100 * pulse + 95 * lvl))))
        grad.setColorAt(1.0, QColor(nr, ng, nb, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(QPointF(cx, cy), orb_r, orb_r)

        # ── 2. Orbital rings ──────────────────────────────────────────────────
        ring_alpha = int(min(255, 120 + 60 * pulse + 75 * lvl))
        for radius, speed in zip(_ORBITAL_RADII, _ORBITAL_SPEEDS, strict=False):
            rect = QRectF(cx - radius, cy - radius, radius * 2.0, radius * 2.0)
            stroke_w = 1.5 if radius <= 130 else 0.8

            # Primary arc: 270° span, clockwise rotation
            start_deg = (math.degrees(t * speed)) % 360.0
            p.setPen(QPen(QColor(er, eg, eb, ring_alpha), stroke_w))
            p.drawArc(rect, int(start_deg * 16), 270 * 16)

            # Secondary arc: 180° span, counter-rotating, offset 90°
            start2_deg = (math.degrees(-t * speed * 0.7) + 90.0) % 360.0
            p.setPen(QPen(QColor(er, eg, eb, int(ring_alpha * 0.55)), stroke_w * 0.7))
            p.drawArc(rect, int(start2_deg * 16), 180 * 16)

        # ── 3. Particles ──────────────────────────────────────────────────────
        p.setPen(Qt.PenStyle.NoPen)
        for rad, offset, speed, cw in self._particles:
            direction = 1.0 if cw else -1.0
            angle = offset + direction * t * speed
            px = cx + rad * math.cos(angle)
            py = cy + rad * math.sin(angle)
            pp = (math.sin(t * 3.0 + offset) + 1.0) / 2.0
            r = 1.5 + 1.5 * pp
            alpha = int(100 + 155 * pp)

            # Soft halo
            halo_r = r * 2.5
            hg = QRadialGradient(QPointF(px, py), halo_r)
            hg.setColorAt(0.0, QColor(nr, ng, nb, int(alpha * 0.45)))
            hg.setColorAt(1.0, QColor(nr, ng, nb, 0))
            p.setBrush(QBrush(hg))
            p.drawEllipse(QPointF(px, py), halo_r, halo_r)

            # Core dot
            p.setBrush(QBrush(QColor(nr, ng, nb, alpha)))
            p.drawEllipse(QPointF(px, py), r, r)

        # ── 4. HUD corner brackets ────────────────────────────────────────────
        p.setPen(QPen(self._hud_color, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        w, h = self.width(), self.height()
        m, arm = 10, _HUD_ARM
        # Top-left
        p.drawLine(m, m + arm, m, m)
        p.drawLine(m, m, m + arm, m)
        # Top-right
        p.drawLine(w - m - arm, m, w - m, m)
        p.drawLine(w - m, m, w - m, m + arm)
        # Bottom-left
        p.drawLine(m, h - m - arm, m, h - m)
        p.drawLine(m, h - m, m + arm, h - m)
        # Bottom-right
        p.drawLine(w - m - arm, h - m, w - m, h - m)
        p.drawLine(w - m, h - m, w - m, h - m - arm)

        # ── 5. Provider label ─────────────────────────────────────────────────
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)
        p.setFont(font)
        p.setPen(QPen(self._provider_label_color))
        p.drawText(
            QRectF(0, h - m - 18, w, 16),
            Qt.AlignmentFlag.AlignCenter,
            self._provider_label,
        )


class JarvisWindow(QMainWindow):
    """
    Cinematic orb animation popup.

    Call slide_in() to show and set_state() to control the animation colour.
    Press F2 (or call toggle_text_input()) to show/hide the debug text bar.
    """

    def __init__(
        self,
        on_text_submit: Callable[[str], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jarvis")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Popping up must not steal focus from the app the user is in (paste, "this window").
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(720, 460)

        self._on_text_submit = on_text_submit
        self._show_text_input = SHOW_TEXT_INPUT
        self._slide_anim: QPropertyAnimation | None = None

        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._canvas = _OrbCanvas()
        layout.addWidget(self._canvas, 1)

        self._input_row = self._build_input_row()
        self._input_row.setVisible(self._show_text_input)
        layout.addWidget(self._input_row)

    def _build_input_row(self) -> QWidget:
        row = QWidget()
        row.setStyleSheet("background: #0e1014;")
        hl = QHBoxLayout(row)
        hl.setContentsMargins(14, 10, 14, 10)

        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Type a message (debug)...")
        self._text_input.setStyleSheet(
            "background: #13161b; color: #f0f2f8; border: 1px solid #2a3140; "
            "border-radius: 6px; padding: 6px; font-size: 13px;"
        )
        self._text_input.returnPressed.connect(self._handle_submit)

        send_btn = QPushButton("\u2191")
        send_btn.setFixedSize(34, 34)
        send_btn.setStyleSheet(
            "background: #4d9fff; color: white; border: none; "
            "border-radius: 6px; font-size: 16px; font-weight: 700;"
        )
        send_btn.clicked.connect(self._handle_submit)

        hl.addWidget(self._text_input)
        hl.addWidget(send_btn)
        return row

    def _handle_submit(self) -> None:
        text = self._text_input.text().strip()
        if text and self._on_text_submit:
            self._text_input.clear()
            self._on_text_submit(text)

    def _screen_rect(self) -> QRect:
        screen = QApplication.primaryScreen()
        return screen.availableGeometry() if screen else QRect(0, 0, 1920, 1080)

    def _end_rect(self) -> QRect:
        geom = self._screen_rect()
        w, h = self.width(), self.height()
        return QRect(geom.right() - w - 20, geom.bottom() - h - 40, w, h)

    def _start_rect(self) -> QRect:
        geom = self._screen_rect()
        w, h = self.width(), self.height()
        return QRect(geom.right() - w - 20, geom.bottom(), w, h)

    def showEvent(self, event) -> None:  # noqa: N802
        self._canvas._timer.start(1_000 // ANIMATION_FPS)
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._canvas._timer.stop()
        super().hideEvent(event)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_state(self, state: AgentState) -> None:
        self._canvas.set_state(state)

    def set_theme(self, provider: str) -> None:
        self._canvas.set_theme(provider)

    def slide_in(self) -> None:
        if self._slide_anim is not None:
            self._slide_anim.stop()
        start = self._start_rect()
        end = self._end_rect()
        self.setGeometry(start)
        self.show()
        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(SLIDE_DURATION_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._slide_anim = anim

    def slide_out(self) -> None:
        if not self.isVisible():
            return
        if self._slide_anim is not None:
            self._slide_anim.stop()
        start = self.geometry()
        end = self._start_rect()
        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(SLIDE_DURATION_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.Type.InCubic)
        anim.finished.connect(self.hide)
        anim.start()
        self._slide_anim = anim

    def toggle_text_input(self) -> None:
        self._show_text_input = not self._show_text_input
        self._input_row.setVisible(self._show_text_input)
        if self._show_text_input:
            self.activateWindow()
            self._text_input.setFocus()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_F2:
            self.toggle_text_input()
        super().keyPressEvent(event)
