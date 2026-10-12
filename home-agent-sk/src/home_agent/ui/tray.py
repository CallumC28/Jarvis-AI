from __future__ import annotations

import functools
from collections.abc import Callable

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap, QRadialGradient
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

# Pipeline state → orb colour (matches the JarvisWindow palette)
_STATE_COLOURS: dict[str, QColor] = {
    "idle": QColor(60, 90, 170),
    "greeting": QColor(0, 150, 255),
    "listening": QColor(0, 150, 255),
    "processing": QColor(255, 180, 0),
    "speaking": QColor(0, 220, 80),
    "paused": QColor(110, 110, 110),
}


@functools.cache
def _make_icon(state: str) -> QIcon:
    colour = _STATE_COLOURS.get(state, _STATE_COLOURS["idle"])
    px = QPixmap(64, 64)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QRadialGradient(QPointF(32, 32), 30)
    grad.setColorAt(0.0, colour.lighter(190))
    grad.setColorAt(0.55, colour)
    grad.setColorAt(1.0, colour.darker(160))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    p.drawEllipse(QPointF(32, 32), 30, 30)
    p.end()
    return QIcon(px)


class JarvisTray(QSystemTrayIcon):
    def __init__(
        self,
        on_toggle_text_input: Callable[[], None],
        on_switch_provider: Callable[[], None],
        on_pause_toggled: Callable[[bool], None],
        parent=None,
    ) -> None:
        super().__init__(_make_icon("idle"), parent)
        self.setToolTip("Jarvis — Home Agent")
        self._state = "idle"
        self._on_pause_toggled = on_pause_toggled

        menu = QMenu(parent)

        self._text_action = QAction("Show Text Input (F2)", menu)
        self._text_action.setCheckable(True)
        self._text_action.triggered.connect(on_toggle_text_input)
        menu.addAction(self._text_action)

        self._switch_action = QAction("Switch Provider (Ollama / OpenAI)", menu)
        self._switch_action.triggered.connect(on_switch_provider)
        menu.addAction(self._switch_action)

        self._pause_action = QAction("Pause Listening", menu)
        self._pause_action.setCheckable(True)
        self._pause_action.toggled.connect(self._handle_pause_toggled)
        menu.addAction(self._pause_action)

        menu.addSeparator()

        quit_action = QAction("Quit Jarvis", menu)
        quit_action.triggered.connect(lambda: QApplication.instance() and QApplication.instance().quit())
        menu.addAction(quit_action)

        self.setContextMenu(menu)
        self.show()

    @property
    def is_paused(self) -> bool:
        return self._pause_action.isChecked()

    def set_state(self, state: str) -> None:
        self._state = state
        if not self.is_paused:
            self.setIcon(_make_icon(state))

    def _handle_pause_toggled(self, paused: bool) -> None:
        self.setIcon(_make_icon("paused" if paused else self._state))
        self.setToolTip("Jarvis — Paused" if paused else "Jarvis — Home Agent")
        self._on_pause_toggled(paused)
