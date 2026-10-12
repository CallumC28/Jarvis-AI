from __future__ import annotations

import logging
import sys
import tempfile
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon, QWidget

from home_agent.core.composition import create_home_agent, unload_ollama
from home_agent.core.errors import ConfigurationError
from home_agent.core.settings import AppSettings
from home_agent.ui.image_viewer import ImageViewerWindow
from home_agent.ui.jarvis_window import JarvisWindow
from home_agent.ui.terminal_window import TerminalWindow
from home_agent.ui.tray import JarvisTray
from home_agent.voice.pipeline import VoicePipeline
from home_agent.voice.tts import speak_sync

logger = logging.getLogger(__name__)

_TEMP_IMAGE_DIRS = ("home-agent-screenshots", "home-agent-generated-images")
_IMAGE_MAX_AGE_HOURS = 24


def _cleanup_old_temp_images() -> None:
    cutoff = time.time() - _IMAGE_MAX_AGE_HOURS * 3600
    for name in _TEMP_IMAGE_DIRS:
        directory = Path(tempfile.gettempdir()) / name
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)


def _exclude_from_capture(*widgets: QWidget) -> None:
    """Hide our own windows from screenshots so screen vision sees what's behind them."""
    if sys.platform != "win32":
        return
    import ctypes

    wda_excludefromcapture = 0x11  # Windows 10 2004+; older builds just ignore it
    for w in widgets:
        ctypes.windll.user32.SetWindowDisplayAffinity(int(w.winId()), wda_excludefromcapture)


class _ImageBridge(QObject):
    """Marshals on_image callbacks (called from worker threads) to the main thread."""
    image_ready = Signal(str)


class JarvisApp:
    """
    Top-level orchestrator: owns all windows, the tray, and the voice pipeline.
    Wires signals between components.
    """

    def __init__(self) -> None:
        self._active_provider = AppSettings.load().ai_provider

        self._image_bridge = _ImageBridge()
        self._image_viewer = ImageViewerWindow()
        self._image_bridge.image_ready.connect(self._image_viewer.show_image)

        self._agent = create_home_agent(
            on_image=self._on_image_produced,
            on_progress=self._on_agent_progress,
        )

        self._terminal = TerminalWindow()
        self._terminal.attach_logging()

        self._jarvis = JarvisWindow(on_text_submit=self._on_text_submitted)
        self._jarvis.set_theme(self._active_provider)
        _exclude_from_capture(self._jarvis, self._terminal, self._image_viewer)

        self._pipeline = VoicePipeline(self._agent)
        self._connect_pipeline_signals()

        self._tray = JarvisTray(
            on_toggle_text_input=self._toggle_text_input,
            on_switch_provider=self._switch_provider,
            on_pause_toggled=self._on_pause_toggled,
        )
        self._tray.activated.connect(self._on_tray_activated)

    def _connect_pipeline_signals(self) -> None:
        self._pipeline.state_changed.connect(self._on_state_changed)
        self._pipeline.processing_started.connect(self._terminal.open_session)
        self._pipeline.agent_error.connect(self._on_agent_error)
        self._pipeline.response_ready.connect(self._on_response_ready)
        self._pipeline.provider_switch_requested.connect(self._on_provider_switch_requested)

    def _disconnect_pipeline_signals(self) -> None:
        try:
            self._pipeline.state_changed.disconnect(self._on_state_changed)
            self._pipeline.processing_started.disconnect(self._terminal.open_session)
            self._pipeline.agent_error.disconnect(self._on_agent_error)
            self._pipeline.response_ready.disconnect(self._on_response_ready)
            self._pipeline.provider_switch_requested.disconnect(self._on_provider_switch_requested)
        except RuntimeError:
            pass  # Signal already disconnected — safe to ignore

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_state_changed(self, state: str) -> None:
        self._tray.set_state(state)
        if state == "greeting":
            self._jarvis.slide_in()
            self._jarvis.set_state("listening")
        elif state == "listening":
            self._jarvis.set_state("listening")
        elif state == "processing":
            self._jarvis.set_state("thinking")
        elif state == "speaking":
            self._jarvis.set_state("speaking")
        elif state == "idle":
            self._jarvis.set_state("idle")
            self._jarvis.slide_out()
            self._terminal.close_session()   # countdown only once the conversation is over

    def _on_text_submitted(self, text: str) -> None:
        self._jarvis.slide_in()
        self._pipeline.submit_text(text)

    def _on_response_ready(self, response: str) -> None:
        logger.info('◈ RESPONSE  "%s"', response)

    def _on_agent_error(self, message: str) -> None:
        logger.error("✗ ERROR   %s", message)
        self._tray.showMessage("Jarvis", message, QSystemTrayIcon.MessageIcon.Warning)

    def _on_agent_progress(self, msg: str) -> None:
        logger.info("  ↳ %s", msg)

    def _on_image_produced(self, path: object) -> None:
        self._image_bridge.image_ready.emit(str(path))

    def _on_pause_toggled(self, paused: bool) -> None:
        self._pipeline.set_wake_word_enabled(not paused)
        logger.info("◈ %s", "LISTENING PAUSED" if paused else "LISTENING RESUMED")

    def _toggle_text_input(self) -> None:
        self._jarvis.toggle_text_input()

    def _on_provider_switch_requested(self, provider: str) -> None:
        self._switch_provider(target=provider)

    def _switch_provider(self, target: str | None = None) -> None:
        next_provider = target or ("openai" if self._active_provider == "ollama" else "ollama")
        label = "OpenAI" if next_provider == "openai" else "local"
        if next_provider == self._active_provider:
            threading.Thread(
                target=lambda: speak_sync(f"Already using the {label} agent, sir."),
                daemon=True,
            ).start()
            return
        try:
            new_agent = create_home_agent(
                provider_override=next_provider,
                on_image=self._on_image_produced,
                on_progress=self._on_agent_progress,
            )
            self._disconnect_pipeline_signals()
            self._pipeline.stop()
            self._pipeline = VoicePipeline(new_agent)
            self._connect_pipeline_signals()
            self._pipeline.start(wake_word=not self._tray.is_paused)
            self._active_provider = next_provider
            self._jarvis.set_theme(next_provider)
            logger.info("◈ SWITCH COMPLETE → %s", next_provider)
            threading.Thread(
                target=lambda: speak_sync(f"Switched to {label} agent, sir."),
                daemon=True,
            ).start()
        except ConfigurationError as exc:
            logger.error("✗ ERROR   Cannot switch provider: %s", exc)

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self._jarvis.isVisible():
                self._jarvis.hide()
            else:
                self._jarvis.slide_in()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._pipeline.start()
        logger.info("◈ JARVIS READY — SAY HEY JARVIS")

    def stop(self) -> None:
        self._pipeline.stop()
        unload_ollama(AppSettings.load())   # give the VRAM back


def run_desktop() -> None:
    if sys.stderr is None:
        # Launched via the gui-scripts (pythonw) launcher: no console, so keep output in a file.
        log_file = open(  # lives for the whole process
            Path(tempfile.gettempdir()) / "jarvis.log", "w", encoding="utf-8", buffering=1
        )
        sys.stdout = sys.stderr = log_file
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
    _cleanup_old_temp_images()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "Error", "System tray is not available on this system.")
        sys.exit(1)

    try:
        jarvis_app = JarvisApp()
    except ConfigurationError as exc:
        QMessageBox.critical(None, "Configuration Error", str(exc))
        sys.exit(1)

    app.aboutToQuit.connect(jarvis_app.stop)
    jarvis_app.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_desktop()
