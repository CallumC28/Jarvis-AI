from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRegularExpression, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import QMainWindow, QTextEdit, QVBoxLayout, QWidget

# ── Configuration ─────────────────────────────────────────────────────────────
TERMINAL_CLOSE_DELAY_SECONDS: int = 3    # change this to adjust auto-close delay
ERROR_CLOSE_MULTIPLIER: int = 2          # delay multiplier when session had errors

FONT_FAMILY = "Cascadia Mono, Fira Code, Consolas, Courier New, monospace"
FONT_SIZE = 12
BG_COLOUR = "#0a0a0a"
FG_COLOUR = "#ffffff"
ERROR_COLOUR = "#ff3333"


def _fmt(colour: str, bold: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(colour))
    if bold:
        f.setFontWeight(QFont.Weight.Bold)
    return f


# Highlight rules applied in order — later rules override earlier ones on overlaps.
_HIGHLIGHT_RULES: list[tuple[str, QTextCharFormat]] = [
    (r"◈ JARVIS ONLINE.*",                          _fmt("#00bcd4", bold=True)),
    (r"◈ (LISTENING|PROCESSING|SPEAKING|SWITCH).*", _fmt("#00bcd4")),
    (r"◈ (USER|RESPONSE).*",                        _fmt("#e0e0e0")),
    (r"◈ (SILENCE|DISMISSED|IDLE|OFFLINE).*",       _fmt("#546e7a")),
    (r"→ INVOKE.*",                                 _fmt("#ff9800", bold=True)),
    (r"✓ RESULT.*",                                 _fmt("#69f0ae")),
    (r"✗ ERROR.*",                                  _fmt("#ff3333", bold=True)),
    (r'"[^"]*"',                                    _fmt("#ffffff")),
    (r"\[.*?\]",                                    _fmt("#90a4ae")),
    (r"\b\d+(\.\d+)?\b",                            _fmt("#80cbc4")),
]


class _LogHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for log output in the terminal debug window."""

    def __init__(self, document) -> None:
        super().__init__(document)
        self._rules = [
            (QRegularExpression(pattern), fmt)
            for pattern, fmt in _HIGHLIGHT_RULES
        ]

    def highlightBlock(self, text: str) -> None:  # noqa: N802
        for regex, char_fmt in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), char_fmt)


class _HomeAgentLogFilter(logging.Filter):
    """Block any log record not from the home_agent namespace."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith("home_agent.")


class _TerminalEmitter(QObject):
    line_ready = Signal(str, bool)   # (text, is_error)


class TerminalLogHandler(logging.Handler):
    """Logging handler that pipes records to the terminal window."""

    def __init__(self, emitter: _TerminalEmitter) -> None:
        super().__init__(level=logging.DEBUG)
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        is_error = record.levelno >= logging.ERROR
        try:
            msg = self.format(record).strip()
        except Exception:
            msg = record.getMessage()
        if msg:
            self._emitter.line_ready.emit(msg, is_error)


class TerminalWindow(QMainWindow):
    """
    Hacker-style terminal debug window.

    Usage:
        terminal = TerminalWindow()
        terminal.attach_logging()          # start receiving log records
        # ... when agent starts processing:
        terminal.open_session()            # clears output, shows window
        # ... when agent finishes:
        terminal.close_session()           # starts auto-close countdown
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jarvis — Debug")
        self.resize(960, 520)
        self.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._had_error = False
        self._emitter = _TerminalEmitter(self)
        self._emitter.line_ready.connect(self._append_line)
        self._log_handler: TerminalLogHandler | None = None

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.hide)

        # ── UI ────────────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(QFont(FONT_FAMILY, FONT_SIZE))
        self._output.setStyleSheet(
            f"QTextEdit {{ background: {BG_COLOUR}; color: {FG_COLOUR}; border: none; }}"
        )
        layout.addWidget(self._output)

        self._highlighter = _LogHighlighter(self._output.document())

    # ── Public API ────────────────────────────────────────────────────────────

    def attach_logging(self) -> None:
        """Wire the root logger to this window. Call once at startup."""
        if self._log_handler is not None:
            return
        handler = TerminalLogHandler(self._emitter)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.addFilter(_HomeAgentLogFilter())
        root = logging.getLogger()
        root.setLevel(min(root.level or logging.DEBUG, logging.DEBUG))
        root.addHandler(handler)
        self._log_handler = handler

    def detach_logging(self) -> None:
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    def open_session(self) -> None:
        """Call when the agent starts processing. Shows the window; a new conversation
        (window hidden) starts with a clean slate, a follow-up keeps the transcript."""
        self._close_timer.stop()
        if not self.isVisible():
            self._had_error = False
            self._output.clear()
            self._append_line("◈ JARVIS ONLINE ──────────────────────────────────", False)
        self.show()
        self.raise_()

    def close_session(self) -> None:
        """Call when the conversation ends (pipeline idle). Starts the auto-close countdown."""
        multiplier = ERROR_CLOSE_MULTIPLIER if self._had_error else 1
        delay_ms = TERMINAL_CLOSE_DELAY_SECONDS * multiplier * 1_000
        self._close_timer.start(delay_ms)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append_line(self, text: str, is_error: bool) -> None:
        if is_error:
            self._had_error = True
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(FG_COLOUR))
        cursor.setCharFormat(fmt)
        cursor.insertText(text + "\n")
        self._output.setTextCursor(cursor)
        self._output.ensureCursorVisible()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.detach_logging()
        super().closeEvent(event)
