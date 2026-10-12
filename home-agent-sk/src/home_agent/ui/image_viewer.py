from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

_MAX_W = 1200
_MAX_H = 900


class ImageViewerWindow(QMainWindow):
    """
    Displays generated images and screenshots.
    Stays open until the user manually closes it.
    Calling show_image() again replaces the current image.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jarvis — Image")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background: #0a0a0a;")

        central = QWidget()
        central.setStyleSheet("background: #0a0a0a;")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background: #0a0a0a;")
        layout.addWidget(self._label)

    def show_image(self, path: str) -> None:
        """Load *path*, scale to fit the max viewport, resize the window, and show."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(
            _MAX_W,
            _MAX_H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self.resize(scaled.width(), scaled.height())
        self.show()
        self.raise_()
        self.activateWindow()
