from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from home_agent.voice.wake_word import CHUNK_SIZE, WakeWordThread


@pytest.fixture(scope="session")
def qt_app():
    return QCoreApplication.instance() or QCoreApplication(sys.argv)


class _EndOfAudio(Exception):
    """Ends run()'s loop — requestInterruption() is ignored on a thread that was never started."""


def test_one_wake_word_fires_once_not_every_pause(qt_app):
    thread = WakeWordThread()

    class StickyModel:
        """Like openWakeWord: keeps scoring the last wake word until reset()."""

        heard = True

        def predict(self, _audio: np.ndarray) -> dict[str, float]:
            if not self.heard:
                raise _EndOfAudio   # run() logs and returns
            return {"hey_jarvis": 1.0 if self.heard else 0.0}

        def reset(self) -> None:
            self.heard = False

    stream = MagicMock(read_available=0)
    stream.__enter__ = lambda s: s
    stream.__exit__ = MagicMock(return_value=False)
    stream.read.return_value = (np.zeros((CHUNK_SIZE, 1), dtype=np.int16), False)
    emitted: list[float] = []
    thread.wake_detected.connect(emitted.append)

    with (
        patch("openwakeword.utils.download_models"),
        patch("openwakeword.model.Model", return_value=StickyModel()),
        patch("home_agent.voice.wake_word.sd.InputStream", return_value=stream),
        patch.object(thread, "msleep"),
    ):
        thread.run()

    assert emitted == [1.0]
