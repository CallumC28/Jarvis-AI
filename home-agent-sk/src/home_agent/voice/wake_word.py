from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal

from home_agent.voice import level

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_SIZE = 1_280        # 80ms of audio at 16kHz
WAKE_WORD_MODEL = "hey_jarvis"
DETECTION_THRESHOLD = 0.5
RETRIGGER_PAUSE_MS = 500     # the model is reset after a hit, so this only debounces


class WakeWordThread(QThread):
    """Always-running thread. Emits wake_detected when the wake word is heard."""

    wake_detected = Signal(float)   # detection score

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

    def run(self) -> None:
        try:
            import openwakeword
            from openwakeword.model import Model

            openwakeword.utils.download_models([WAKE_WORD_MODEL])
            model = Model(wakeword_models=[WAKE_WORD_MODEL], inference_framework="onnx")
        except Exception:
            logger.exception("Failed to initialise wake word model.")
            return

        logger.info("Wake word thread started. Listening for '%s'...", WAKE_WORD_MODEL)

        # Log model keys once so key-miss failures are immediately diagnosable
        sample_prediction = model.predict(np.zeros(CHUNK_SIZE, dtype=np.int16))
        logger.debug("openWakeWord model keys: %s", list(sample_prediction.keys()))

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE,
            ) as stream:
                while not self.isInterruptionRequested():
                    audio_chunk, _ = stream.read(CHUNK_SIZE)
                    audio_np = audio_chunk[:, 0]  # mono slice, shape (CHUNK_SIZE,) int16
                    level.mic = level.loudness(audio_np, 32_768)   # the orb pulses with this
                    prediction = model.predict(audio_np)
                    score = float(prediction.get(WAKE_WORD_MODEL, 0.0))
                    if score > DETECTION_THRESHOLD:
                        logger.info("Wake word detected (score=%.2f).", score)
                        self.wake_detected.emit(score)
                        level.mic = 0.0
                        self.msleep(RETRIGGER_PAUSE_MS)
                        # The model's rolling buffer still holds this "Hey Jarvis"; without a
                        # reset it re-fires after every pause (and interrupts replies).
                        model.reset()
                        if stream.read_available:
                            stream.read(stream.read_available)   # drop audio queued meanwhile
        except Exception:
            logger.exception("Wake word stream failed.")

    def stop(self) -> None:
        self.requestInterruption()
        if not self.wait(3_000):
            logger.warning("WakeWordThread did not stop within 3 s; it may still be running.")
