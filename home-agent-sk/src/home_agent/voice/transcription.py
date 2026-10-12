from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from faster_whisper.vad import get_vad_model
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_DURATION = 0.5               # seconds per audio chunk
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
VAD_THRESHOLD = 0.5                # Silero speech probability; raise if noise keeps turns open
SILENCE_DURATION = 1.5             # silence after speech -> end of turn (raise if cut off)
START_TIMEOUT = 4.0                # seconds to wait for speech to begin before giving up
LISTEN_TIMEOUT = 10.0              # max seconds per utterance, from first speech
GPU_MODEL = "large-v3-turbo"       # used when CUDA works (pip install -e .[gpu])
CPU_MODEL = "small"                # fallback: tiny | base | small | medium

# Loaded once at startup; every TranscriptionThread reuses this instance.
_model_lock = threading.Lock()
_whisper_model: WhisperModel | None = None


def _add_cuda_dll_dirs() -> None:
    """The nvidia-cublas/cudnn wheels don't put their DLLs on PATH; CTranslate2 looks there."""
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        for bin_dir in Path(base).glob("*/bin"):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def _load_model() -> WhisperModel:
    _add_cuda_dll_dirs()
    try:
        logger.info("Loading Whisper model '%s' on CUDA...", GPU_MODEL)
        model = WhisperModel(GPU_MODEL, device="cuda", compute_type="float16")
        # Missing CUDA DLLs only surface on the first decode; it also warms up the kernels.
        list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="en")[0])
        return model
    except Exception as exc:
        logger.warning("CUDA Whisper unavailable (%s); using CPU '%s'.", exc, CPU_MODEL)
        return WhisperModel(CPU_MODEL, device="cpu", compute_type="int8")


def _get_whisper_model() -> WhisperModel:
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            _whisper_model = _load_model()
        return _whisper_model


def preload_whisper() -> None:
    """Call once at startup (off the main thread) to warm up the model."""
    _get_whisper_model()


def _is_speech(chunk: np.ndarray) -> bool:
    """Silero VAD (bundled with faster-whisper) — hears voices, not volume, so fans and
    keyboards don't hold the turn open and quiet speakers aren't cut off."""
    audio = chunk.astype(np.float32) / 32_768.0
    audio = np.pad(audio, (0, -len(audio) % 512))   # Silero works in 512-sample windows
    return float(np.max(get_vad_model()(audio))) >= VAD_THRESHOLD


class TranscriptionThread(QThread):
    """Records from mic until silence, then transcribes with faster-whisper."""

    transcription_ready = Signal(str)
    transcription_failed = Signal()

    def __init__(self, parent=None, start_timeout: float | None = None) -> None:
        super().__init__(parent)
        self._start_timeout = start_timeout

    def run(self) -> None:
        try:
            self._record_and_transcribe()
        except Exception:
            logger.exception("TranscriptionThread encountered an unexpected error.")
            self.transcription_failed.emit()

    def _record_and_transcribe(self) -> None:
        model = _get_whisper_model()
        audio_chunks: list[np.ndarray] = []
        silence_start: float | None = None
        heard_speech = False
        start_time = time.monotonic()
        start_timeout = START_TIMEOUT if self._start_timeout is None else self._start_timeout

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while not self.isInterruptionRequested():
                if heard_speech and time.monotonic() - start_time > LISTEN_TIMEOUT:
                    logger.info("Transcription: listen timeout.")
                    break

                chunk, _ = stream.read(CHUNK_SAMPLES)
                # chunk is shape (CHUNK_SAMPLES, 1) from sounddevice; take mono channel
                audio_np = chunk[:, 0]
                audio_chunks.append(audio_np.copy())

                if _is_speech(audio_np):
                    if not heard_speech:
                        start_time = time.monotonic()   # utterance cap counts from first speech
                    heard_speech = True
                    silence_start = None
                elif not heard_speech:
                    audio_chunks = audio_chunks[-1:]   # keep one chunk of pre-roll, drop the rest
                    if time.monotonic() - start_time >= start_timeout:
                        break
                elif silence_start is None:
                    silence_start = time.monotonic()
                elif time.monotonic() - silence_start >= SILENCE_DURATION:
                    break

        # Whisper invents text ("Thank you.", "you") when fed pure silence — don't let it.
        if self.isInterruptionRequested() or not heard_speech:
            self.transcription_failed.emit()
            return

        combined = np.concatenate(audio_chunks).astype(np.float32) / 32_768.0
        segments, _ = model.transcribe(combined, beam_size=1, language="en", vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()

        if text:
            logger.info("Transcribed: %r", text)
            self.transcription_ready.emit(text)
        else:
            self.transcription_failed.emit()

    def stop(self) -> None:
        self.requestInterruption()
        if not self.wait(3_000):
            logger.warning("TranscriptionThread did not stop within 3 s; it may still be running.")
