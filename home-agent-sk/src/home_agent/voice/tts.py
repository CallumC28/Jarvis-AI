from __future__ import annotations

import functools
import logging
import math
import queue
import re
import threading
import urllib.request
from collections.abc import Callable

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

from home_agent.core.settings import DATA_DIR
from home_agent.voice import level

logger = logging.getLogger(__name__)

VOICE = "bm_lewis"        # Kokoro British male; alternatives: bm_george, bm_daniel, bm_fable
LANG = "en-gb"
SAMPLE_RATE = 24_000      # Kokoro's output rate
SENTENCE_GAP_S = 0.15     # breath between streamed sentences (Kokoro trims its own silence)

_MODEL_DIR = DATA_DIR / "models"
_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
_MODEL_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
_BLOCK = 2_400            # 0.1 s — how quickly playback notices a stop request

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_model_lock = threading.Lock()
_model: Kokoro | None = None


def _clean_for_speech(text: str) -> str:
    """Strip content that sounds bad when read aloud."""
    text = _URL_RE.sub("", text)        # remove URLs
    text = re.sub(r"[*#`]", "", text)   # markdown the model slipped in
    text = re.sub(r" {2,}", " ", text)  # collapse extra spaces
    return text.strip()


def _kokoro() -> Kokoro:
    """Load the voice once, downloading it (~350 MB) on first run."""
    global _model
    with _model_lock:
        if _model is None:
            for name in _MODEL_FILES:
                path = _MODEL_DIR / name
                if not path.exists():
                    logger.info("Downloading voice model %s (first run only)...", name)
                    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(_MODEL_URL + name, tmp)
                    tmp.replace(path)
            _model = Kokoro(str(_MODEL_DIR / _MODEL_FILES[0]), str(_MODEL_DIR / _MODEL_FILES[1]))
        return _model


def _synth(text: str) -> np.ndarray | None:
    text = _clean_for_speech(text)
    if not text:
        return None
    audio, _ = _kokoro().create(text, voice=VOICE, lang=LANG)
    return audio


_cached_synth = functools.lru_cache(maxsize=32)(_synth)


def _play(audio: np.ndarray, stop_event: threading.Event | None) -> bool:
    """Blocking playback on its own stream; returns False if stopped early.

    Stereo, not mono: on multichannel devices (e.g. 7.1 headsets) Windows may route a mono
    stream to a single channel you can't hear.
    """
    stereo = np.repeat(audio.astype(np.float32)[:, None], 2, axis=1)
    try:
        with sd.OutputStream(samplerate=SAMPLE_RATE, channels=2, dtype="float32") as out:
            for i in range(0, len(stereo), _BLOCK):
                if stop_event is not None and stop_event.is_set():
                    return False
                block = stereo[i : i + _BLOCK]
                level.voice = level.loudness(block[:, 0], 1.0)   # the orb pulses with this
                out.write(block)
        return True
    finally:
        level.voice = 0.0


def speak_sync(
    text: str, stop_event: threading.Event | None = None, cache: bool = False
) -> None:
    """Speak text, blocking. Safe from any thread; stops early if stop_event is set.

    cache=True keeps the audio in memory — use it for fixed phrases only.
    """
    audio = _cached_synth(text) if cache else _synth(text)
    if audio is not None:
        _play(audio, stop_event)


def speak_stream(
    sentences: queue.Queue[str | None],
    stop_event: threading.Event,
    on_start: Callable[[], None] | None = None,
) -> None:
    """Speak sentences as they arrive (None ends the stream), rendering the next during playback."""
    rendered: queue.Queue[np.ndarray | None] = queue.Queue()

    def render() -> None:
        try:
            while not stop_event.is_set() and (sentence := sentences.get()) is not None:
                audio = _synth(sentence)
                if audio is not None:
                    rendered.put(audio)
        except Exception:
            logger.exception("Speech synthesis failed.")
        finally:
            rendered.put(None)

    threading.Thread(target=render, daemon=True).start()
    gap = np.zeros(int(SAMPLE_RATE * SENTENCE_GAP_S), dtype=np.float32)
    started = False
    while not stop_event.is_set():
        try:
            audio = rendered.get(timeout=0.1)
        except queue.Empty:
            continue
        if audio is None:
            return
        if not started and on_start is not None:
            on_start()
        started = True
        if not _play(np.concatenate([audio, gap]), stop_event):
            return


def preload_voice(*phrases: str) -> None:
    """Load and warm the voice (the first synthesis is slow) and pre-render fixed phrases."""
    _synth("Ready.")
    for phrase in phrases:
        _cached_synth(phrase)


@functools.cache
def _chime() -> np.ndarray:
    rate = SAMPLE_RATE
    notes = []
    for freq, dur in ((880, 0.07), (1_320, 0.09)):   # two rising notes, ~160 ms total
        n = int(rate * dur)
        i = np.arange(n)
        fade = np.minimum(1.0, np.minimum(i / 200, (n - i) / 400))  # avoid clicks at note edges
        notes.append(0.25 * fade * np.sin(2 * math.pi * freq * i / rate))
    return np.concatenate(notes).astype(np.float32)


def play_chime() -> None:
    """Short non-blocking 'I'm listening' cue."""
    sd.play(np.repeat(_chime()[:, None], 2, axis=1), SAMPLE_RATE)
