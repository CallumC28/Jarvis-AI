from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from home_agent.voice import tts
from home_agent.voice.tts import speak_stream, speak_sync


def _fake_kokoro() -> MagicMock:
    model = MagicMock()
    model.create.side_effect = lambda text, **_: (np.zeros(10, dtype=np.float32), 24_000)
    return model


def test_voice_is_british_male():
    assert tts.VOICE.startswith("bm_") and tts.LANG == "en-gb"


def test_speak_sync_synthesises_cleaned_text_and_plays():
    model = _fake_kokoro()
    with (
        patch("home_agent.voice.tts._kokoro", return_value=model),
        patch("home_agent.voice.tts._play") as play,
    ):
        speak_sync("See **this**: https://example.com now")

    model.create.assert_called_once()
    assert model.create.call_args.args[0] == "See this: now"
    play.assert_called_once()


def test_cached_phrase_is_synthesised_once():
    tts._cached_synth.cache_clear()
    model = _fake_kokoro()
    with (
        patch("home_agent.voice.tts._kokoro", return_value=model),
        patch("home_agent.voice.tts._play") as play,
    ):
        speak_sync("Yes sir?", cache=True)
        speak_sync("Yes sir?", cache=True)

    model.create.assert_called_once()
    assert play.call_count == 2


def test_speak_stream_plays_each_sentence_in_order_then_returns():
    sentences: queue.Queue[str | None] = queue.Queue()
    for s in ("First sentence here.", "Second one.", None):
        sentences.put(s)
    played: list[int] = []
    started = MagicMock()
    lengths = {"First sentence here.": 5, "Second one.": 7}
    model = MagicMock()
    model.create.side_effect = lambda text, **_: (np.zeros(lengths[text], np.float32), 24_000)

    with (
        patch("home_agent.voice.tts._kokoro", return_value=model),
        patch("home_agent.voice.tts._play", side_effect=lambda a, _: played.append(len(a)) or True),
    ):
        speak_stream(sentences, threading.Event(), on_start=started)

    gap = int(tts.SAMPLE_RATE * tts.SENTENCE_GAP_S)
    assert played == [5 + gap, 7 + gap]
    started.assert_called_once()


def test_speak_stream_stops_when_interrupted():
    sentences: queue.Queue[str | None] = queue.Queue()
    sentences.put("Never heard.")   # no None: the model is still "writing"
    stop = threading.Event()
    stop.set()
    with (
        patch("home_agent.voice.tts._kokoro", return_value=_fake_kokoro()),
        patch("home_agent.voice.tts._play") as play,
    ):
        speak_stream(sentences, stop)   # must return promptly, not hang

    play.assert_not_called()
