from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication

from home_agent.voice.transcription import CHUNK_SAMPLES, TranscriptionThread, _is_speech


@pytest.fixture(scope="session")
def qt_app():
    import sys
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def loud_means_speech():
    """The synthetic chunks below aren't real voices — treat any non-zero chunk as speech."""
    with patch("home_agent.voice.transcription._is_speech", lambda chunk: bool(chunk.any())):
        yield


def test_vad_hears_no_speech_in_silence_or_a_steady_hum():
    t = np.arange(CHUNK_SAMPLES) / 16_000
    hum = (3_000 * np.sin(2 * np.pi * 120 * t)).astype(np.int16)   # loud, but not a voice
    assert not _is_speech(np.zeros(CHUNK_SAMPLES, dtype=np.int16))
    assert not _is_speech(hum)


def _speech_chunk() -> np.ndarray:
    """Return a non-silent (CHUNK_SAMPLES, 1) int16 array."""
    return (np.ones((CHUNK_SAMPLES, 1), dtype=np.int16) * 5_000)


def _silence_chunk() -> np.ndarray:
    """Return a (CHUNK_SAMPLES, 1) int16 array of zeros (silence)."""
    return np.zeros((CHUNK_SAMPLES, 1), dtype=np.int16)


def test_transcription_ready_emitted_on_speech(qt_app):
    """Thread emits transcription_ready with the transcribed text."""
    call_count = 0

    def fake_stream_read(n):
        nonlocal call_count
        call_count += 1
        # First 3 chunks: speech; then enough silence to trigger stop
        if call_count <= 3:
            return _speech_chunk(), False
        return _silence_chunk(), False

    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = fake_stream_read

    mock_segment = MagicMock()
    mock_segment.text = " Hello Sir"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    results = []
    failures = []

    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription.WhisperModel", return_value=mock_model),
    ):
        thread = TranscriptionThread()
        thread.transcription_ready.connect(results.append)
        thread.transcription_failed.connect(lambda: failures.append(True))
        # Call run() directly — executes synchronously in test thread without Qt machinery
        thread.run()

    assert results == ["Hello Sir"]
    assert not failures


def test_transcription_failed_emitted_when_only_silence(qt_app):
    """Thread emits transcription_failed when all audio is silence."""
    call_count = 0

    def fake_stream_read(n):
        nonlocal call_count
        call_count += 1
        return _silence_chunk(), False

    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = fake_stream_read

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    failures = []
    results = []

    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription.WhisperModel", return_value=mock_model),
    ):
        thread = TranscriptionThread()
        thread.transcription_ready.connect(results.append)
        thread.transcription_failed.connect(lambda: failures.append(True))
        thread.run()

    assert failures
    assert not results


def test_long_speech_is_transcribed_at_timeout_not_discarded(qt_app):
    """Talking past LISTEN_TIMEOUT must still yield a transcription."""
    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = lambda n: (_speech_chunk(), False)   # never goes quiet

    mock_segment = MagicMock()
    mock_segment.text = " a very long request"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())

    results = []
    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription._get_whisper_model", return_value=mock_model),
        patch("home_agent.voice.transcription.LISTEN_TIMEOUT", 0.05),
    ):
        thread = TranscriptionThread()
        thread.transcription_ready.connect(results.append)
        thread.run()

    assert results == ["a very long request"]


def test_silence_is_never_sent_to_whisper(qt_app):
    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = lambda n: (_silence_chunk(), False)
    mock_model = MagicMock()

    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription._get_whisper_model", return_value=mock_model),
        patch("home_agent.voice.transcription.START_TIMEOUT", 0.05),
    ):
        TranscriptionThread().run()

    mock_model.transcribe.assert_not_called()


def test_start_timeout_allows_a_long_pause_before_speaking(qt_app):
    """Follow-up listening waits start_timeout for speech to begin; SILENCE_DURATION only ends a turn."""
    clock = [0.0]
    reads = [0]

    def fake_read(n):
        reads[0] += 1
        clock[0] += 0.5
        # 6 s of silence (longer than SILENCE_DURATION), then speech, then silence
        return (_speech_chunk() if 13 <= reads[0] <= 15 else _silence_chunk()), False

    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = fake_read
    seg = MagicMock()
    seg.text = " and the second one?"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([seg], MagicMock())

    results = []
    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription._get_whisper_model", return_value=mock_model),
        patch("home_agent.voice.transcription.time.monotonic", lambda: clock[0]),
    ):
        thread = TranscriptionThread(start_timeout=8.0)
        thread.transcription_ready.connect(results.append)
        thread.run()

    assert results == ["and the second one?"]
    # leading silence is trimmed rather than all sent to Whisper
    assert len(mock_model.transcribe.call_args[0][0]) < 10 * CHUNK_SAMPLES


def test_gives_up_when_nobody_speaks_within_start_timeout(qt_app):
    clock = [0.0]

    def fake_read(n):
        clock[0] += 0.5
        return _silence_chunk(), False

    mock_stream = MagicMock()
    mock_stream.__enter__ = lambda s: s
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.read = fake_read
    failures = []
    with (
        patch("home_agent.voice.transcription.sd.InputStream", return_value=mock_stream),
        patch("home_agent.voice.transcription._get_whisper_model", return_value=MagicMock()),
        patch("home_agent.voice.transcription.time.monotonic", lambda: clock[0]),
    ):
        thread = TranscriptionThread(start_timeout=8.0)
        thread.transcription_failed.connect(lambda: failures.append(True))
        thread.run()

    assert failures
    assert 8.0 <= clock[0] <= 9.0
