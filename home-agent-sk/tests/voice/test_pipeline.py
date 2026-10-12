from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication, QThreadPool

from home_agent.voice.pipeline import VoicePipeline


@pytest.fixture(scope="session")
def qt_app():
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    return app


@pytest.fixture
def mock_agent():
    return MagicMock()


def test_pipeline_emits_greeting_state_on_wake(qt_app, mock_agent):
    states = []

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.play_chime"),
        patch("home_agent.voice.pipeline.TranscriptionThread"),
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline.state_changed.connect(states.append)
        pipeline._on_wake_detected()

    assert "greeting" in states


def test_pipeline_emits_listening_after_greeting(qt_app, mock_agent):
    states = []

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.play_chime"),
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread
        pipeline = VoicePipeline(mock_agent)
        pipeline.state_changed.connect(states.append)
        pipeline._start_listening()

    assert "listening" in states
    mock_thread.start.assert_called_once()


def test_pipeline_emits_idle_on_transcription_failed(qt_app, mock_agent):
    states = []

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.play_chime"),
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline.state_changed.connect(states.append)
        pipeline._on_transcription_failed()

    assert states[-1] == "idle"


def test_speaking_done_listens_for_followup_without_asking(qt_app, mock_agent):
    """After a reply Jarvis just keeps listening (long start window) — no "Is that all, sir?"."""
    from home_agent.voice.pipeline import FOLLOWUP_START_TIMEOUT

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.speak_sync") as speak,
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"
        pipeline._on_speaking_done()

    speak.assert_not_called()
    mock_thread_cls.assert_called_once_with(start_timeout=FOLLOWUP_START_TIMEOUT)
    mock_thread_cls.return_value.start.assert_called_once()
    assert pipeline._current_state == "listening"


def test_wake_word_during_speaking_sets_interrupt_flag(qt_app, mock_agent):
    """Wake word during SPEAKING sets interrupt_requested and the tts stop event."""
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.QThreadPool"),
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"
        pipeline._on_wake_detected()

    assert pipeline._interrupt_requested is True
    assert pipeline._tts_stop_event.is_set()


def test_speaking_done_skips_relisten_when_interrupted(qt_app, mock_agent):
    """_on_speaking_done must not re-enter listening if an interrupt is in progress."""
    listening_calls: list[int] = []

    with patch("home_agent.voice.pipeline.WakeWordThread"):
        pipeline = VoicePipeline(mock_agent)
        pipeline._interrupt_requested = True
        pipeline._listening_requested.connect(lambda: listening_calls.append(1))
        pipeline._on_speaking_done()

    assert len(listening_calls) == 0
    assert pipeline._interrupt_requested is False  # flag was cleared


def test_wake_word_during_speaking_eventually_emits_listening(qt_app, mock_agent):
    """Interrupt must eventually emit _listening_requested after 'Yes sir?' plays."""
    listening_calls: list[int] = []

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.speak_sync"),  # suppress audio
        patch("home_agent.voice.pipeline.TranscriptionThread"),  # don't open the real mic
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"
        pipeline._listening_requested.connect(lambda: listening_calls.append(1))
        pipeline._on_wake_detected()
        QThreadPool.globalInstance().waitForDone(2000)  # let worker finish
        qt_app.processEvents()  # deliver queued signals

    assert len(listening_calls) == 1


def test_wake_from_idle_chimes_and_listens_immediately(qt_app, mock_agent):
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.play_chime") as chime,
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._on_wake_detected()

    chime.assert_called_once()
    mock_thread_cls.return_value.start.assert_called_once()
    assert pipeline._current_state == "listening"


def test_agent_error_speaks_apology_and_goes_idle(qt_app, mock_agent):
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.speak_sync") as speak,
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._on_agent_error("boom")
        QThreadPool.globalInstance().waitForDone(2000)

    assert pipeline._current_state == "idle"
    speak.assert_called_once()


def test_is_dismissal_normalises_whisper_output() -> None:
    from home_agent.voice.pipeline import _is_dismissal

    for text in ("That is all.", "That's all, sir.", "Shut up!", "Thank you, Jarvis."):
        assert _is_dismissal(text), text
    assert not _is_dismissal("That is all I need to know about Python?")


def test_wake_while_speaking_needs_confident_score_after_guard(qt_app, mock_agent):
    """Long replies must stay interruptible; weak/early detections (own-voice echo) are ignored."""
    import time

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.QThreadPool"),
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"

        pipeline._speaking_started_at = time.monotonic()
        pipeline._on_wake_detected(0.99)          # inside the echo guard
        assert not pipeline._interrupt_requested

        pipeline._speaking_started_at = time.monotonic() - 30   # 30 s into a long reply
        pipeline._on_wake_detected(0.6)           # weak — likely our own voice
        assert not pipeline._interrupt_requested

        pipeline._on_wake_detected(0.99)
        assert pipeline._interrupt_requested


def test_pure_command_goes_idle_after_speaking_instead_of_listening(qt_app, mock_agent):
    mock_agent.last_tools = ["pc.open_app", "pc.set_volume"]
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"
        pipeline._on_agent_response("Spotify's up, volume at thirty.")
        pipeline._on_speaking_done()

    mock_thread_cls.assert_not_called()
    assert pipeline._current_state == "idle"


def test_a_question_that_used_a_tool_still_listens_for_follow_up(qt_app, mock_agent):
    mock_agent.last_tools = ["pc.open_app", "web.web_search"]
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._current_state = "speaking"
        pipeline._on_agent_response("Here's what I found.")
        pipeline._on_speaking_done()

    mock_thread_cls.return_value.start.assert_called_once()


def test_short_answers_to_a_question_are_not_dismissals() -> None:
    from home_agent.voice.pipeline import _is_dismissal

    for text in ("Yes.", "Yeah", "No.", "Nothing."):
        assert not _is_dismissal(text), text


def test_strict_provider_switch_fast_path() -> None:
    from home_agent.voice.pipeline import _detect_provider_switch

    assert _detect_provider_switch("Switch to OpenAI.") == "openai"
    assert _detect_provider_switch("Jarvis, go back to the local one, please.") == "ollama"
    assert _detect_provider_switch("Switch to the local news") is None
    assert _detect_provider_switch("Can you use your local knowledge?") is None


def _finish_turn(pipeline, tools):
    pipeline._agent.last_tools = tools
    pipeline._current_state = "speaking"
    pipeline._on_agent_response("Righto.")
    pipeline._on_speaking_done()


def test_end_conversation_tool_goes_idle(qt_app, mock_agent):
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        _finish_turn(pipeline, ["web.web_search", "assistant.end_conversation"])

    mock_thread_cls.assert_not_called()
    assert pipeline._current_state == "idle"


def test_switch_tool_requests_provider_after_speaking(qt_app, mock_agent):
    with patch("home_agent.voice.pipeline.WakeWordThread"):
        pipeline = VoicePipeline(mock_agent)
        switched: list[str] = []
        pipeline.provider_switch_requested.connect(switched.append)
        _finish_turn(pipeline, ["assistant.use_openai"])

    assert switched == ["openai"]
    assert pipeline._current_state == "idle"


def test_dictation_tool_starts_dictating_after_speaking(qt_app, mock_agent):
    from home_agent.voice.pipeline import DICTATION_START_TIMEOUT

    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.play_chime"),
        patch("home_agent.voice.pipeline.TranscriptionThread") as mock_thread_cls,
    ):
        pipeline = VoicePipeline(mock_agent)
        _finish_turn(pipeline, ["assistant.start_dictation"])

    assert pipeline._dictating
    mock_thread_cls.assert_called_once_with(start_timeout=DICTATION_START_TIMEOUT)


def test_dictation_stop_phrase_at_end_pastes_the_rest(qt_app, mock_agent):
    with (
        patch("home_agent.voice.pipeline.WakeWordThread"),
        patch("home_agent.voice.pipeline.paste_text") as paste,
        patch("home_agent.voice.pipeline.TranscriptionThread"),
    ):
        pipeline = VoicePipeline(mock_agent)
        pipeline._dictating = True
        pipeline._on_transcription_ready("See you on Friday. Stop dictation, Jarvis.")

    paste.assert_called_once_with("See you on Friday. ")
    assert not pipeline._dictating
    assert pipeline._current_state == "idle"
