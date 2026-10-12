from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal

from home_agent.core.agent import HomeAgent
from home_agent.core.errors import HomeAgentError
from home_agent.plugins.pc import paste_text
from home_agent.voice.transcription import TranscriptionThread, preload_whisper
from home_agent.voice.tts import play_chime, preload_voice, speak_stream, speak_sync
from home_agent.voice.wake_word import WakeWordThread

logger = logging.getLogger(__name__)


# Instant dismissals that skip the model. Anything looser ("right, I'm off") reaches the model,
# which calls assistant.end_conversation. No bare yes/no/nothing: those answer Jarvis's questions.
_DISMISSAL_PHRASES: frozenset[str] = frozenset({
    "that's all", "thats all", "that is all",
    "that'll be all", "thatll be all", "that will be all",
    "nothing else", "nothing more", "nothing further",
    "i'm good", "im good", "all good",
    "thank you", "thanks", "cheers",
    "stop", "shut up", "be quiet", "quiet", "never mind", "nevermind",
    "go away", "goodbye", "bye", "dismissed", "no thanks",
})


# Dictation (started by the model via assistant.start_dictation): everything said is typed into
# the active window, verbatim, until an utterance ends with a stop phrase. The model isn't in
# the loop while dictating, so this one stays a phrase match.
_DICTATION_STOP = re.compile(
    r"\b(?:stop|end|finish) (?:dictation|dictating|typing)[\s,.!]*(?:sir|jarvis)?[.!]*$",
    re.IGNORECASE,
)
DICTATION_START_TIMEOUT = 15.0   # pauses to think are longer when writing

# A turn that only used these tools was a command: Jarvis confirms and goes, no follow-up.
_COMMAND_TOOLS: frozenset[str] = frozenset({
    "pc.open_app", "pc.close_app", "pc.switch_to_app", "pc.move_window", "pc.arrange_window",
    "pc.minimise_all", "pc.open_uri", "pc.media", "pc.set_volume", "pc.copy_to_clipboard",
    "pc.paste_text", "browser.open_website", "git.open_dashboard", "desktop.create_desktop_folder",
    "organizer.add_note", "organizer.add_todo", "organizer.complete_todo",
})

_YES_SIR = "Yes sir?"
_ERROR_REPLY = "Sorry sir, something went wrong."

# After a reply we just keep listening, like someone in the room; silence this long → idle.
FOLLOWUP_START_TIMEOUT = 4.0   # raise if follow-ups get cut off before you start talking

# While speaking, "Hey Jarvis" interrupts. Our own voice through the speakers can fool the
# detector, so require a confident score (a real "Hey Jarvis" scores ~0.99) and ignore the
# first second of playback.
# ponytail: fixed threshold — lower it if interrupts get missed, raise it on self-triggers.
_INTERRUPT_THRESHOLD = 0.8
_ECHO_GUARD_S = 1.0


def _normalise(text: str) -> str:
    # Whisper adds punctuation freely ("Jarvis, that is all, sir.") — strip it and any address.
    words = re.sub(r"[^\w\s']", "", text.lower()).split()
    while words and words[-1] in ("sir", "jarvis"):
        words.pop()
    while words and words[0] in ("hey", "jarvis", "sir"):
        words.pop(0)
    return " ".join(words)


def _is_dismissal(text: str) -> bool:
    return _normalise(text) in _DISMISSAL_PHRASES


# Model-free fallback so you can still switch when the current model is down. Whole-sentence
# match only ("switch to the local news" must not switch); looser wording goes through the
# model's assistant.use_openai / use_local tools.
_SWITCH = re.compile(
    r"(?:can you |please )?(?:switch|change|go)(?: back)?(?: over)? to (?:the )?"
    r"(openai|open ai|cloud|local|ollama)(?: agent| model| one| ai| version)?(?: please)?"
)


def _detect_provider_switch(text: str) -> str | None:
    match = _SWITCH.fullmatch(_normalise(text))
    if match is None:
        return None
    return "ollama" if match.group(1) in ("local", "ollama") else "openai"


def _run_loop_forever() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="agent-loop", daemon=True).start()
    return loop


# One loop for the app's lifetime: the OpenAI/Ollama client keeps connections bound to the
# loop that opened them, so a fresh asyncio.run() per turn made every first request fail
# ("Event loop is closed") and retry.
_AGENT_LOOP = _run_loop_forever()


class _AgentChatThread(QThread):
    """Runs HomeAgent.chat() off the main thread."""

    finished_message = Signal(str)
    failed_message = Signal(str)

    def __init__(
        self,
        agent: HomeAgent,
        user_text: str,
        on_sentence: Callable[[str | None], None],
        image_paths: list[Path] | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._user_text = user_text
        self._on_sentence = on_sentence
        self._image_paths = image_paths or []

    def run(self) -> None:
        try:
            response = asyncio.run_coroutine_threadsafe(
                self._agent.chat(
                    self._user_text, image_paths=self._image_paths, on_sentence=self._on_sentence
                ),
                _AGENT_LOOP,
            ).result()
            self.finished_message.emit(response)
        except HomeAgentError as exc:
            self.failed_message.emit(str(exc))
        except Exception:
            logger.exception("Unexpected error in agent thread.")
            self.failed_message.emit("Unexpected failure — check logs.")
        finally:
            self._on_sentence(None)   # end of speech stream


class _Task(QRunnable):
    """Fire-and-forget callable for the Qt thread pool."""

    def __init__(self, fn: Callable[[], None]) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self._fn()
        except Exception:
            logger.exception("_Task callable raised an unhandled exception.")


class VoicePipeline(QObject):
    """
    State machine: IDLE → GREETING → LISTENING → PROCESSING → SPEAKING → LISTENING (loop)
                   LISTENING → IDLE on silence/timeout; SPEAKING/PROCESSING → IDLE on error

    Emits state_changed(str) on every transition.
    Emits processing_started to drive the terminal window.
    """

    state_changed = Signal(str)
    response_ready = Signal(str)
    agent_error = Signal(str)
    processing_started = Signal()
    provider_switch_requested = Signal(str)

    # Private signals used to marshal transitions back onto the main thread
    # from QThreadPool workers (QObject creation must happen on the correct thread).
    _listening_requested = Signal()
    _speech_started = Signal()
    _speaking_done = Signal()

    def __init__(self, agent: HomeAgent, parent=None) -> None:
        super().__init__(parent)
        self._agent = agent
        self._current_state = "idle"
        self._agent_thread: _AgentChatThread | None = None
        self._transcription_thread: TranscriptionThread | None = None
        self._interrupt_requested: bool = False
        self._dictating: bool = False
        self._follow_up: bool = True   # False after a pure command: go idle once it's said
        # Set from the turn's assistant.* tool calls; acted on once the reply has been spoken.
        self._pending_switch: str | None = None
        self._pending_dictation: bool = False
        # One event per spoken response, so interrupting one can't silence the next.
        self._tts_stop_event: threading.Event = threading.Event()
        self._speaking_started_at: float = 0.0

        self._wake_thread = WakeWordThread()
        self._wake_thread.wake_detected.connect(self._on_wake_detected)

        # These connections are QueuedConnection by default when crossing threads,
        # ensuring _start_listening and _on_idle are called on the main thread.
        self._listening_requested.connect(self._start_listening)
        self._speech_started.connect(self._on_speech_started)
        self._speaking_done.connect(self._on_speaking_done)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, wake_word: bool = True) -> None:
        QThreadPool.globalInstance().start(_Task(preload_whisper))
        QThreadPool.globalInstance().start(
            _Task(lambda: preload_voice(_YES_SIR, _ERROR_REPLY))
        )
        self.set_wake_word_enabled(wake_word)
        self._set_state("idle")
        logger.info("◈ JARVIS ONLINE ──────────────────────────────────")

    def stop(self) -> None:
        self._tts_stop_event.set()
        self._wake_thread.stop()
        if self._transcription_thread and self._transcription_thread.isRunning():
            self._transcription_thread.stop()   # cooperative shutdown
        logger.info("◈ OFFLINE")

    def set_wake_word_enabled(self, enabled: bool) -> None:
        """Start/stop the always-on microphone (tray "Pause Listening")."""
        if enabled:
            self._wake_thread.start()
        else:
            self._wake_thread.stop()

    def submit_text(self, text: str) -> None:
        """Bypass voice input — submit text directly (used by debug text bar)."""
        self._on_transcription_ready(text)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_state(self, state: str) -> None:
        self._current_state = state
        self.state_changed.emit(state)

    # ── State transitions ─────────────────────────────────────────────────────

    def _on_wake_detected(self, score: float = 1.0) -> None:
        if self._current_state == "idle":
            self._interrupt_requested = False
            self._set_state("greeting")
            # Chime instead of a spoken greeting so the mic opens immediately.
            play_chime()
            self._start_listening()

        elif self._current_state == "speaking":
            elapsed = time.monotonic() - self._speaking_started_at
            if elapsed < _ECHO_GUARD_S or score < _INTERRUPT_THRESHOLD:
                logger.info("◈ INTERRUPT IGNORED (score=%.2f, %.1fs in)", score, elapsed)
                return
            logger.info("◈ INTERRUPTED")
            self._interrupt_requested = True
            self._tts_stop_event.set()

            def _acknowledge_then_listen() -> None:
                if self._current_state != "speaking":
                    return
                speak_sync(_YES_SIR, cache=True)
                self._listening_requested.emit()

            QThreadPool.globalInstance().start(_Task(_acknowledge_then_listen))


    def _start_listening(self, start_timeout: float | None = None) -> None:
        self._set_state("listening")
        self._transcription_thread = TranscriptionThread(start_timeout=start_timeout)
        self._transcription_thread.transcription_ready.connect(self._on_transcription_ready)
        self._transcription_thread.transcription_failed.connect(self._on_transcription_failed)
        self._transcription_thread.start()

    def _on_transcription_ready(self, text: str) -> None:
        if self._dictating:
            ended = _DICTATION_STOP.search(text)
            dictated = (text[: ended.start()] if ended else text).strip()
            if dictated:
                logger.info('◈ DICTATE "%s"', dictated)
                try:
                    paste_text(dictated + " ")
                except Exception:
                    logger.exception("Dictation paste failed.")
            if ended:
                self._dictating = False
                logger.info("◈ DICTATION OFF — IDLE")
                self._set_state("idle")
                return
            self._start_listening(start_timeout=DICTATION_START_TIMEOUT)
            return

        if _is_dismissal(text):
            logger.info("◈ DISMISSED — IDLE")
            self._set_state("idle")
            return

        provider = _detect_provider_switch(text)
        if provider is not None:
            logger.info("◈ SWITCH → %s", provider)
            self.provider_switch_requested.emit(provider)
            self._set_state("idle")
            return

        self._set_state("processing")
        self.processing_started.emit()   # open_session clears terminal and emits banner
        logger.info('◈ USER    "%s"', text)
        logger.info("◈ PROCESSING...")

        self._follow_up = True
        # Speech starts with the first finished sentence while the model is still writing.
        sentences: queue.Queue[str | None] = queue.Queue()
        stop = self._tts_stop_event = threading.Event()
        self._agent_thread = _AgentChatThread(self._agent, text, sentences.put)
        self._agent_thread.finished_message.connect(self._on_agent_response)
        self._agent_thread.failed_message.connect(self._on_agent_error)
        self._agent_thread.start()

        def _speak() -> None:
            try:
                speak_stream(sentences, stop, on_start=self._speech_started.emit)
            finally:
                # Emit across thread boundary → _on_speaking_done runs on main thread
                self._speaking_done.emit()

        QThreadPool.globalInstance().start(_Task(_speak))

    def _on_transcription_failed(self) -> None:
        if self._dictating:
            self._dictating = False
            logger.info("◈ DICTATION OFF")
        logger.info("◈ SILENCE — IDLE")
        self._set_state("idle")

    def _on_speech_started(self) -> None:
        if self._current_state == "processing":
            self._set_state("speaking")
            logger.info("◈ SPEAKING...")
            self._speaking_started_at = time.monotonic()

    def _on_agent_response(self, response: str) -> None:
        # Arrives before _speaking_done: the speech stream only ends after chat() returns.
        tools = set(self._agent.last_tools)
        self._follow_up = (
            not (tools and tools <= _COMMAND_TOOLS) and "assistant.end_conversation" not in tools
        )
        self._pending_dictation = "assistant.start_dictation" in tools
        self._pending_switch = (
            "openai" if "assistant.use_openai" in tools
            else "ollama" if "assistant.use_local" in tools
            else None
        )
        self.response_ready.emit(response)

    def _on_speaking_done(self) -> None:
        if self._interrupt_requested:
            self._interrupt_requested = False
            return
        if self._current_state not in ("processing", "speaking"):
            return   # an error or dismissal already moved on
        if self._pending_switch:
            provider, self._pending_switch = self._pending_switch, None
            logger.info("◈ SWITCH → %s", provider)
            self.provider_switch_requested.emit(provider)
            self._set_state("idle")
            return
        if self._pending_dictation:
            self._pending_dictation = False
            self._dictating = True
            logger.info("◈ DICTATION ON — say 'stop dictation' to finish")
            play_chime()
            self._start_listening(start_timeout=DICTATION_START_TIMEOUT)
            return
        if not self._follow_up:
            logger.info("◈ DONE — IDLE")
            self._set_state("idle")
            return
        self._start_listening(start_timeout=FOLLOWUP_START_TIMEOUT)

    def _on_agent_error(self, message: str) -> None:
        self.agent_error.emit(message)
        logger.error("✗ ERROR   %s", message)
        self._set_state("idle")
        QThreadPool.globalInstance().start(_Task(lambda: speak_sync(_ERROR_REPLY, cache=True)))
