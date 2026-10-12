from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import (
    AuthorRole,
    ChatHistory,
    ChatMessageContent,
    FunctionCallContent,
    ImageContent,
    TextContent,
)

from home_agent.core.errors import ChatServiceError, StoreError
from home_agent.core.kernel_factory import CHAT_SERVICE_ID
from home_agent.memory.recall import append_turn

if TYPE_CHECKING:
    from home_agent.memory.json_store import JsonStore
    from home_agent.plugins.image_generation import ImageGenerationPlugin
    from home_agent.plugins.screenshot import ScreenshotPlugin
    from home_agent.plugins.tavily_search import TavilySearchPlugin

logger = logging.getLogger(__name__)

_COMPACT_AT = 40      # messages in context before the oldest are folded into the summary
_KEEP_RECENT = 20     # messages kept word-for-word after folding
_SCREENSHOT_NOTE = "[Screenshot you just captured of the user's screen]"
_SUMMARY_PROMPT = (
    "Update the running summary of a spoken conversation between the user and Jarvis, their "
    "assistant. Keep names, decisions, preferences, open threads and anything the user may "
    "refer back to; drop small talk. Plain prose, under 200 words.\n\n"
    "Then write a line saying FACTS: and under it any lasting facts about the user stated in "
    "the conversation (preferences, people, projects, routines) that aren't already known, one "
    "per line starting with '- '. Write 'none' if there are none.\n\n"
    "Already known:\n{facts}\n\n"
    "Current summary:\n{summary}\n\nConversation to fold in:\n{transcript}"
)

_MIN_SENTENCE = 20    # chars; shorter pieces ("Ah.", "Mr.") wait to join the next sentence
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop reasoning blocks (and an unclosed one at the end) so they're never spoken or stored."""
    return _THINK.sub("", text).split("<think>")[0]


def pop_sentences(buffer: str) -> tuple[list[str], str]:
    """Split finished sentences off streamed text so speech can start before the reply ends.

    Returns (sentences, unfinished tail to keep buffering).
    """
    buffer = _THINK.sub("", buffer)
    open_at = buffer.find("<think>")
    head, held = (buffer[:open_at], buffer[open_at:]) if open_at != -1 else (buffer, "")
    *done, tail = _SENTENCE_END.split(head)
    sentences: list[str] = []
    pending = ""
    for piece in done:
        pending = f"{pending} {piece}".strip()
        if len(pending) >= _MIN_SENTENCE:
            sentences.append(pending)
            pending = ""
    if pending:
        tail = f"{pending} {tail}"
    return sentences, tail + held


@dataclass
class HomeAgent:
    kernel: Kernel
    history: ChatHistory
    settings: OpenAIChatPromptExecutionSettings
    tavily_plugin: TavilySearchPlugin | None = None
    image_plugin: ImageGenerationPlugin | None = None
    screenshot_plugin: ScreenshotPlugin | None = None
    history_path: Path | None = field(default=None)
    system_prompt: Callable[[], str] | None = None   # rebuilt each turn: base prompt + known facts
    turn_context: Callable[[], str] | None = None    # time/active window, prefixed to each message
    log_path: Path | None = None                     # every turn, for memory.recall
    summary: str = ""                                # older conversation, folded down
    fact_store: JsonStore | None = None              # facts the summariser spots are saved here
    last_tools: list[str] = field(default_factory=list)   # "plugin.function" called last turn
    _last_screenshot: ChatMessageContent | None = field(default=None, init=False, repr=False)

    async def chat(
        self,
        user_text: str,
        image_paths: list[Path] | None = None,
        on_sentence: Callable[[str], None] | None = None,
    ) -> str:
        """Run one turn. on_sentence receives each spoken sentence as soon as it is complete."""
        text = user_text.strip()
        images = image_paths or []
        if not text and not images:
            return ""
        self._refresh_system_message()
        prompt = "\n".join(p for p in (self.turn_context() if self.turn_context else "", text) if p)
        if self.image_plugin is not None and images:
            self.image_plugin.set_latest_user_images(images)

        shots_before = len(self.screenshot_plugin.captures) if self.screenshot_plugin else 0
        initial_tavily_call_count = (
            self.tavily_plugin.call_count if self.tavily_plugin is not None else None
        )

        if images:
            message_parts: list[object] = []
            if prompt:
                message_parts.append(TextContent(text=prompt))
            for image_path in images:
                try:
                    message_parts.append(ImageContent.from_image_file(str(image_path)))
                except Exception:
                    logger.exception("Failed to load image for model input: %s", image_path)
            if message_parts:
                self.history.add_user_message(message_parts)
            else:
                self.history.add_user_message(prompt or "Please analyse the attached images.")
        else:
            self.history.add_user_message(prompt)
        turn_start = len(self.history.messages)

        model_response = await self._complete(on_sentence)
        new_shots = self.screenshot_plugin.captures[shots_before:] if self.screenshot_plugin else []
        if new_shots:
            self._attach_screenshot(new_shots[-1])
            model_response = await self._complete(on_sentence)   # one follow-up only — no loops
        self.last_tools = [
            f"{item.plugin_name}.{item.function_name}"
            for m in self.history.messages[turn_start:]
            for item in m.items
            if isinstance(item, FunctionCallContent)
        ]
        self.history.add_assistant_message(model_response)
        if self.log_path is not None:
            append_turn(self.log_path, text, model_response)
        # ponytail: runs before chat() returns, so every ~10th turn the follow-up listen starts
        # a second or two late; move it to a background task if that's noticeable.
        await self._compact()
        self._persist_history()

        display_content = model_response
        if (
            self.tavily_plugin is not None
            and initial_tavily_call_count is not None
            and self.tavily_plugin.call_count > initial_tavily_call_count
        ):
            tool_output = self.tavily_plugin.last_result_text.strip()
            if tool_output:
                display_content = (
                    f"{model_response}\n\nFull web search results:\n{tool_output}"
                    if model_response.strip()
                    else tool_output
                )
        return display_content

    def _shot_count(self) -> int:
        return len(self.screenshot_plugin.captures) if self.screenshot_plugin else 0

    async def _complete(self, on_sentence: Callable[[str], None] | None = None) -> str:
        shots = self._shot_count()
        reply = buffer = ""
        try:
            chat_service = self.kernel.get_service(service_id=CHAT_SERVICE_ID)
            async for chunk in chat_service.get_streaming_chat_message_content(
                chat_history=self.history,
                settings=self.settings,
                kernel=self.kernel,
            ):
                if chunk is None or chunk.role == AuthorRole.TOOL:
                    continue
                piece = chunk.content or ""
                calls_tool = any(isinstance(i, FunctionCallContent) for i in chunk.items)
                if calls_tool and not reply.endswith(" "):
                    piece += " "   # keep the words either side of a tool call apart
                reply += piece
                if on_sentence is None or self._shot_count() != shots:
                    continue   # a screenshot was just taken: this pass is a blind guess
                sentences, buffer = pop_sentences(buffer + piece)
                for sentence in sentences:
                    on_sentence(sentence)
        except Exception as exc:
            logger.exception("Chat completion call failed.")
            raise ChatServiceError("Failed to get a response from the chat service.") from exc
        if on_sentence is not None and self._shot_count() == shots:
            tail = strip_think(buffer).strip()
            if tail:
                on_sentence(tail)
        return strip_think(reply).strip()

    def _refresh_system_message(self) -> None:
        if self.system_prompt is None:
            return
        content = self.system_prompt()
        if self.summary:
            content += f"\n\nSummary of your earlier conversation with the user:\n{self.summary}"
        system = ChatMessageContent(role=AuthorRole.SYSTEM, content=content)
        messages = self.history.messages
        if messages and messages[0].role == AuthorRole.SYSTEM:
            messages[0] = system
        else:
            messages.insert(0, system)

    async def _compact(self) -> None:
        """Fold the oldest messages into a running summary instead of dropping them."""
        messages = self.history.messages
        start = 1 if messages and messages[0].role == AuthorRole.SYSTEM else 0
        if len(messages) - start <= _COMPACT_AT:
            return
        # Cut at a user message so a tool call is never separated from its result.
        cut = next(
            (i for i in range(len(messages) - _KEEP_RECENT, len(messages))
             if messages[i].role == AuthorRole.USER),
            None,
        )
        if cut is None:
            return
        old = messages[start:cut]
        transcript = "\n".join(
            f"{m.role.value}: {m.content}"
            for m in old
            if m.role in (AuthorRole.USER, AuthorRole.ASSISTANT) and m.content
        )
        known = self._known_facts()
        request = ChatHistory()
        request.add_user_message(
            _SUMMARY_PROMPT.format(
                facts="\n".join(f"- {f}" for f in known) or "(none)",
                summary=self.summary or "(none yet)",
                transcript=transcript,
            )
        )
        try:
            chat_service = self.kernel.get_service(service_id=CHAT_SERVICE_ID)
            response = await chat_service.get_chat_message_content(
                chat_history=request,
                settings=OpenAIChatPromptExecutionSettings(extra_body=self.settings.extra_body),
            )
        except Exception:
            logger.warning("Could not summarise old conversation; will retry.", exc_info=True)
            return
        summary, _, facts_part = strip_think(response.content or "").partition("FACTS:")
        self.summary = summary.strip() or self.summary
        self._save_facts(
            [line.strip()[2:].strip() for line in facts_part.splitlines()
             if line.strip().startswith("- ")],
            known,
        )
        if any(m is self._last_screenshot for m in old):
            self._last_screenshot = None
        del messages[start:cut]
        self._refresh_system_message()

    def _known_facts(self) -> list[str]:
        if self.fact_store is None:
            return []
        try:
            return self.fact_store.load().facts
        except StoreError:
            logger.warning("Could not read remembered facts.", exc_info=True)
            return []

    def _save_facts(self, facts: list[str], known: list[str]) -> None:
        """Backstop for remember_fact: the model sometimes says "noted" without calling it."""
        seen = {f.lower() for f in known}
        new = [f for f in facts if f and f.lower() not in seen]
        if not new or self.fact_store is None:
            return
        try:
            state = self.fact_store.load()
            self.fact_store.save(state.model_copy(update={"facts": [*state.facts, *new]}))
            logger.info("◈ REMEMBERED %s", new)
        except StoreError:
            logger.warning("Could not save facts from the summary.", exc_info=True)

    def _attach_screenshot(self, path: Path) -> None:
        """Tool results are text-only, so the pixels reach the model as a follow-up user message.

        Only the latest screenshot keeps its image; older ones are reduced to the note so they
        don't get re-sent (and re-billed) on every later turn.
        """
        if self._last_screenshot is not None:
            placeholder = TextContent(text=f"{_SCREENSHOT_NOTE} (no longer attached)")
            self._last_screenshot.items = [placeholder]
        try:
            image = ImageContent.from_image_file(str(path))
        except Exception:
            logger.exception("Failed to load screenshot for model input: %s", path)
            return
        message = ChatMessageContent(
            role=AuthorRole.USER,
            items=[TextContent(text=f"{_SCREENSHOT_NOTE} Use it to answer my request."), image],
        )
        self.history.add_message(message)
        self._last_screenshot = message

    def _persist_history(self) -> None:
        if self.history_path is None:
            return
        # Tool calls and results are kept: with text alone, a restarted model sees Jarvis saying
        # "Steam's up." with no action behind it — and learns to claim actions without doing them.
        saved = ChatHistory()
        for msg in self.history.messages:
            if msg.role == AuthorRole.SYSTEM:
                continue   # rebuilt every turn
            if any(isinstance(i, ImageContent) for i in msg.items):   # don't store screenshots
                msg = ChatMessageContent(role=msg.role, items=[
                    TextContent(text=f"{_SCREENSHOT_NOTE} (no longer attached)")
                    if isinstance(i, ImageContent) else i
                    for i in msg.items
                ])
            saved.add_message(msg)
        payload = {"summary": self.summary, "history": json.loads(saved.serialize())}
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.history_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(self.history_path)
        except OSError as exc:
            logger.warning("Failed to persist conversation history: %s", exc)
