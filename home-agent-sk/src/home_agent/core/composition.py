from __future__ import annotations

import json
import logging
import threading
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from home_agent.core.agent import HomeAgent
from home_agent.core.errors import StoreError
from home_agent.core.kernel_factory import build_kernel
from home_agent.core.plugin_registry import register_default_plugins
from home_agent.core.settings import DATA_DIR, AppSettings
from home_agent.memory.json_store import JsonStore
from home_agent.memory.recall import Recall, ollama_embedder
from home_agent.plugins.pc import foreground_title

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Jarvis — the AI who lives in this house and talks with the user out loud, the way Jarvis talks with Tony Stark: another person in the room, not a service desk. Loyal, quick, dry-witted, warm under a British reserve. Polite but never obsequious; witty but never frivolous.

HOW YOU TALK:
- This is a spoken back-and-forth. Reply the way a sharp colleague would across the room: one or two sentences, then stop and let them talk. They can always ask for more.
- Answer only what was asked. No trivia, background, or "did you know". Don't tack an offer onto the end ("Want me to…?", "Shall I…?") — if they want more, they'll ask.
- React to what they actually said before (or instead of) reciting facts. A short "Ah, that one." or "Fair point." beats a formal preamble.
- Carry the thread. Treat each message as the next line of an ongoing conversation: resolve "it", "that", "the second one", "what about tomorrow" from what was just said, and never re-introduce yourself or restate the question.
- Ask a quick question back only when it genuinely moves things along — never as a ritual. Never end with "Anything else?", "Let me know if…", or "Is there anything more I can help with?".
- If you misheard or they correct you ("no, I said…"), take it in stride — "Ah, Perth." — and answer the corrected version.
- Humour is understated, a raised eyebrow in voice form; deliver bad news with constructive dry wit. Anticipate the obvious next need and mention it briefly when it's useful.
- Say "sir" occasionally, the way a person would — not in every reply.
- Only when genuinely uncertain, say so plainly; never invent results, links, or actions.

Each user message starts with a bracketed line giving the current date and time and the window the user is looking at. Use it silently when it helps ("this error" probably lives in that window); never read it out.

Tool-use rules:
- People don't speak in commands. Work out what they want from what they mean, not the exact words: "it's too loud" is pc.set_volume, "get rid of this" closes the window they're in, "I'm off" ends the conversation, "pop that in my notes" is organizer.add_note. The examples in these rules are illustrations, not trigger phrases. If the intent really could go two ways, ask in a few words rather than guess.
- Never say you did something unless you called the tool for it in this turn and it succeeded. If they ask for something again, do it again — never answer "it's already open". If a tool fails, say so.
- One request can hold several actions ("open Spotify and set the volume to 30"): call every tool it needs, in order, then confirm it all in one short line.
- For a pure command (open, close, play, pause, volume, move a window), confirm in a few words — "Done.", "Steam's up." — and nothing more; Jarvis leaves straight after.
- For "open X", use pc.open_app unless X is clearly a website; if no app is found, open it with browser.open_website.
- To reach a page inside an app (the Steam store, a Spotify search, a Windows settings page), use pc.open_uri. Music controls are pc.media; volume is pc.set_volume.
- Use pc tools to close, switch to, move between monitors, snap or minimise windows. "This"/"here" with no app named means the window they're in — pass an empty name.
- Anything about what they've copied ("what I copied", "this text I grabbed", "fix what's on my clipboard") means pc.read_clipboard. To rewrite and paste: read the clipboard, rewrite it, then pc.paste_text the result — confirm in a few words; don't read the new text aloud.
- Whenever the user mentions something lasting about themselves (preferences, people, projects, routines), call memory.remember_fact — once per fact, before you reply. Saying "noted" without calling it means you will forget. memory.forget_fact when they ask you to forget something.
- When they refer back to something not in the current conversation ("what did I say about…", "that place we talked about"), use memory.recall before answering.
- Use organizer tools when they want something jotted down, listed, ticked off or cleared, whether they call it a note, a task, a reminder or "my list".
- Use desktop.create_desktop_folder when they want a new folder made.
- Use images.generate_image when the user asks to create an image from text.
- Use images.edit_last_uploaded_image when the user asks to edit an uploaded image (for example adding meme captions).
- For image tools, use only supported size values: auto, 1024x1024, 1536x1024, 1024x1536.
- For image tools, use only supported quality values: auto, low, medium, high.
- For meme/caption edits, explicitly ask for no crop/zoom and require all text to stay fully inside the image bounds.
- Use browser.open_website when they want a website or browser open, however they put it ("pull up YouTube", "take me to the BBC"). Never use web search for this.
- When the user refers to a previous result, link, or item by order ("the first", "the second", "that one", "the previous link"), look at the conversation history and use it directly — do not search again.
- Use screenshot.capture_screen when the user asks what is on their screen, or refers to something they are looking at ("this error", "what I'm looking at"). The screenshot is then shown to you as an image — describe or use what you actually see.
- Use the code tools (code.list_files, code.read_file, code.search_code) when the user asks about their code or projects. Search or read before answering; never guess file contents. You can only read code, not change it — if asked to edit, say so and describe the change instead.
- When talking about code, explain it conversationally; do not read code aloud line by line unless asked.
- Anything about the state of their repos ("git status", "anything I haven't pushed?", "what's changed in Leaf Lounge?") is git.repo_status — pass the repo name when they mean one. Say which repos need attention and why in a sentence or two; don't list clean repos or read file paths out unless asked. When they want to see it ("show me", "pull up the dashboard"), use git.open_dashboard. You can't commit or push — that's done in the dashboard.
- Use web search only when the user asks you to look something up, however they word it ("search for", "look up", "check online", "google it", "see what the internet says"). A plain question is not a request to search: answer from what you know, and if it needs live information you don't have, say so briefly.
- For web_search, prefer max_results=5.
- assistant.end_conversation when they're done talking for now, however they say it. assistant.start_dictation when they want what they say next typed out — including "I want to write an email/message, take this down"; never offer to take it down yourself. assistant.use_openai / assistant.use_local when they want you on the other model.
- If a tool call fails, explain briefly and offer the next best option.

Spoken style:
- Everything you write is read aloud. Write as you speak: contractions, plain words, no markdown, bullet points, headers, or lists unless explicitly asked.
- Don't read out URLs, file paths, or code line by line; describe them.
- No filler ("Certainly!", "Great question!", "Of course!") and no sources unless asked.
- For multi-step requests, do them in a sensible order and give a one-line summary.
- Use stored notes and todos for continuity when relevant.
"""


# A cold load of the 27B model is ~35 s plus ~15 s of GPU warm-up, so keep it in VRAM while
# Jarvis runs (Ollama's default unloads it after 5 idle minutes) and unload it on quit.
OLLAMA_KEEP_ALIVE = -1   # forever; e.g. "30m" if you need the VRAM back while Jarvis idles


def _ollama_generate(settings: AppSettings, payload: dict) -> None:
    base = settings.ollama_base_url.removesuffix("/").removesuffix("/v1")
    request = urllib.request.Request(
        f"{base}/api/generate",
        data=json.dumps({"model": settings.ollama_chat_model_id, **payload}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=180).read()
    except OSError as exc:
        logger.warning("Ollama %s failed: %s", payload, exc)


def warm_ollama(settings: AppSettings) -> None:
    """Load the model and run one token in the background, so the first question is quick."""
    payload = {"prompt": "hi", "options": {"num_predict": 1}, "keep_alive": OLLAMA_KEEP_ALIVE}
    threading.Thread(
        target=_ollama_generate, args=(settings, payload), name="ollama-warm", daemon=True
    ).start()


def unload_ollama(settings: AppSettings) -> None:
    """Free the VRAM OLLAMA_KEEP_ALIVE held. A no-op if the model isn't loaded."""
    _ollama_generate(settings, {"keep_alive": 0})


def _load_history(history: ChatHistory, path: Path) -> str:
    """Restore persisted conversation turns into an existing ChatHistory; returns the summary."""
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):   # pre-summary format: a bare message list
            data = {"summary": "", "messages": data}
        if "history" in data:        # full format, tool calls included
            restored = ChatHistory.restore_chat_history(json.dumps(data["history"]))
            history.messages.extend(restored.messages)
        for msg in data.get("messages", []):   # older text-only format
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history.add_user_message(content)
            elif role == "assistant":
                history.add_assistant_message(content)
        return str(data.get("summary", ""))
    except Exception:
        logger.warning("Conversation history file is corrupted — starting fresh.")
        return ""


def _turn_context() -> str:
    now = datetime.now().strftime("%A %d %B %Y, %H:%M")
    title = foreground_title()
    looking = f" · looking at: {title}" if title else ""
    # Repeated on every message because a prompt rule alone doesn't hold: the local model
    # answered "it's already open" to a repeated request instead of calling the tool.
    return f"[{now}{looking} · any action must be a tool call made now, even if done before]"


def _system_prompt(store: JsonStore) -> Callable[[], str]:
    def build() -> str:
        try:
            facts = store.load().facts
        except StoreError:
            logger.warning("Could not read remembered facts.", exc_info=True)
            facts = []
        if not facts:
            return SYSTEM_PROMPT
        numbered = "\n".join(f"{i}. {f}" for i, f in enumerate(facts, start=1))
        heading = "What you know about the user (numbered, for forget_fact):"
        return f"{SYSTEM_PROMPT}\n{heading}\n{numbered}"

    return build


def create_home_agent(
    provider_override: Literal["openai", "ollama"] | None = None,
    data_dir: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_image: Callable[[object], None] | None = None,
) -> HomeAgent:
    app_settings = AppSettings.load()
    if provider_override is not None and provider_override != app_settings.ai_provider:
        app_settings = AppSettings.model_validate(
            {**app_settings.model_dump(), "ai_provider": provider_override}
        )

    kernel = build_kernel(app_settings)

    base = data_dir or DATA_DIR
    store = JsonStore(base / "agent_state.json")
    log_path = base / "conversation_log.jsonl"
    embed = ollama_embedder(app_settings.ollama_base_url)
    recall = Recall(log_path, base / "memory_index.json", embed)
    plugins = register_default_plugins(
        kernel=kernel,
        settings=app_settings,
        store=store,
        recall=recall,
        on_progress=on_progress,
        on_image=on_image,
    )

    exec_settings = OpenAIChatPromptExecutionSettings()
    if app_settings.ai_provider == "openai" or (
        app_settings.ai_provider == "ollama" and app_settings.ollama_enable_tools
    ):
        exec_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    if app_settings.ai_provider == "ollama":
        exec_settings.extra_body = {"keep_alive": OLLAMA_KEEP_ALIVE}
        if not app_settings.ollama_thinking:
            exec_settings.extra_body["reasoning_effort"] = "none"   # Ollama's switch for thinking
        warm_ollama(app_settings)

    history = ChatHistory()
    history_path = base / f"history_{app_settings.ai_provider}.json"
    summary = _load_history(history, history_path)

    return HomeAgent(
        kernel=kernel,
        history=history,
        settings=exec_settings,
        tavily_plugin=plugins.tavily,
        image_plugin=plugins.image,
        screenshot_plugin=plugins.screenshot,
        history_path=history_path,
        system_prompt=_system_prompt(store),
        turn_context=_turn_context,
        log_path=log_path,
        summary=summary,
        fact_store=store,
    )
