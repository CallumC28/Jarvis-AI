# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`home-agent-sk` — a personal home assistant ("Jarvis") built on Microsoft Semantic Kernel. Supports Ollama (local) and OpenAI as interchangeable chat providers. Has two UIs: a Rich-based CLI and a PySide6 desktop app.

All source lives under `home-agent-sk/`. Run commands from that directory unless noted.

## Setup

```bash
cd home-agent-sk
pip install -e .[desktop,voice]    # recommended for full desktop use
pip install -e .[dev]              # core + dev tools (add alongside the above)
pip install -e .[desktop]          # PySide6 only
pip install -e .[voice]            # faster-whisper, openwakeword, kokoro-onnx, sounddevice
ollama pull nomic-embed-text      # embeddings for memory.recall (used even with AI_PROVIDER=openai)
pip install -e .[gpu]              # CUDA 12 cuBLAS/cuDNN wheels → Whisper large-v3-turbo on GPU
```

Create a `.env` file (loaded automatically by `python-dotenv` at module import time via `settings.py`):

```
AI_PROVIDER=ollama             # or openai
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL_ID=qwen3.5:32b
OLLAMA_ENABLE_TOOLS=true
OLLAMA_THINKING=false          # default; true makes thinking models reason first (~50 s replies)

OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL_ID=gpt-4o
OPENAI_IMAGE_MODEL_ID=gpt-image-1   # optional, default shown

TAVILY_API_KEY=tvly-...        # optional – enables web_search plugin
CODE_FOLDERS=C:/path/a;C:/path/b  # optional – enables read-only code plugin (use / not \)
GITSTAT_DIR=C:/path/to/gitstat-ui  # optional – enables git plugin
```

> **Data location**: notes, facts, history, the conversation log and the Kokoro voice model live in
> `%LOCALAPPDATA%\Jarvis` (`settings.DATA_DIR`), whatever directory the app is launched from.

> **Ollama tool use**: `OLLAMA_ENABLE_TOOLS=true` requires a model that supports function calling. Small 4B models may struggle with multi-turn context and tool invocation; 9B+ recommended.

## Commands

```bash
# Run
home-agent                     # CLI (Rich terminal UI)
home-agent-desktop             # Desktop UI (PySide6)

# Dev
pytest                         # all tests (asyncio_mode = auto)
pytest tests/test_json_store.py          # single file
pytest tests/test_json_store.py::test_name  # single test
ruff check .                   # lint
ruff format .                  # format
mypy src                       # type-check (strict=false, disallow_untyped_defs=true)
```

## Architecture

```
core/
  settings.py        AppSettings — reads env vars; load_dotenv() runs at module import (once only);
                     DATA_DIR = %LOCALAPPDATA%\Jarvis
  kernel_factory.py  build_kernel() — creates SK Kernel; registers _tool_invocation_filter which
                     auto-logs every plugin call to home_agent.tools logger (→ INVOKE / ✓ RESULT)
  plugin_registry.py register_default_plugins() — returns RegisteredPlugins(NamedTuple) with
                     .tavily and .image fields; accepts on_image and on_progress callbacks
  composition.py     create_home_agent() — top-level factory; holds SYSTEM_PROMPT; loads persisted
                     conversation history via _load_history(); saves per-provider history files
  agent.py           HomeAgent dataclass — async chat(user_text, image_paths=None, on_sentence=None);
                     streams the reply and hands each finished sentence to on_sentence (pop_sentences;
                     <think> blocks are stripped). Each turn: system message rebuilt (system_prompt()
                     + summary), user message prefixed with turn_context() ("[time · looking at:
                     window]"), turn appended to log_path, then _compact() folds the oldest messages
                     into .summary once there are > 40 (the summariser also lists new facts,
                     saved to fact_store — backstop for when the model skips remember_fact).
                     .last_tools lists "plugin.function" calls from the last turn. Image attachments are added as ImageContent
                     and forwarded to image_plugin.set_latest_user_images
  errors.py          HomeAgentError, ConfigurationError, StoreError, StoreCorruptedError, ChatServiceError

memory/
  json_store.py      JsonStore — atomic write using a .tmp swap
  models.py          AgentState (Pydantic) — notes, todos: list[TodoItem(task, done)], facts
  recall.py          append_turn() logs every turn to conversation_log.jsonl; Recall.search() does
                     semantic search over the log + notes with Ollama nomic-embed-text vectors cached
                     in memory_index.json (brute-force cosine)

plugins/
  personal_organiser.py  Notes/todos — always registered; functions: add/list/delete_note,
                         clear_notes, add/list/complete/delete_todo, clear_todos
  desktop_folder.py      Creates desktop folders with Windows-safe name sanitisation — always registered
  screenshot.py          Screen capture via mss (downscaled to 1920 px with Pillow); resolves
                         natural-language monitor refs — always registered; calls on_image(path)
                         and appends to .captures, which HomeAgent sends to the model as an image
  code_reader.py         Read-only list_files/read_file/search_code, confined to CODE_FOLDERS;
                         refuses .env/keys; skips node_modules/.git/.venv — only when CODE_FOLDERS set
  git_status.py          repo_status / open_dashboard — only when GITSTAT_DIR is set. Reads the
                         gitstat dashboard's /api/repos (127.0.0.1:8765, a Next.js app in
                         ReactJS-UI/gitstat-ui); if it's down, starts it hidden (node → next start,
                         needs `npm run build` once) and polls up to 30 s. The server outlives Jarvis.
                         Read-only on purpose: commit/push stay behind the dashboard's confirm dialogs.
                         Spoken repo names match via pc._best_match ("leaf lounge" → Leaf-Lounge)
  browser.py             Opens Google Chrome — always registered; _KNOWN_SITES maps common names
                         to URLs; falls back to Google search for unrecognised phrases
  pc.py                  Windows control via pywin32 — always registered: open_app (Get-StartApps →
                         shell:AppsFolder, so Store apps work), close/switch_to_app, move_window
                         (reuses screenshot._resolve_monitor), arrange_window (max/min/snap left/right),
                         minimise_all, open_uri (allowlisted schemes only: steam, spotify, ms-settings,
                         discord, mailto, vscode, ms-windows-store — never widen to "any"), media
                         (media keys), set_volume (pycaw; "30", "+10", "-10"), read_clipboard,
                         copy_to_clipboard, paste_text. foreground_window()
                         skips Jarvis's own windows; _best_match ranks exact > whole word > prefix >
                         contains > fuzzy. paste_text() is also used by dictation mode
  assistant.py           end_conversation / start_dictation / use_openai / use_local — no-ops;
                         VoicePipeline reads agent.last_tools and acts — always registered
  memory.py              remember_fact / forget_fact (facts are always in the system prompt) and
                         recall (semantic search of past conversations) — always registered
  tavily_search.py       Web search via Tavily API — only when TAVILY_API_KEY is set; fires
                         on_progress callbacks mid-search ("Calling web_search...", etc.)
  image_generation.py    OpenAI image gen + edit — only when AI_PROVIDER=openai; _save_b64_image()
                         is the shared helper for both generate and edit paths

voice/
  pipeline.py    VoicePipeline — Qt state machine: IDLE→GREETING→LISTENING→PROCESSING→SPEAKING→IDLE
                 After each response it silently re-listens (FOLLOWUP_START_TIMEOUT = 4 s to start
                 talking) — no "Is that all, sir?"; silence → IDLE.
                 Intents are understood by the model, not phrase-matched: assistant.end_conversation /
                 start_dictation / use_openai / use_local tool calls are acted on after the reply is
                 spoken (_on_agent_response → _on_speaking_done). Model-free fast paths remain: exact
                 dismissals (no bare yes/no — those answer questions) and a whole-sentence
                 "switch to OpenAI/local" (_SWITCH) so you can switch when the model is down.
                 Interrupt: "Hey Jarvis" while speaking stops TTS at any point after a 1 s guard,
                 but only if the wake score >= _INTERRUPT_THRESHOLD (0.8) so our own voice through
                 the speakers can't self-trigger. Tune that constant if interrupts miss/misfire.
                 Streaming: _on_transcription_ready starts the agent thread (sentences → a Queue) and
                 speak_stream together; state goes SPEAKING on the first audio. Each response gets
                 its own stop Event so interrupting one can't silence the "Yes sir?" that follows.
                 Commands: if every tool a turn used is in _COMMAND_TOOLS (open/close apps,
                 media, volume, open_uri…), Jarvis goes idle after speaking — no follow-up listen.
                 Dictation: once started, every transcription is pasted verbatim into the active
                 window (no LLM) until one ends with "stop dictation/typing" (text before it is still
                 pasted) or 15 s of silence.
  wake_word.py   WakeWordThread — always-running; uses openWakeWord + ONNX; model: "hey_jarvis"
                 (threshold 0.5); downloaded automatically on first run. After a hit it MUST
                 model.reset() and drain the stream, or the model's buffer re-fires every pause
                 (phantom wakes that interrupt replies). Also publishes level.mic every 80 ms
  level.py       level.mic / level.voice — current loudness 0–1 (-60…-15 dBFS) for the orb
  transcription.py TranscriptionThread — faster-whisper large-v3-turbo on CUDA (falls back to CPU
                   "small" if CUDA fails); Whisper is a module-level
                   singleton loaded once at pipeline start via preload_whisper() — NOT reloaded
                   per-thread (reloading caused 6 s delay and missed speech); speech detection
                   via Silero VAD (_is_speech, bundled with faster-whisper, VAD_THRESHOLD 0.5); waits start_timeout (START_TIMEOUT 4 s, follow-up 4 s) for speech to
                   begin, then ends the turn after SILENCE_DURATION (1.5 s) of silence or a 10 s cap
                   from first speech (still transcribed); pure silence is never sent to Whisper
  tts.py         Local Kokoro voice (kokoro-onnx, "bm_lewis", en-gb; model auto-downloads to
                 DATA_DIR/models on first run) played through sounddevice, one OutputStream per
                 playback — STEREO: mono was inaudible on 7.1 headsets; publishes
                 level.voice per block. speak_sync(cache=True keeps fixed phrases in memory); speak_stream()
                 renders the next sentence while the current one plays; play_chime(); wake word
                 plays a chime and listens immediately; _clean_for_speech() strips URLs/markdown

ui/
  desktop_app.py     JarvisApp — top-level orchestrator; wires all windows, tray, and pipeline.
                     _ImageBridge(QObject) marshals on_image callbacks from worker threads to the
                     main thread via a queued Signal. _switch_provider() rebuilds agent +
                     VoicePipeline on provider change. _cleanup_old_temp_images() runs at startup
                     to delete screenshot/generated image files older than 24 hours.
  jarvis_window.py   JarvisWindow — frameless, translucent, WA_ShowWithoutActivating (never steals
                     focus from the user's window); animated orb canvas that swells with level.mic
                     while listening and level.voice while speaking (VOICE_SWELL); slides in/out from
                     bottom-right; F2 toggles debug text input. set_theme("openai"|"ollama") switches
                     the full colour palette and draws a provider label at the bottom of the canvas.
  image_viewer.py    ImageViewerWindow — shows generated images and screenshots; no auto-close;
                     show_image(path) scales to fit 1200×900.
  terminal_window.py TerminalWindow — cinematic debug console; stays open for the whole conversation
                     (transcript accumulates across follow-ups) and auto-closes 3 s after the
                     pipeline goes idle (2× for errors). Filters to home_agent.* loggers only (_HomeAgentLogFilter). Format is
                     message-only — no level/logger prefix. Highlight rules colour ◈ events (cyan),
                     → INVOKE (amber), ✓ RESULT (green), ✗ ERROR (red).
  tray.py            JarvisTray — system tray icon; menu: toggle text input, switch provider, quit
main.py              Rich CLI; both Ollama and OpenAI agents instantiated simultaneously
```

### Key design decisions

**Dual-provider at runtime**: the user switches providers by saying "switch to OpenAI agent" / "switch to local agent" (intercepted in `pipeline._on_transcription_ready` before reaching the agent) or via the tray menu. `_switch_provider(target)` rebuilds the agent and pipeline. Each provider keeps its own `ChatHistory` and its own persisted history file.

**Plugin loading**: `ImageGenerationPlugin` is only registered when `AI_PROVIDER=openai`; `TavilySearchPlugin` is only registered when `TAVILY_API_KEY` is set. The other plugins are always registered (code only when CODE_FOLDERS is set, git only when GITSTAT_DIR is set). `register_default_plugins()` returns `RegisteredPlugins(NamedTuple)` — access fields as `.tavily` and `.image`, not by position.

**Tool invocation filter**: `kernel_factory.py` registers `_tool_invocation_filter` on every kernel via `kernel.add_filter("function_invocation", ...)` — never `function_invocation_filters.append()`, which breaks every tool call. This SK hook fires automatically before and after every plugin call, logging to `home_agent.tools`. This is why the debug terminal shows tool calls without any code in the plugins themselves.

**`on_image` callback**: both `ScreenshotPlugin` and `ImageGenerationPlugin` accept `on_image: Callable[[Path], None] | None`. When fired from the agent worker thread, `JarvisApp._ImageBridge` marshals it to the main thread via a queued Qt signal before calling `ImageViewerWindow.show_image()`.

**`on_progress` callback**: wired in `desktop_app.py` as `_on_agent_progress`, which logs `  ↳ msg` to the terminal. `TavilySearchPlugin` uses it to emit mid-search status. Pass it through `create_home_agent` — both on initial creation and on provider switch.

**Tool-use**: `FunctionChoiceBehavior.Auto()` is set for OpenAI and for Ollama when `OLLAMA_ENABLE_TOOLS=true`. Without this, the kernel won't invoke plugins automatically.

**SYSTEM_PROMPT** (in `core/composition.py`): optimised for TTS output — no bullet points, no markdown, concise. Key rules: web search only when user explicitly asks; browser plugin for all "open site" requests; do not re-search when user refers to a prior result by order; image tools constrained to specific sizes/quality values.

**Screen vision**: tool results are text-only, so when a turn produced a screenshot `HomeAgent.chat()` appends it as a user `ImageContent` message and calls the model once more; only the latest screenshot keeps its image in history. Jarvis's own windows are excluded from captures via `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` (`_exclude_from_capture` in `desktop_app.py`).

**Speed**: `OLLAMA_THINKING=false` (default) sends `extra_body={"reasoning_effort": "none"}` — qwen3.8:27b went from 56 s to 0.6 s. Replies stream sentence by sentence into Kokoro, so speech starts ~0.3 s after the model's first sentence. After a screenshot the first pass is a blind guess, so `_complete` stops emitting sentences once a capture appears.

**Chat history**: `HomeAgent.chat()` stores only the model's response text in `self.history` — the raw Tavily dump is appended to the return value for display only, intentionally excluded from history to avoid polluting the context window.

**Conversation persistence**: `HomeAgent._persist_history()` saves `{"summary", "messages"}` to `DATA_DIR/history_{provider}.json` after every turn using an atomic tmp-file swap. Nothing is truncated: `_compact()` summarises old messages (cutting at a user message so tool calls stay paired with results). `_load_history()` in `composition.py` restores both (and still reads the old bare-list format). Per-provider files mean switching providers loads the correct prior context; `conversation_log.jsonl` is shared and keeps every turn for `memory.recall`.

**Per-turn context**: the time and active window title go at the start of the user message, not the system prompt, so the prompt prefix stays stable and Ollama/OpenAI prompt caching keeps working. Remembered facts go in the system prompt (they rarely change).

**State persistence**: `JsonStore` saves `AgentState` (notes, todos, facts) to `DATA_DIR/agent_state.json`. Always use `model_copy(update={...})` when modifying Pydantic models — never mutate list fields in place.

**Whisper singleton**: `voice/transcription.py` exposes a module-level `_whisper_model` loaded once via `preload_whisper()`. `VoicePipeline.start()` fires `preload_whisper` in a `QThreadPool` worker so the model is warm before the first wake word fires. Never reinstantiate `WhisperModel` per-thread.

**Follow-up loop**: after each agent response, `_on_speaking_done` goes straight back to listening with `start_timeout=FOLLOWUP_START_TIMEOUT` — no spoken prompt, so conversation flows like talking to someone in the room. No speech within that window → idle. Dismissal phrases (normalised: punctuation and a trailing "sir"/"jarvis" stripped) go idle at any point. `SYSTEM_PROMPT`'s "HOW YOU TALK" section drives the conversational tone (short volleys, carry the thread, never end with "anything else?").

**Debug terminal log format**: all `home_agent.*` log messages render as plain `%(message)s`. Use cinematic prefixes: `◈ STATE`, `◈ USER "..."`, `◈ RESPONSE "..."` for pipeline events. The SK filter handles tool call lines automatically — don't add manual `logger.info` inside plugin functions for entry/exit.

**Adding a plugin**: create a class in `plugins/` with `@kernel_function`-decorated methods, register in `core/plugin_registry.py` via `kernel.add_plugin(...)`, add a tool-use rule to `SYSTEM_PROMPT` in `core/composition.py`. The SK filter will automatically log invocations — no manual logging needed inside the plugin.

### Desktop controls

| Action                  | How                                                       |
| ----------------------- | --------------------------------------------------------- |
| Wake voice pipeline     | Say "Hey Jarvis"                                          |
| Interrupt mid-response  | Say "Hey Jarvis" while speaking                           |
| Dismiss and go idle     | Say you're done, any wording ("right, I'm off"), or stay quiet |
| Switch to OpenAI        | Ask in any words ("switch to OpenAI", "use the cloud one") |
| Switch to local         | Ask in any words ("go back to local")                     |
| Dictate into a window   | "Take this down" / "type this for me" … "stop dictation"  |
| Toggle debug text input | F2 or tray menu                                           |
| Switch provider         | Tray menu → "Switch Provider"                             |

### Scripts

```bash
# Run from repo root (not from home-agent-sk/)
python scripts/train_wake_word.py   # train a custom "jarvis" wake word (requires openwakeword[training])
```

Output: `home-agent-sk/models/jarvis.onnx` — update `WAKE_WORD_MODEL` in `voice/wake_word.py` to use it.
