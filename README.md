# Jarvis — a voice-first AI home assistant

A personal assistant you talk to like someone in the room. Say "Hey Jarvis", ask for something, and it controls your Windows PC, looks at your screen, remembers what you tell it, searches the web, reads your code and types for you. It runs fully local on [Ollama](https://ollama.com/) or in the cloud on OpenAI, and you can switch between them mid-conversation.

Built in Python on [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel), with a PySide6 desktop app (animated orb, voice pipeline, live debug console) and a Rich terminal CLI.

**Stack:** Python 3.10+ · Semantic Kernel · OpenAI / Ollama · PySide6 · faster-whisper · openWakeWord · Kokoro TTS (ONNX) · Pydantic · pytest

---

## Highlights

- **Hands-free voice, fully local**: wake word, speech recognition and the voice all run on your machine. Replies stream sentence by sentence, so speech starts about 0.3 s after the model's first sentence.
- **Understands intent, not phrases**: there are no trigger words to memorise. "Right, I'm off", "use the cloud one" and "take this down for me" all work, because the model chooses the tool.
- **Controls your PC**: opens, closes, switches, moves and snaps apps, and handles media keys, volume, the clipboard and deep links into Steam, Spotify and Windows Settings.
- **Sees your screen**: "what's this error?" takes a screenshot and sends it to the model as an image.
- **Remembers you**: facts you share go into every prompt, and past conversations can be searched by meaning.
- **Local or cloud, live**: Ollama and OpenAI are interchangeable at runtime, and each keeps its own history.

---

## Engineering notes

A few problems that took more than a library call to solve:

- **Latency.** Thinking models were taking about 56 s per reply. Turning off reasoning for voice, keeping the model resident in VRAM with a warm-up on launch, and streaming sentences into TTS while the next one renders brings a warm reply to under a second.
- **Prompt caching.** The time and the active window title go at the start of each *user* message, not in the system prompt. That keeps the prompt prefix stable, so both Ollama and OpenAI can reuse their cache.
- **Unbounded conversations.** Nothing is truncated. Once history passes 40 messages, the oldest are folded into a running summary, cut at a user message so tool calls stay paired with their results. The summariser also extracts facts the model forgot to save.
- **Vision from tools.** Tool results are text only, so when a turn takes a screenshot the agent attaches it as an image message and calls the model again. Only the latest screenshot keeps its image in history.
- **Voice robustness.** Silero VAD means silence never reaches Whisper. "Hey Jarvis" can interrupt mid-reply, with a confidence threshold that stops Jarvis's own voice through the speakers from triggering it. The wake-word model is reset after every hit, which stops phantom re-triggers.
- **Observability.** A Semantic Kernel function-invocation filter logs every tool call and result to the debug console automatically, so plugins need no logging code of their own.
- **Safety.** The code reader is read-only and confined to configured folders, and it refuses `.env` and key files. Deep links are restricted to an allowlist of URI schemes. Jarvis's own windows are excluded from screen captures.

---

## Features

### Voice pipeline

- **Wake word**: "Hey Jarvis" (openWakeWord, ONNX) plays a chime and starts listening straight away.
- **Speech-to-text**: faster-whisper `large-v3-turbo` on CUDA, falling back to CPU `small`. A turn ends after 1.5 s of silence or 10 s of speech.
- **Text-to-speech**: a local Kokoro voice (`bm_lewis`, British English). URLs and markdown are stripped before speaking.
- **Natural follow-ups**: after each reply Jarvis listens again for 4 s. Keep talking to continue, or stay quiet and it goes idle.
- **Commands end cleanly**: if a turn only ran commands (open an app, change the volume…), Jarvis confirms and goes idle without waiting for a follow-up.
- **Dictation**: "take this down" pastes everything you say, word for word, into the active window until you say "stop dictation" or go quiet for 15 s. This path skips the LLM.

### Desktop app

- **Animated orb**: a frameless, translucent overlay. It never steals focus, and it swells with your voice while listening and with Jarvis's voice while speaking.
- **Provider theming**: a blue/amber palette for Ollama and a green one for OpenAI.
- **Debug console**: a cinematic log of every state change and tool call (see below).
- **Image viewer**, **tray menu** (pause listening, switch provider, quit) and **F2** to type instead of talk.

### Plugins

| Plugin                     | Ask things like                                                        | What it does                                                                                     |
| -------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **pc**                     | "open Spotify", "put this on the left", "volume to 30"                 | Apps (Store apps too), window moving/snapping across monitors, media keys, volume, clipboard     |
| **pc.open_uri**            | "open my Steam library", "open display settings"                       | Deep links, allowlisted to `steam`, `spotify`, `ms-settings`, `discord`, `mailto`, `vscode`, `ms-windows-store` |
| **screenshot**             | "what's on my screen?", "look at my second monitor"                    | Captures a monitor and shows it to the model                                                     |
| **memory**                 | "remember I take my coffee black", "what did I say about the car?"     | Remember and forget facts; semantic recall of past conversations                                  |
| **organizer**              | "add milk to my todos", "tick off the first one"                       | Notes and todos, persisted to disk                                                               |
| **browser**                | "open YouTube", "go to GitHub"                                         | Opens Chrome; common names map to URLs, anything else becomes a Google search                    |
| **desktop**                | "make a folder on my desktop called Taxes 2026"                        | Creates desktop folders with Windows-safe names                                                  |
| **assistant**              | "that's all", "take this down", "use the cloud one"                    | Ends the conversation, starts dictation, switches provider                                       |
| **web** _(Tavily)_         | "search the web for…"                                                  | Only when `TAVILY_API_KEY` is set, and only when you explicitly ask                              |
| **code** _(read-only)_     | "what does the transcription thread do?"                               | Lists, reads and searches files in `CODE_FOLDERS`                                                |
| **git** _(read-only)_      | "anything I haven't pushed?"                                           | Repo status via a local gitstat dashboard (`GITSTAT_DIR`)                                        |
| **images** _(OpenAI only)_ | "generate an image of…", "make this photo look like a painting"        | Generates or edits images and opens them in the viewer                                           |

One request can run several actions: "open Spotify, play something and turn it up to 40".

---

## Architecture

```
home-agent-sk/src/home_agent/
  core/      kernel + provider setup, the agent loop (streaming, compaction, persistence), settings
  plugins/   one Semantic Kernel plugin per capability (see table above)
  memory/    JSON state store, conversation log, embedding-based recall
  voice/     wake word → VAD + Whisper → agent → streamed Kokoro TTS, as a Qt state machine
  ui/        PySide6 orb window, debug console, image viewer, tray
  main.py    Rich CLI
```

The voice pipeline is a Qt state machine: `IDLE → LISTENING → PROCESSING → SPEAKING → (follow-up) LISTENING → IDLE`. The heavy work (Whisper, the agent, TTS) runs on worker threads, and results come back to the UI thread through queued signals.

More detailed design notes are in [CLAUDE.md](CLAUDE.md).

---

## Setup

**You need:** Windows 10/11, Python 3.10+, [Ollama](https://ollama.com/), a microphone and speakers, and Google Chrome. An NVIDIA GPU (CUDA 12) is optional but makes Whisper much faster.

### Quick start

1. **Clone the repo and create a virtual environment:**
   ```bash
   git clone https://github.com/CallumC28/Jarvis.git
   cd Jarvis
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This also installs the app and its `home-agent` / `home-agent-desktop` commands. For GPU Whisper or the dev tools, uncomment the optional lines in `requirements.txt`.
3. **Pull a chat model and the embedding model:**
   ```bash
   ollama pull qwen3.5:32b         # or any tool-calling model, 9B+ recommended
   ollama pull nomic-embed-text    # used by memory recall
   ```
4. **Configure:** create `home-agent-sk/.env` (see below). Setting `AI_PROVIDER=ollama` and the Ollama lines is enough to start.
5. **Run:** `home-agent-desktop`, then say "Hey Jarvis". The Whisper, wake-word and voice models download automatically on first run.

### Configuration

`home-agent-sk/.env`:

```env
AI_PROVIDER=ollama                 # or openai

OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_CHAT_MODEL_ID=qwen3.5:32b   # needs function-calling support; 9B+ recommended
OLLAMA_ENABLE_TOOLS=true
OLLAMA_THINKING=false              # true lets thinking models reason first (much slower)

OPENAI_API_KEY=sk-...              # optional; also enables image generation
OPENAI_CHAT_MODEL_ID=gpt-4o
OPENAI_IMAGE_MODEL_ID=gpt-image-1

TAVILY_API_KEY=tvly-...            # optional: web search
CODE_FOLDERS=C:/path/a;C:/path/b   # optional: read-only code plugin (use / not \)
GITSTAT_DIR=C:/path/to/gitstat-ui  # optional: git plugin
```

User data (notes, facts, history, the recall index and the voice model) is stored in `%LOCALAPPDATA%\Jarvis`, never in the repo.

## Running

```bash
home-agent-desktop    # desktop app with the voice pipeline
home-agent            # terminal UI
```

| Action                 | How                                                            |
| ---------------------- | -------------------------------------------------------------- |
| Wake Jarvis            | Say "Hey Jarvis"                                               |
| Continue the chat      | Keep talking within ~4 s of the reply                          |
| Interrupt              | Say "Hey Jarvis" while it's speaking                           |
| Dismiss                | Say you're done in any words, or stay quiet                    |
| Dictate                | "Take this down" … "stop dictation"                            |
| Switch provider        | "Switch to OpenAI" / "go back to local", or the tray menu      |
| Type instead of talk   | F2                                                             |

### Debug console

```
◈ STATE   LISTENING
◈ USER    "open spotify and turn it up to 40"
→ INVOKE  pc → open_app  [ name="spotify" ]
✓ RESULT  Opened Spotify.
→ INVOKE  pc → set_volume  [ level="40" ]
✓ RESULT  Volume set to 40%.
◈ RESPONSE  "Spotify's open, volume's at 40."
```

---

## Adding a plugin

1. Create a class in `plugins/` with `@kernel_function` methods.
2. Register it in `core/plugin_registry.py` with `kernel.add_plugin(...)`.
3. Add a tool-use rule to `SYSTEM_PROMPT` in `core/composition.py`. Describe the *intent*, not trigger phrases.

```python
from typing import Annotated
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    @kernel_function(name="get_weather", description="Get current weather for a city.")
    def get_weather(self, city: Annotated[str, "City name"]) -> str:
        return f"Weather for {city}: sunny, 18°C"
```

## Development

```bash
cd home-agent-sk
pytest            # test suite (pytest-asyncio, Qt and audio mocked)
ruff check .      # lint
mypy src          # type-check
```

To train a custom wake word, run `pip install openwakeword[training]`, then `python scripts/train_wake_word.py` from the repo root.
