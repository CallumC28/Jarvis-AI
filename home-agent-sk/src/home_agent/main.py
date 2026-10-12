from __future__ import annotations

import asyncio
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Callable

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from home_agent.core.agent import HomeAgent
from home_agent.core.composition import create_home_agent, unload_ollama
from home_agent.core.errors import ConfigurationError, HomeAgentError
from home_agent.core.settings import AppSettings

console = Console()
logger = logging.getLogger(__name__)

ACCENT = "cyan"
DIM = "grey50"
AGENT_COLOUR = "bright_white"
USER_COLOUR = "cyan"
ERROR_COLOUR = "red"
PROGRESS_COLOUR = "grey42"
SHOW_DEBUG_OUTPUT = True #NOTE - set to True to show progress messages and enable console status spinner
SWITCH_PHRASE = "switch agent"
USAGE_LOGGER_NAME = "semantic_kernel.connectors.ai.open_ai.services.open_ai_handler"


class UsageProviderFilter(logging.Filter):
    def __init__(self, get_provider: Callable[[], str]) -> None:
        super().__init__()
        self._get_provider = get_provider

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != USAGE_LOGGER_NAME:
            return True
        if not isinstance(record.msg, str):
            return True
        if "OpenAI usage:" not in record.msg:
            return True

        provider = self._get_provider()
        label = "Ollama usage:" if provider == "ollama" else "OpenAI usage:"
        record.msg = record.msg.replace("OpenAI usage:", label)
        return True


def _header() -> None:
    title = Text()
    title.append("HOME", style=f"bold {AGENT_COLOUR}")
    title.append(" AGENT", style=f"bold {ACCENT}")
    title.append("  //  ", style=DIM)
    title.append("Semantic Kernel Interface", style=DIM)

    ts = Text(datetime.now().strftime("%d-%m-%Y  %H:%M"), style=DIM, justify="right")

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right", no_wrap=True)
    grid.add_row(title, ts)

    console.print(
        Panel(
            grid,
            border_style=ACCENT,
            padding=(0, 1),
            box=box.HEAVY_HEAD,
        )
    )
    console.print()


def _divider(label: str = "") -> None:
    console.print(Rule(label, style=DIM, characters="-"))


def on_progress(msg: str) -> None:
    if not SHOW_DEBUG_OUTPUT:
        return
    console.print(f"  [dim]>[/dim]  [{PROGRESS_COLOUR}]{msg}[/{PROGRESS_COLOUR}]")


def _render_user(text: str) -> None:
    label_style = Style(color="#000000", bgcolor=USER_COLOUR, bold=True)
    label = Text("  YOU  ", style=label_style)
    body = Text(f"  {text}", style="white")
    console.print()
    console.print(Text.assemble(label, body))


def _render_agent(text: str) -> None:
    label_style = Style(color="#000000", bgcolor=AGENT_COLOUR, bold=True)
    label = Text("  AGENT  ", style=label_style)
    body = Markdown(text or " ")
    console.print()
    console.print(label)
    console.print(
        Panel(
            body,
            border_style=DIM,
            padding=(0, 2),
            box=box.SIMPLE_HEAVY,
        )
    )


def _render_error(message: str) -> None:
    console.print(
        Panel(
            f"[bold]Error:[/bold]  {message}",
            border_style=ERROR_COLOUR,
            box=box.SIMPLE_HEAVY,
            padding=(0, 2),
            style=ERROR_COLOUR,
        )
    )


async def _amain() -> None:
    console.clear()
    _header()
    console.print(
        f" [bold green]System online.[/bold green] "
        f"[{DIM}]Type [bold white]exit[/bold white] to terminate.[/{DIM}]\n"
    )
    _divider()

    agents: dict[str, HomeAgent] = {}
    for provider in ("ollama", "openai"):
        try:
            agents[provider] = create_home_agent(
                provider_override=provider,
                on_progress=on_progress,
            )
        except ConfigurationError:
            logger.info("Skipping %s agent due to missing configuration.", provider)

    if not agents:
        raise ConfigurationError(
            "No chat providers are configured. Set Ollama or OpenAI settings in .env."
        )

    active_provider = "ollama" if "ollama" in agents else next(iter(agents))
    setattr(_amain, "_active_provider", active_provider)
    console.print(
        f"[{DIM}]Active agent:[/{DIM}] [bold]{active_provider}[/bold]  "
        f"[{DIM}]Type \"{SWITCH_PHRASE}\" to toggle.[/{DIM}]"
    )

    while True:
        try:
            user_text = console.input(f"\n[bold {USER_COLOUR}]>[/bold {USER_COLOUR}] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            console.print(f"\n[{DIM}]Session terminated.[/{DIM}]\n")
            break
        normalized = user_text.strip().lower()
        if SWITCH_PHRASE in normalized:
            other_provider = "openai" if active_provider == "ollama" else "ollama"
            if other_provider not in agents:
                _render_error(
                    f"Cannot switch: {other_provider} agent is not configured."
                )
            else:
                active_provider = other_provider
                setattr(_amain, "_active_provider", active_provider)
                _render_agent(f"Switched to {active_provider} agent.")
            _divider()
            continue

        _render_user(user_text)
        console.print()

        try:
            status_ctx = (
                console.status(
                    f"[{DIM}]processing ...[/{DIM}]",
                    spinner="line",
                    spinner_style=ACCENT,
                )
                if SHOW_DEBUG_OUTPUT
                else nullcontext()
            )
            with status_ctx:
                response = await agents[active_provider].chat(user_text)
            _render_agent(response)
        except HomeAgentError as exc:
            _render_error(str(exc))
        except Exception:
            logger.exception("Unexpected error in interactive loop.")
            _render_error("Unexpected failure. Check logs for details.")

        _divider()


def run() -> None:
    level = logging.INFO if SHOW_DEBUG_OUTPUT else logging.WARNING
    logging.basicConfig(level=level)
    provider_filter = UsageProviderFilter(
        get_provider=lambda: getattr(_amain, "_active_provider", "ollama")
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(provider_filter)
    if not SHOW_DEBUG_OUTPUT:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("semantic_kernel").setLevel(logging.WARNING)
    try:
        asyncio.run(_amain())
    finally:
        unload_ollama(AppSettings.load())   # give the VRAM back


if __name__ == "__main__":
    run()
