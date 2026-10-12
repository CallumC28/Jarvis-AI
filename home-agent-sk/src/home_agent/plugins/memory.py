from __future__ import annotations

import logging
from typing import Annotated

from semantic_kernel.functions import kernel_function

from home_agent.memory.json_store import JsonStore
from home_agent.memory.recall import Recall

logger = logging.getLogger(__name__)


class MemoryPlugin:
    """Long-term facts about the user, and recall of past conversations by meaning."""

    def __init__(self, store: JsonStore, recall: Recall) -> None:
        self._store = store
        self._recall = recall

    @kernel_function(
        name="remember_fact",
        description=(
            "Permanently remember a lasting fact about the user: preferences, people in their "
            "life, projects, routines. Call it unprompted when they mention one. Not for "
            "one-off notes or tasks."
        ),
    )
    def remember_fact(self, fact: Annotated[str, "The fact, as a short sentence"]) -> str:
        fact = fact.strip()
        if not fact:
            return "Fact cannot be empty."
        state = self._store.load()
        if fact in state.facts:
            return "Already known."
        self._store.save(state.model_copy(update={"facts": [*state.facts, fact]}))
        return "Remembered."

    @kernel_function(
        name="forget_fact",
        description="Forget a remembered fact by its number (as listed in your instructions).",
    )
    def forget_fact(self, number: Annotated[int, "The fact number (1-based)"]) -> str:
        state = self._store.load()
        idx = number - 1
        if idx < 0 or idx >= len(state.facts):
            return f"Invalid fact number: {number}"
        self._store.save(
            state.model_copy(update={"facts": [*state.facts[:idx], *state.facts[idx + 1:]]})
        )
        return f"Forgot fact #{number}."

    @kernel_function(
        name="recall",
        description=(
            "Search every past conversation and saved note by meaning. Use when the user "
            "refers to something from before that isn't in the current conversation "
            "('what did I say about…', 'that restaurant we talked about last week')."
        ),
    )
    def recall(self, query: Annotated[str, "What to look for, in plain words"]) -> str:
        try:
            notes = [f"Note: {n}" for n in self._store.load().notes]
            hits = self._recall.search(query, extra=notes)
        except Exception as exc:
            logger.exception("Memory search failed.")
            return f"Memory search unavailable: {exc}"
        return "\n\n".join(hits) if hits else "Nothing relevant found."
