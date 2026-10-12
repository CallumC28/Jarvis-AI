from __future__ import annotations

from typing import Annotated

from semantic_kernel.functions import kernel_function

from home_agent.memory.json_store import JsonStore
from home_agent.memory.models import AgentState, TodoItem


class PersonalOrganizerPlugin:
    """Simple personal organizer tools: notes + todo list."""

    def __init__(self, store: JsonStore) -> None:
        self._store = store

    def _read(self) -> AgentState:
        return self._store.load()

    def _write(self, state: AgentState) -> None:
        self._store.save(state)

    # ── Notes ─────────────────────────────────────────────────────────────────

    @kernel_function(name="add_note", description="Save a short note for the user.")
    def add_note(self, text: Annotated[str, "The note text to store"]) -> str:
        text = text.strip()
        if not text:
            return "Note text cannot be empty."
        state = self._read()
        self._write(state.model_copy(update={"notes": [*state.notes, text]}))
        return "Note saved."

    @kernel_function(name="list_notes", description="List all saved notes.")
    def list_notes(self) -> str:
        notes = self._read().notes
        if not notes:
            return "No notes saved."
        return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(notes))

    @kernel_function(name="delete_note", description="Delete a specific note by its number.")
    def delete_note(self, number: Annotated[int, "The note number (1-based) to delete"]) -> str:
        state = self._read()
        idx = number - 1
        if idx < 0 or idx >= len(state.notes):
            return f"Invalid note number: {number}"
        updated = [*state.notes[:idx], *state.notes[idx + 1:]]
        self._write(state.model_copy(update={"notes": updated}))
        return f"Deleted note #{number}."

    @kernel_function(name="clear_notes", description="Delete all saved notes.")
    def clear_notes(self) -> str:
        state = self._read()
        self._write(state.model_copy(update={"notes": []}))
        return "All notes cleared."

    # ── Todos ─────────────────────────────────────────────────────────────────

    @kernel_function(name="add_todo", description="Add a todo item to the user's todo list.")
    def add_todo(self, task: Annotated[str, "The todo item to add"]) -> str:
        task = task.strip()
        if not task:
            return "Todo task cannot be empty."
        state = self._read()
        self._write(state.model_copy(update={"todos": [*state.todos, TodoItem(task=task)]}))
        return "Todo added."

    @kernel_function(name="list_todos", description="List the user's todo items.")
    def list_todos(self) -> str:
        todos = self._read().todos
        if not todos:
            return "No todos yet."
        return "\n".join(
            f"{i}. {'[x]' if t.done else '[ ]'} {t.task}"
            for i, t in enumerate(todos, start=1)
        )

    @kernel_function(
        name="complete_todo",
        description="Mark a todo item as complete by its number.",
    )
    def complete_todo(
        self, number: Annotated[int, "The todo number (1-based) to mark complete"]
    ) -> str:
        state = self._read()
        idx = number - 1
        if idx < 0 or idx >= len(state.todos):
            return f"Invalid todo number: {number}"
        updated = [
            t.model_copy(update={"done": True}) if i == idx else t
            for i, t in enumerate(state.todos)
        ]
        self._write(state.model_copy(update={"todos": updated}))
        return f"Marked todo #{number} complete."

    @kernel_function(
        name="delete_todo",
        description="Delete a specific todo item by its number.",
    )
    def delete_todo(self, number: Annotated[int, "The todo number (1-based) to delete"]) -> str:
        state = self._read()
        idx = number - 1
        if idx < 0 or idx >= len(state.todos):
            return f"Invalid todo number: {number}"
        updated = [*state.todos[:idx], *state.todos[idx + 1:]]
        self._write(state.model_copy(update={"todos": updated}))
        return f"Deleted todo #{number}."

    @kernel_function(name="clear_todos", description="Delete all todo items.")
    def clear_todos(self) -> str:
        state = self._read()
        self._write(state.model_copy(update={"todos": []}))
        return "Todo list cleared."
