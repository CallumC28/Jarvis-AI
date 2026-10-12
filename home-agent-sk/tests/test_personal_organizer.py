from __future__ import annotations

from pathlib import Path

import pytest

from home_agent.memory.json_store import JsonStore
from home_agent.plugins.personal_organiser import PersonalOrganizerPlugin


@pytest.fixture
def plugin(tmp_path: Path) -> PersonalOrganizerPlugin:
    return PersonalOrganizerPlugin(JsonStore(tmp_path / "state.json"))


# ── Todos ─────────────────────────────────────────────────────────────────────

def test_add_and_list_todo(plugin: PersonalOrganizerPlugin) -> None:
    assert plugin.add_todo("Buy milk") == "Todo added."
    listing = plugin.list_todos()
    assert "Buy milk" in listing
    assert "[ ]" in listing


def test_complete_todo(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_todo("Buy milk")
    assert plugin.complete_todo(1) == "Marked todo #1 complete."
    assert "[x]" in plugin.list_todos()


def test_complete_todo_invalid_number(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_todo("Buy milk")
    result = plugin.complete_todo(99)
    assert "Invalid" in result


def test_delete_todo_by_number(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_todo("Buy milk")
    plugin.add_todo("Walk dog")
    assert plugin.delete_todo(1) == "Deleted todo #1."
    listing = plugin.list_todos()
    assert "Buy milk" not in listing
    assert "Walk dog" in listing


def test_delete_todo_invalid_number(plugin: PersonalOrganizerPlugin) -> None:
    result = plugin.delete_todo(5)
    assert "Invalid" in result


def test_clear_todos(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_todo("Buy milk")
    plugin.add_todo("Walk dog")
    assert plugin.clear_todos() == "Todo list cleared."
    assert plugin.list_todos() == "No todos yet."


def test_list_todos_empty(plugin: PersonalOrganizerPlugin) -> None:
    assert plugin.list_todos() == "No todos yet."


def test_add_todo_empty_string(plugin: PersonalOrganizerPlugin) -> None:
    result = plugin.add_todo("   ")
    assert "cannot be empty" in result.lower()


# ── Notes ─────────────────────────────────────────────────────────────────────

def test_add_and_list_notes(plugin: PersonalOrganizerPlugin) -> None:
    assert plugin.add_note("Meeting at 3pm") == "Note saved."
    listing = plugin.list_notes()
    assert "Meeting at 3pm" in listing
    assert "1." in listing


def test_add_note_empty_string(plugin: PersonalOrganizerPlugin) -> None:
    result = plugin.add_note("   ")
    assert "cannot be empty" in result.lower()


def test_list_notes_empty(plugin: PersonalOrganizerPlugin) -> None:
    assert plugin.list_notes() == "No notes saved."


def test_delete_note_by_number(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_note("First")
    plugin.add_note("Second")
    assert plugin.delete_note(1) == "Deleted note #1."
    listing = plugin.list_notes()
    assert "First" not in listing
    assert "Second" in listing


def test_delete_note_invalid_number(plugin: PersonalOrganizerPlugin) -> None:
    result = plugin.delete_note(5)
    assert "Invalid" in result


def test_clear_notes(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_note("First")
    plugin.add_note("Second")
    assert plugin.clear_notes() == "All notes cleared."
    assert plugin.list_notes() == "No notes saved."


def test_multiple_todos_ordering_preserved(plugin: PersonalOrganizerPlugin) -> None:
    plugin.add_todo("Alpha")
    plugin.add_todo("Beta")
    plugin.add_todo("Gamma")
    plugin.delete_todo(2)  # delete Beta
    listing = plugin.list_todos()
    assert "Alpha" in listing
    assert "Beta" not in listing
    assert "Gamma" in listing
