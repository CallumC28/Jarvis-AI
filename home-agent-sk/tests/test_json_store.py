from __future__ import annotations

from pathlib import Path

import pytest

from home_agent.core.errors import StoreCorruptedError
from home_agent.memory.json_store import JsonStore
from home_agent.memory.models import AgentState, TodoItem


def test_json_store_roundtrip(tmp_path: Path) -> None:
    store = JsonStore(tmp_path / "state.json")
    state = AgentState(notes=["note"], todos=[TodoItem(task="task", done=False)])

    store.save(state)
    loaded = store.load()

    assert loaded == state


def test_json_store_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{invalid json", encoding="utf-8")
    store = JsonStore(path)

    with pytest.raises(StoreCorruptedError):
        store.load()
