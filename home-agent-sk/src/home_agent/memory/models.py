from __future__ import annotations

from pydantic import BaseModel, Field


class TodoItem(BaseModel):
    task: str = Field(min_length=1)
    done: bool = False


class AgentState(BaseModel):
    notes: list[str] = Field(default_factory=list)
    todos: list[TodoItem] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)   # about the user; always in context
