from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from home_agent.core.errors import StoreCorruptedError, StoreError
from home_agent.memory.models import AgentState

@dataclass(frozen=True)
class JsonStore:
    path: Path

    def load(self) -> AgentState:
        if not self.path.exists():
            return AgentState()
        try:
            raw_data = json.loads(self.path.read_text(encoding="utf-8"))
            return AgentState.model_validate(raw_data)
        except json.JSONDecodeError as exc:
            raise StoreCorruptedError(f"State file is not valid JSON: {self.path}") from exc
        except ValidationError as exc:
            raise StoreCorruptedError(f"State file failed schema validation: {self.path}") from exc
        except OSError as exc:
            raise StoreError(f"Failed reading state file: {self.path}") from exc

    def save(self, data: AgentState) -> None:
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = json.dumps(data.model_dump(mode="json"), indent=2, ensure_ascii=False)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise StoreError(f"Failed writing state file: {self.path}") from exc
