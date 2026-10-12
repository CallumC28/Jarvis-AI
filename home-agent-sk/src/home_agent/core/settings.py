from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from home_agent.core.errors import ConfigurationError

load_dotenv()

# One home for notes, history and models, wherever the app is launched from.
DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "Jarvis"


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ai_provider: Literal["openai", "ollama"] = "ollama"
    openai_api_key: str = ""
    openai_chat_model_id: str = ""
    openai_image_model_id: str = "gpt-image-1"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_chat_model_id: str = Field(default="qwen3.5:9b", min_length=1)
    ollama_enable_tools: bool = True
    ollama_thinking: bool = False   # thinking models take ~50 s to reply by voice; off by default
    tavily_api_key: str = ""
    code_folders: list[Path] = []
    gitstat_dir: Path | None = None   # ReactJS-UI/gitstat-ui; enables the git plugin

    @model_validator(mode="after")
    def _validate_provider_settings(self) -> AppSettings:
        if self.ai_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when AI_PROVIDER=openai.")
            if not self.openai_chat_model_id:
                raise ValueError("OPENAI_CHAT_MODEL_ID is required when AI_PROVIDER=openai.")
        return self

    @staticmethod
    def load() -> AppSettings:
        try:
            return AppSettings(
                ai_provider=os.environ.get("AI_PROVIDER", "ollama").lower(),
                openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
                openai_chat_model_id=os.environ.get("OPENAI_CHAT_MODEL_ID", ""),
                openai_image_model_id=os.environ.get("OPENAI_IMAGE_MODEL_ID", "gpt-image-1"),
                ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                ollama_chat_model_id=os.environ.get("OLLAMA_CHAT_MODEL_ID", "qwen3.5:9b"),
                ollama_enable_tools=(
                    os.environ.get("OLLAMA_ENABLE_TOOLS", "true").strip().lower()
                    in {"1", "true", "yes", "on"}
                ),
                ollama_thinking=(
                    os.environ.get("OLLAMA_THINKING", "false").strip().lower()
                    in {"1", "true", "yes", "on"}
                ),
                tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
                code_folders=[
                    Path(p.strip())
                    for p in os.environ.get("CODE_FOLDERS", "").split(os.pathsep)
                    if p.strip()
                ],
                gitstat_dir=Path(d) if (d := os.environ.get("GITSTAT_DIR", "").strip()) else None,
            )
        except ValidationError as exc:
            raise ConfigurationError(
                "Invalid application configuration in environment variables."
            ) from exc
