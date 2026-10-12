from __future__ import annotations

import pytest

from home_agent.core.errors import ConfigurationError
from home_agent.core.settings import AppSettings


def test_default_provider_is_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    settings = AppSettings.load()
    assert settings.ai_provider == "ollama"


def test_ollama_loads_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_CHAT_MODEL_ID", "llama3")
    settings = AppSettings.load()
    assert settings.ollama_chat_model_id == "llama3"


def test_openai_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_CHAT_MODEL_ID", "gpt-4o")
    with pytest.raises(ConfigurationError):
        AppSettings.load()


def test_openai_requires_chat_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_CHAT_MODEL_ID", raising=False)
    with pytest.raises(ConfigurationError):
        AppSettings.load()


def test_openai_valid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_CHAT_MODEL_ID", "gpt-4o")
    settings = AppSettings.load()
    assert settings.ai_provider == "openai"
    assert settings.openai_api_key == "sk-test"


def test_ollama_enable_tools_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("true", "True", "1", "yes", "on"):
        monkeypatch.setenv("OLLAMA_ENABLE_TOOLS", value)
        assert AppSettings.load().ollama_enable_tools is True


def test_ollama_enable_tools_falsy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("false", "False", "0", "no", "off"):
        monkeypatch.setenv("OLLAMA_ENABLE_TOOLS", value)
        assert AppSettings.load().ollama_enable_tools is False


def test_tavily_api_key_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    settings = AppSettings.load()
    assert settings.tavily_api_key == ""
