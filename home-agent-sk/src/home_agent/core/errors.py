from __future__ import annotations


class HomeAgentError(Exception):
    """Base exception for home agent failures."""


class ConfigurationError(HomeAgentError):
    """Raised when required application configuration is invalid or missing."""


class StoreError(HomeAgentError):
    """Raised for persistent store read/write failures."""


class StoreCorruptedError(StoreError):
    """Raised when persisted state cannot be decoded into valid domain models."""


class ChatServiceError(HomeAgentError):
    """Raised when chat completion service calls fail."""
