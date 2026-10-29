"""Exceptions for the LLM abstraction layer."""

from __future__ import annotations


class LLMNotConfiguredError(Exception):
    """Raised when an LLM provider is requested without required configuration or credentials."""

    pass


class LLMRequestError(Exception):
    """Raised on non-retryable 4xx errors from an LLM provider. Message includes status and provider."""

    def __init__(self, provider: str, status_code: int, message: str) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"LLM request error ({provider}): HTTP {status_code} - {message}")


class StructuredOutputError(Exception):
    """Raised when structured output parsing/validation fails after retry attempts.

    Holds the raw model output in `raw_text` as an attribute; callers must never log it.
    """

    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text

