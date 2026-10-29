"""Aegis LLM abstraction package."""

from __future__ import annotations

from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.llm.base import LLMProvider
from aegis.llm.errors import LLMNotConfiguredError, LLMRequestError, StructuredOutputError
from aegis.llm.factory import get_llm_provider
from aegis.llm.openai_provider import OpenAIProvider
from aegis.llm.pricing import calculate_cost, load_pricing_registry
from aegis.llm.types import (
    Block,
    LLMResponse,
    Message,
    StopReason,
    StructuredResult,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

__all__ = [
    "AnthropicProvider",
    "Block",
    "LLMNotConfiguredError",
    "LLMProvider",
    "LLMRequestError",
    "LLMResponse",
    "Message",
    "OpenAIProvider",
    "StopReason",
    "StructuredOutputError",
    "StructuredResult",
    "TextBlock",
    "ToolResultBlock",
    "ToolSpec",
    "ToolUseBlock",
    "Usage",
    "calculate_cost",
    "get_llm_provider",
    "load_pricing_registry",
]
