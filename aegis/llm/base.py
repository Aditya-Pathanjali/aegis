"""Protocol definition for Aegis LLM providers."""

from __future__ import annotations

from typing import List, Optional, Protocol, Type, TypeVar, runtime_checkable

from aegis.llm.types import LLMResponse, Message, StructuredResult, ToolSpec

T = TypeVar("T")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for provider-neutral LLM execution."""

    def chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Execute a conversational completion with optional tool calls."""
        ...

    def structured(
        self,
        messages: List[Message],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        """Generate schema-constrained structured output validated against a Pydantic model."""
        ...

