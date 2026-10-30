"""Fake LLM implementation conforming to LLMProvider protocol for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from aegis.llm.types import (
    LLMResponse,
    Message,
    StructuredResult,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

T = TypeVar("T")


def fake_text(
    text: str,
    model: str = "fake-model",
    input_tokens: int = 10,
    output_tokens: int = 10,
    latency_ms: float = 50.0,
    cost_usd: Optional[float] = None,
) -> LLMResponse:
    """Helper to construct an LLMResponse containing plain text."""
    return LLMResponse(
        text=text,
        tool_calls=[],
        stop_reason="end_turn",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def fake_tool_call(
    name: str,
    args: Dict[str, Any],
    tool_call_id: str = "call-1",
    model: str = "fake-model",
    input_tokens: int = 10,
    output_tokens: int = 10,
    latency_ms: float = 50.0,
    cost_usd: Optional[float] = None,
) -> LLMResponse:
    """Helper to construct an LLMResponse containing a tool call."""
    return LLMResponse(
        text=None,
        tool_calls=[ToolUseBlock(id=tool_call_id, name=name, arguments=args)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


def fake_structured(
    value: Any,
    input_tokens: int = 10,
    output_tokens: int = 10,
    model: str = "fake-model",
    latency_ms: float = 50.0,
    attempts: int = 1,
    cost_usd: Optional[float] = None,
) -> StructuredResult:
    """Helper to construct a StructuredResult."""
    return StructuredResult(
        value=value,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        model=model,
        latency_ms=latency_ms,
        attempts=attempts,
        cost_usd=cost_usd,
    )


class FakeLLM:
    """Fake LLM provider implementing the aegis.llm.base.LLMProvider protocol.

    Records all calls and returns queued responses.
    Raises AssertionError when the response queue is empty.
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        model: str = "fake-model",
    ) -> None:
        self.model = model
        self.responses: List[Any] = list(responses or [])
        self.calls: List[Dict[str, Any]] = []
        self.requests = self.calls  # Alias for backward compatibility

    def queue_response(self, response: Any) -> None:
        """Queue a response to be returned by next call."""
        self.responses.append(response)

    def chat(
        self,
        messages: List[Message],
        system: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        self.calls.append({
            "method": "chat",
            "messages": messages,
            "system": system,
            "tools": tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if not self.responses:
            raise AssertionError("FakeLLM: no scripted response left")

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, LLMResponse):
            return resp
        if isinstance(resp, str):
            return fake_text(resp, model=self.model)
        raise AssertionError(f"FakeLLM: queued item is {type(resp)}, expected LLMResponse or str")

    def structured(
        self,
        messages: List[Message],
        schema: Type[T],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> StructuredResult:
        self.calls.append({
            "method": "structured",
            "messages": messages,
            "schema": schema,
            "system": system,
            "max_tokens": max_tokens,
        })
        if not self.responses:
            raise AssertionError("FakeLLM: no scripted response left")

        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        if isinstance(resp, StructuredResult):
            return resp
        if isinstance(resp, (schema, BaseModel)):
            return fake_structured(resp, model=self.model)
        if isinstance(resp, dict):
            val = schema.model_validate(resp) if hasattr(schema, "model_validate") else resp
            return fake_structured(val, model=self.model)

        raise AssertionError(f"FakeLLM: queued item is {type(resp)}, expected StructuredResult or {schema}")
