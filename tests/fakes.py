"""Fake LLM implementation for testing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from aegis.models.llm_provider import BaseLLMProvider

T = TypeVar("T", bound=BaseModel)


class FakeLLM(BaseLLMProvider):
    """
    Fake LLM provider for tests.
    Records all requests and returns queued responses.
    Raises AssertionError when no response is left in the queue.
    """

    def __init__(
        self,
        responses: Optional[List[Any]] = None,
        model: str = "fake-model",
    ):
        super().__init__(model=model)
        self.responses: List[Any] = list(responses or [])
        self.requests: List[Dict[str, Any]] = []

    def queue_response(self, response: Any) -> None:
        """Queue a response to be returned by next call."""
        self.responses.append(response)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        self.requests.append({
            "method": "generate_text",
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return str(resp)
        raise AssertionError("FakeLLM: no scripted response left")

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        self.requests.append({
            "method": "generate_structured",
            "prompt": prompt,
            "schema": schema,
            "system_prompt": system_prompt,
            "temperature": temperature,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, schema):
                return resp
            if isinstance(resp, dict):
                return schema.model_validate(resp)
            return resp

        raise AssertionError("FakeLLM: no scripted response left")
