"""LLM Provider abstraction supporting OpenAI and Anthropic."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from aegis.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract Base Class for LLM providers."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        """Generate free-form text from prompt."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate structured output adhering to a Pydantic schema."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(model=model or "gpt-4o", api_key=api_key or os.getenv("OPENAI_API_KEY"))
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            self.client = None
            self._init_error = str(e)

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        if not self.client:
            raise RuntimeError(f"OpenAI client not initialized: {getattr(self, '_init_error', 'No API key')}")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        if not self.client:
            raise RuntimeError(f"OpenAI client not initialized: {getattr(self, '_init_error', 'No API key')}")

        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instructions = (
            f"You are a structured data extractor and generator. "
            f"Always output ONLY valid JSON adhering strictly to this schema:\n{schema_json}\n"
            f"Do not include Markdown backticks, fences, or explanation. Output pure JSON."
        )
        messages = [{"role": "system", "content": f"{system_prompt or ''}\n\n{system_instructions}"}]
        messages.append({"role": "user", "content": prompt})

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=schema,
                temperature=temperature,
            )
            parsed = completion.choices[0].message.parsed
            if parsed:
                return parsed
        except Exception:
            pass

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return schema.model_validate_json(content)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__(model=model or "claude-3-5-sonnet-20241022", api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("Anthropic API key is not configured.")
        import httpx
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        with httpx.Client(timeout=60.0) as client:
            resp = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"You are a structured output generator. Output ONLY a valid JSON object matching this schema:\n{schema_json}\n"
            "Do not include markdown fences, backticks, or explanatory text before or after the JSON."
        )
        full_system = f"{system_prompt or ''}\n\n{instruction}"
        raw = self.generate_text(prompt=prompt, system_prompt=full_system, temperature=temperature)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
        return schema.model_validate_json(cleaned)


def get_llm_provider(settings: Optional[Settings] = None) -> BaseLLMProvider:
    """
    Factory function to instantiate the configured LLM provider.
    Selects only from settings. Fails loudly if no provider is configured.
    """
    if settings is None:
        settings = get_settings()

    if not settings.provider:
        raise RuntimeError(
            "No LLM provider configured. Set AEGIS_PROVIDER to 'anthropic' or 'openai'."
        )

    provider = settings.provider.lower()
    if provider == "openai":
        return OpenAIProvider(model=settings.model)
    elif provider == "anthropic":
        return AnthropicProvider(model=settings.model)
    else:
        raise ValueError(
            f"Unsupported LLM provider '{settings.provider}'. Must be 'anthropic' or 'openai'."
        )
