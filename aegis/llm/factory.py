"""Factory for instantiating configured LLMProvider implementations."""

from __future__ import annotations

import os
from typing import Optional

from aegis.config import Settings, get_settings
from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.llm.base import LLMProvider
from aegis.llm.errors import LLMNotConfiguredError
from aegis.llm.openai_provider import OpenAIProvider


def get_llm_provider(settings: Optional[Settings] = None) -> LLMProvider:
    """Select and instantiate the configured LLMProvider based on settings.

    Raises LLMNotConfiguredError if the provider is unset, model is missing,
    or the required provider API key is not present in the environment.
    """
    cfg = settings or get_settings()

    if not cfg.provider:
        raise LLMNotConfiguredError("No LLM provider configured. Set AEGIS_PROVIDER in environment or settings.")

    if not cfg.model:
        raise LLMNotConfiguredError("No LLM model configured. Set AEGIS_MODEL in environment or settings.")

    if cfg.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError("ANTHROPIC_API_KEY environment variable is missing or empty.")
        return AnthropicProvider(model=cfg.model, api_key=api_key, settings=cfg)

    elif cfg.provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMNotConfiguredError("OPENAI_API_KEY environment variable is missing or empty.")
        return OpenAIProvider(model=cfg.model, api_key=api_key, settings=cfg)

    raise LLMNotConfiguredError(f"Unsupported LLM provider '{cfg.provider}'. Must be 'anthropic' or 'openai'.")
