"""Unit tests for aegis/config.py, provider selection, and 503 behavior."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aegis.api.main import app
from aegis.config import Settings, reset_settings
from aegis.models.llm_provider import (
    AnthropicProvider,
    OpenAIProvider,
    get_llm_provider,
)


def test_settings_defaults():
    s = Settings()
    assert s.agent_step_limit == 8
    assert s.tool_output_limit == 8000
    assert s.sandbox_timeout == 20.0
    assert s.dataset_max_bytes == 20 * 1024 * 1024
    assert s.dataset_max_rows == 200_000
    assert s.dataset_max_cols == 100
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.embedding_dim == 384


def test_settings_provider_validation():
    # Valid providers
    s_anthropic = Settings(provider="anthropic")
    assert s_anthropic.provider == "anthropic"

    s_openai = Settings(provider="openai")
    assert s_openai.provider == "openai"

    # Invalid provider rejected
    with pytest.raises(ValidationError):
        Settings(provider="invalid_provider")


def test_get_llm_provider_from_settings():
    # Unconfigured provider raises error
    s_none = Settings(provider=None)
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        get_llm_provider(settings=s_none)

    # Anthropic
    s_anth = Settings(provider="anthropic", model="claude-custom")
    provider_anth = get_llm_provider(settings=s_anth)
    assert isinstance(provider_anth, AnthropicProvider)
    assert provider_anth.model == "claude-custom"

    # OpenAI
    s_oai = Settings(provider="openai", model="gpt-4o-mini")
    provider_oai = get_llm_provider(settings=s_oai)
    assert isinstance(provider_oai, OpenAIProvider)
    assert provider_oai.model == "gpt-4o-mini"


def test_submit_task_returns_503_when_no_provider():
    reset_settings(Settings(provider=None))
    client = TestClient(app)

    resp = client.post(
        "/api/tasks",
        json={"query": "Test task with no provider"},
    )
    assert resp.status_code == 503
    assert "No LLM provider configured" in resp.json()["detail"]


def test_submit_task_succeeds_when_provider_configured(monkeypatch):
    from tests.fakes import FakeLLM
    reset_settings(Settings(provider="anthropic"))
    # Patch orchestrator service worker so it uses FakeLLM
    monkeypatch.setattr("aegis.services.orchestrator_service.get_llm_provider", lambda: FakeLLM())

    client = TestClient(app)
    resp = client.post(
        "/api/tasks",
        json={"query": "Test task with provider"},
    )
    assert resp.status_code == 200
    assert "task_id" in resp.json()
    assert resp.json()["status"] == "submitted"
