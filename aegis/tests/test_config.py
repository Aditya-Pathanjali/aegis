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
    assert s.provider is None
    assert s.model is None
    assert s.agent_step_limit == 8
    assert s.tool_output_limit == 8000
    assert s.sandbox_timeout == 20.0
    assert s.dataset_max_bytes == 20 * 1024 * 1024
    assert s.dataset_max_rows == 200_000
    assert s.dataset_max_cols == 100
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.embedding_dim == 384


def test_settings_provider_and_model_validation():
    # Valid provider + model
    s_anthropic = Settings(provider="anthropic", model="claude-sonnet-5-5")
    assert s_anthropic.provider == "anthropic"
    assert s_anthropic.model == "claude-sonnet-5-5"

    s_openai = Settings(provider="openai", model="gpt-4o")
    assert s_openai.provider == "openai"
    assert s_openai.model == "gpt-4o"

    # Provider set without model raises ValidationError
    with pytest.raises(ValidationError, match="model is required"):
        Settings(provider="anthropic", model=None)

    with pytest.raises(ValidationError, match="model is required"):
        Settings(provider="openai", model=None)

    # Invalid provider rejected
    with pytest.raises(ValidationError):
        Settings(provider="invalid_provider", model="some-model")


def test_provider_classes_require_model_without_defaults():
    # Calling provider without model raises TypeError
    with pytest.raises(TypeError):
        OpenAIProvider()  # type: ignore

    with pytest.raises(TypeError):
        AnthropicProvider()  # type: ignore


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
    s_oai = Settings(provider="openai", model="gpt-4o-custom")
    provider_oai = get_llm_provider(settings=s_oai)
    assert isinstance(provider_oai, OpenAIProvider)
    assert provider_oai.model == "gpt-4o-custom"


def test_submit_task_returns_503_when_no_provider():
    reset_settings(Settings(provider=None))
    client = TestClient(app)

    resp = client.post(
        "/api/tasks",
        json={"query": "Test task with no provider"},
    )
    assert resp.status_code == 503
    assert "No LLM provider configured" in resp.json()["detail"]


def test_submit_task_returns_503_when_api_key_missing(monkeypatch):
    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    client = TestClient(app)
    resp = client.post(
        "/api/tasks",
        json={"query": "Test task with missing anthropic key"},
    )
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY is not configured" in resp.json()["detail"]

    reset_settings(Settings(provider="openai", model="gpt-4o"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    resp_oai = client.post(
        "/api/tasks",
        json={"query": "Test task with missing openai key"},
    )
    assert resp_oai.status_code == 503
    assert "OPENAI_API_KEY is not configured" in resp_oai.json()["detail"]


def test_submit_task_succeeds_when_provider_and_key_configured(monkeypatch):
    from tests.fakes import FakeLLM
    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setattr("aegis.services.orchestrator_service.get_llm_provider", lambda: FakeLLM(model="test"))

    client = TestClient(app)
    resp = client.post(
        "/api/tasks",
        json={"query": "Test task with provider"},
    )
    assert resp.status_code == 200
    assert "task_id" in resp.json()
    assert resp.json()["status"] == "submitted"
