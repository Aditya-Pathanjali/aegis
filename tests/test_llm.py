"""Unit tests for aegis/llm: adapters, structured output, pricing, and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel, Field

from aegis.config import Settings
from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.llm.errors import LLMRequestError, StructuredOutputError
from aegis.llm.openai_provider import OpenAIProvider
from aegis.llm.pricing import (
    calculate_cost,
    clear_pricing_cache,
    load_pricing_registry,
    resolve_pricing_path,
)
from aegis.llm.types import (
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from tests.fakes import FakeLLM, fake_structured


class DummyOutput(BaseModel):
    summary: str
    count: int = Field(ge=1)


# ==============================================================================
# Anthropic Provider Tests (Mocked Client)
# ==============================================================================

def test_anthropic_text_reply():
    mock_client = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from Claude"
    mock_resp = MagicMock()
    mock_resp.model = "claude-sonnet-5-5-20261022"
    mock_resp.content = [text_block]
    mock_resp.stop_reason = "end_turn"
    mock_resp.usage.input_tokens = 42
    mock_resp.usage.output_tokens = 18
    mock_client.messages.create.return_value = mock_resp

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    res = provider.chat([Message(role="user", content=[TextBlock(text="Hi")])])

    assert res.text == "Hello from Claude"
    assert res.usage.input_tokens == 42
    assert res.usage.output_tokens == 18
    assert res.stop_reason == "end_turn"
    # Model should reflect the API response model string
    assert res.model == "claude-sonnet-5-5-20261022"
    assert res.latency_ms > 0

    # Ensure temperature omitted when None
    kwargs = mock_client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs


def test_anthropic_tool_call_and_result_flow():
    mock_client = MagicMock()

    # Step 1: LLM returns tool use block
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "call_abc123"
    tool_block.name = "rag_search"
    tool_block.input = {"query": "hydraulic fluid"}

    resp1 = MagicMock()
    resp1.model = "claude-sonnet-5-5"
    resp1.content = [tool_block]
    resp1.stop_reason = "tool_use"
    resp1.usage.input_tokens = 50
    resp1.usage.output_tokens = 20

    # Step 2: Final text
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Fluid level is normal."

    resp2 = MagicMock()
    resp2.model = "claude-sonnet-5-5"
    resp2.content = [text_block]
    resp2.stop_reason = "end_turn"
    resp2.usage.input_tokens = 80
    resp2.usage.output_tokens = 30

    mock_client.messages.create.side_effect = [resp1, resp2]

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    tool_spec = ToolSpec(name="rag_search", description="Search docs", input_schema={"type": "object"})

    # First turn: trigger tool call
    res1 = provider.chat([Message(role="user", content=[TextBlock(text="Check fluid")])], tools=[tool_spec])
    assert res1.stop_reason == "tool_use"
    assert len(res1.tool_calls) == 1
    assert res1.tool_calls[0].name == "rag_search"
    assert res1.tool_calls[0].arguments == {"query": "hydraulic fluid"}

    # Second turn: send tool result back (with is_error=True test)
    history = [
        Message(role="user", content=[TextBlock(text="Check fluid")]),
        Message(role="assistant", content=[res1.tool_calls[0]]),
        Message(role="user", content=[ToolResultBlock(tool_call_id="call_abc123", content="Sensor error", is_error=True)]),
    ]
    res2 = provider.chat(history, tools=[tool_spec])
    assert res2.text == "Fluid level is normal."
    assert res2.stop_reason == "end_turn"

    # Verify is_error was passed through to SDK
    call2_kwargs = mock_client.messages.create.call_args.kwargs
    call2_messages = call2_kwargs["messages"]
    tool_result_content = call2_messages[2]["content"][0]
    assert tool_result_content["type"] == "tool_result"
    assert tool_result_content["is_error"] is True
    assert tool_result_content["content"] == "Sensor error"


def test_anthropic_error_handling_401():
    import anthropic
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.AuthenticationError(
        message="Invalid API Key",
        response=MagicMock(status_code=401, headers={}),
        body={"error": {"message": "Invalid API Key"}},
    )

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="bad-key", client=mock_client)
    with pytest.raises(LLMRequestError) as exc_info:
        provider.chat([Message(role="user", content=[TextBlock(text="test")])])

    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == "anthropic"
    assert "Invalid API Key" in str(exc_info.value)


# ==============================================================================
# OpenAI Provider Tests (Mocked Client)
# ==============================================================================

def test_openai_text_reply():
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = "Hello from GPT-4o"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"

    mock_resp = MagicMock()
    mock_resp.model = "gpt-4o-2024-08-06"
    mock_resp.choices = [choice]
    mock_resp.usage.prompt_tokens = 30
    mock_resp.usage.completion_tokens = 15
    mock_client.chat.completions.create.return_value = mock_resp

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    res = provider.chat([Message(role="user", content=[TextBlock(text="Hello")])])

    assert res.text == "Hello from GPT-4o"
    assert res.usage.input_tokens == 30
    assert res.usage.output_tokens == 15
    assert res.stop_reason == "end_turn"
    # Model should reflect the API response model string
    assert res.model == "gpt-4o-2024-08-06"

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs


def test_openai_tool_call_and_result_flow():
    mock_client = MagicMock()

    # Step 1: Tool call returned
    tc_mock = MagicMock()
    tc_mock.id = "call_oai_1"
    tc_mock.function.name = "sql_query"
    tc_mock.function.arguments = json.dumps({"query": "SELECT * FROM dataset"})

    choice1 = MagicMock()
    choice1.message.content = None
    choice1.message.tool_calls = [tc_mock]
    choice1.finish_reason = "tool_calls"

    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 40
    resp1.usage.completion_tokens = 25

    # Step 2: Final text response
    choice2 = MagicMock()
    choice2.message.content = "Query executed successfully."
    choice2.message.tool_calls = None
    choice2.finish_reason = "stop"

    resp2 = MagicMock()
    resp2.model = "gpt-4o"
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 70
    resp2.usage.completion_tokens = 15

    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    tool_spec = ToolSpec(name="sql_query", description="Execute SQL", input_schema={"type": "object"})

    res1 = provider.chat([Message(role="user", content=[TextBlock(text="Run query")])], tools=[tool_spec])
    assert res1.stop_reason == "tool_use"
    assert len(res1.tool_calls) == 1
    assert res1.tool_calls[0].arguments == {"query": "SELECT * FROM dataset"}

    history = [
        Message(role="user", content=[TextBlock(text="Run query")]),
        Message(role="assistant", content=[res1.tool_calls[0]]),
        Message(role="user", content=[ToolResultBlock(tool_call_id="call_oai_1", content="[{'id': 1}]")]),
    ]
    res2 = provider.chat(history, tools=[tool_spec])
    assert res2.text == "Query executed successfully."
    assert res2.stop_reason == "end_turn"

    # Check that tool role message was mapped properly
    call2_kwargs = mock_client.chat.completions.create.call_args.kwargs
    call2_messages = call2_kwargs["messages"]
    tool_msg = next(m for m in call2_messages if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "call_oai_1"
    assert tool_msg["content"] == "[{'id': 1}]"


def test_openai_tool_argument_parsing_failure_flow():
    """Unparseable tool argument JSON must not wrap as {'raw': ...}, keeps raw argument,

    and next turn returns tool error to model.
    """
    mock_client = MagicMock()

    # Step 1: Model outputs broken JSON for arguments
    bad_json = "{bad_json: missing_quotes"
    tc_mock = MagicMock()
    tc_mock.id = "call_bad_1"
    tc_mock.function.name = "calc_tool"
    tc_mock.function.arguments = bad_json

    choice1 = MagicMock()
    choice1.message.content = None
    choice1.message.tool_calls = [tc_mock]
    choice1.finish_reason = "tool_calls"

    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 30
    resp1.usage.completion_tokens = 15

    # Step 2: Next turn text response
    choice2 = MagicMock()
    choice2.message.content = "I apologize for the error."
    choice2.message.tool_calls = None
    choice2.finish_reason = "stop"

    resp2 = MagicMock()
    resp2.model = "gpt-4o"
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 50
    resp2.usage.completion_tokens = 10

    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    res1 = provider.chat([Message(role="user", content=[TextBlock(text="Calculate")])])

    assert len(res1.tool_calls) == 1
    call = res1.tool_calls[0]
    # Critical: arguments is empty dict, not {"raw": ...}
    assert call.arguments == {}
    assert call.raw_arguments == bad_json
    assert call.parse_error is not None

    # Step 2: Return tool error to model on next turn
    history = [
        Message(role="user", content=[TextBlock(text="Calculate")]),
        Message(role="assistant", content=[call]),
        Message(
            role="user",
            content=[
                ToolResultBlock(
                    tool_call_id=call.id,
                    content=f"Tool error: invalid JSON arguments ({call.parse_error})",
                    is_error=True,
                )
            ],
        ),
    ]
    res2 = provider.chat(history)
    assert res2.text == "I apologize for the error."

    # Verify that in assistant turn, raw_arguments was mapped to API, not {"raw": ...}
    call2_messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    assistant_msg = next(m for m in call2_messages if m.get("role") == "assistant")
    assert assistant_msg["tool_calls"][0]["function"]["arguments"] == bad_json
    tool_msg = next(m for m in call2_messages if m.get("role") == "tool")
    assert "Tool error: invalid JSON arguments" in tool_msg["content"]


def test_openai_error_handling_401():
    import openai
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
        message="Incorrect API key provided",
        response=MagicMock(status_code=401, headers={}),
        body={"error": {"message": "Incorrect API key provided"}},
    )

    provider = OpenAIProvider(model="gpt-4o", api_key="bad-key", client=mock_client)
    with pytest.raises(LLMRequestError) as exc_info:
        provider.chat([Message(role="user", content=[TextBlock(text="test")])])

    assert exc_info.value.status_code == 401
    assert exc_info.value.provider == "openai"


# ==============================================================================
# Native Structured Output Tests (Anthropic & OpenAI)
# ==============================================================================

def test_anthropic_structured_native_valid_first_time():
    mock_client = MagicMock()
    parsed_block = MagicMock()
    parsed_block.parsed_output = DummyOutput(summary="All systems nominal", count=5)

    resp = MagicMock()
    resp.model = "claude-sonnet-5-5-20261022"
    resp.content = [parsed_block]
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 30
    mock_client.messages.parse.return_value = resp

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate status")])],
        schema=DummyOutput,
    )

    assert result.attempts == 1
    assert isinstance(result.value, DummyOutput)
    assert result.value.summary == "All systems nominal"
    assert result.value.count == 5
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 30
    assert result.model == "claude-sonnet-5-5-20261022"
    assert mock_client.messages.parse.called


def test_anthropic_structured_native_invalid_then_repaired():
    mock_client = MagicMock()

    # Attempt 1: parse returns empty / no parsed output
    resp1 = MagicMock()
    resp1.model = "claude-sonnet-5-5"
    resp1.content = [MagicMock(text="I cannot parse this", parsed_output=None)]
    resp1.usage.input_tokens = 100
    resp1.usage.output_tokens = 25

    # Attempt 2: parse succeeds
    parsed_block = MagicMock()
    parsed_block.parsed_output = DummyOutput(summary="Repaired summary", count=3)
    resp2 = MagicMock()
    resp2.model = "claude-sonnet-5-5"
    resp2.content = [parsed_block]
    resp2.usage.input_tokens = 120
    resp2.usage.output_tokens = 30

    mock_client.messages.parse.side_effect = [resp1, resp2]

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate status")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    assert result.value.count == 3
    assert result.value.summary == "Repaired summary"
    assert result.usage.input_tokens == 220
    assert result.usage.output_tokens == 55

    # Check that repair turn passed error note in user message
    second_call_messages = mock_client.messages.parse.call_args_list[1].kwargs["messages"]
    last_user_blocks = second_call_messages[-1]["content"]
    assert any("Schema validation error" in b.get("text", "") for b in last_user_blocks)


def test_anthropic_structured_native_invalid_twice_raises_error():
    mock_client = MagicMock()

    resp = MagicMock()
    resp.model = "claude-sonnet-5-5"
    resp.content = [MagicMock(text="Still failing", parsed_output=None)]
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 20
    mock_client.messages.parse.return_value = resp

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    with pytest.raises(StructuredOutputError) as exc_info:
        provider.structured(
            [Message(role="user", content=[TextBlock(text="Generate")])],
            schema=DummyOutput,
        )

    assert "Structured output validation failed after repair turn" in str(exc_info.value)
    assert "Still failing" in exc_info.value.raw_text


def test_anthropic_structured_tool_mode_fallback():
    """Explicit structured_mode='tool' uses forced tool calls instead of native parse."""
    mock_client = MagicMock()

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tc_1"
    tool_block.name = "DummyOutput"
    tool_block.input = {"summary": "Tool fallback working", "count": 10}

    resp = MagicMock()
    resp.model = "claude-sonnet-5-5"
    resp.content = [tool_block]
    resp.stop_reason = "tool_use"
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 30
    mock_client.messages.create.return_value = resp

    settings = Settings(provider="anthropic", model="claude-sonnet-5-5", structured_mode="tool")
    provider = AnthropicProvider(
        model="claude-sonnet-5-5",
        api_key="fake",
        settings=settings,
        client=mock_client,
    )
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate status")])],
        schema=DummyOutput,
    )

    assert result.attempts == 1
    assert result.value.count == 10
    assert result.value.summary == "Tool fallback working"
    assert mock_client.messages.create.called
    assert not mock_client.messages.parse.called


def test_anthropic_tool_mode_no_tool_call_repair_no_consecutive_user_messages():
    """When attempt 1 returns no tool call, error is appended to the last user message,

    not adding consecutive user messages.
    """
    mock_client = MagicMock()

    # Attempt 1: Model returns text without tool call
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "I am a helpful assistant."
    resp1 = MagicMock()
    resp1.model = "claude-sonnet-5-5"
    resp1.content = [text_block]
    resp1.stop_reason = "end_turn"
    resp1.usage.input_tokens = 50
    resp1.usage.output_tokens = 20

    # Attempt 2: Model returns proper tool call
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tc_repaired"
    tool_block.name = "DummyOutput"
    tool_block.input = {"summary": "Now conforming", "count": 2}
    resp2 = MagicMock()
    resp2.model = "claude-sonnet-5-5"
    resp2.content = [tool_block]
    resp2.stop_reason = "tool_use"
    resp2.usage.input_tokens = 90
    resp2.usage.output_tokens = 30

    mock_client.messages.create.side_effect = [resp1, resp2]

    settings = Settings(provider="anthropic", model="claude-sonnet-5-5", structured_mode="tool")
    provider = AnthropicProvider(
        model="claude-sonnet-5-5",
        api_key="fake",
        settings=settings,
        client=mock_client,
    )
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Please output data")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    assert result.value.count == 2

    # Verify message list in second call has exactly 1 message (user), not 2 consecutive user messages
    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    assert len(second_call_messages) == 1
    assert second_call_messages[0]["role"] == "user"
    # Content has original prompt + error instructions
    assert len(second_call_messages[0]["content"]) == 2
    assert "Please output data" in second_call_messages[0]["content"][0]["text"]
    assert "You must call the tool 'DummyOutput'" in second_call_messages[0]["content"][1]["text"]


def test_openai_structured_native_valid_first_time():
    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.parsed = DummyOutput(summary="OpenAI structured native", count=7)
    choice.message.refusal = None

    mock_resp = MagicMock()
    mock_resp.model = "gpt-4o-2024-08-06"
    mock_resp.choices = [choice]
    mock_resp.usage.prompt_tokens = 80
    mock_resp.usage.completion_tokens = 20
    mock_client.chat.completions.parse.return_value = mock_resp

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 1
    assert isinstance(result.value, DummyOutput)
    assert result.value.count == 7
    assert result.value.summary == "OpenAI structured native"
    assert result.model == "gpt-4o-2024-08-06"
    assert mock_client.chat.completions.parse.called


def test_openai_structured_native_invalid_then_repaired():
    mock_client = MagicMock()

    # Attempt 1: parsed is None
    choice1 = MagicMock()
    choice1.message.parsed = None
    choice1.message.refusal = "Validation rejected format"
    choice1.message.content = "bad output"
    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 80
    resp1.usage.completion_tokens = 20

    # Attempt 2: parsed succeeds
    choice2 = MagicMock()
    choice2.message.parsed = DummyOutput(summary="Repaired OpenAI", count=4)
    choice2.message.refusal = None
    resp2 = MagicMock()
    resp2.model = "gpt-4o"
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 110
    resp2.usage.completion_tokens = 30

    mock_client.chat.completions.parse.side_effect = [resp1, resp2]

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    assert result.value.count == 4
    assert result.value.summary == "Repaired OpenAI"
    assert result.usage.input_tokens == 190
    assert result.usage.output_tokens == 50


def test_openai_structured_tool_mode_fallback():
    """Explicit structured_mode='tool' uses forced tool calls for OpenAI."""
    mock_client = MagicMock()

    tc_mock = MagicMock()
    tc_mock.id = "call_dummy_1"
    tc_mock.function.name = "DummyOutput"
    tc_mock.function.arguments = json.dumps({"summary": "OpenAI tool mode", "count": 12})

    choice = MagicMock()
    choice.message.content = None
    choice.message.tool_calls = [tc_mock]
    choice.finish_reason = "tool_calls"

    resp = MagicMock()
    resp.model = "gpt-4o"
    resp.choices = [choice]
    resp.usage.prompt_tokens = 60
    resp.usage.completion_tokens = 25
    mock_client.chat.completions.create.return_value = resp

    settings = Settings(provider="openai", model="gpt-4o", structured_mode="tool")
    provider = OpenAIProvider(
        model="gpt-4o",
        api_key="fake",
        settings=settings,
        client=mock_client,
    )
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 1
    assert result.value.count == 12
    assert result.value.summary == "OpenAI tool mode"
    assert mock_client.chat.completions.create.called
    assert not mock_client.chat.completions.parse.called


def test_structured_unpriced_attempt_returns_none_total_cost(tmp_path):
    """If any attempt is unpriced (model not in pricing.yaml), total cost must be None."""
    mock_client = MagicMock()

    pricing_yaml = tmp_path / "pricing.yaml"
    pricing_yaml.write_text(
        """
gpt-4o:
  input_per_mtok_usd: 2.5
  output_per_mtok_usd: 10.0
  source_url: "https://openai.com/pricing"
  checked_on: "2026-10-01"
""",
        encoding="utf-8",
    )

    # Attempt 1: Known priced model
    choice1 = MagicMock()
    choice1.message.parsed = None
    choice1.message.refusal = "Try again"
    choice1.message.content = ""
    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 100
    resp1.usage.completion_tokens = 50

    # Attempt 2: Unknown unpriced model returned
    choice2 = MagicMock()
    choice2.message.parsed = DummyOutput(summary="Done", count=1)
    choice2.message.refusal = None
    resp2 = MagicMock()
    resp2.model = "unpriced-experimental-model-v99"
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 120
    resp2.usage.completion_tokens = 60

    mock_client.chat.completions.parse.side_effect = [resp1, resp2]

    settings = Settings(provider="openai", model="gpt-4o", pricing_file=str(pricing_yaml))
    provider = OpenAIProvider(model="gpt-4o", api_key="fake", settings=settings, client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    # Since attempt 2 was unpriced (None), total cost must be None
    assert result.cost_usd is None


# ==============================================================================
# Pricing Tests
# ==============================================================================

def test_pricing_known_unknown_and_malformed(tmp_path):
    pricing_yaml = tmp_path / "pricing.yaml"
    pricing_yaml.write_text(
        """
claude-sonnet-5-5:
  input_per_mtok_usd: 3.0
  output_per_mtok_usd: 15.0
  source_url: "https://anthropic.com/pricing"
  checked_on: "2026-10-01"
""",
        encoding="utf-8",
    )

    # Known model calculation
    cost = calculate_cost(
        model="claude-sonnet-5-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        pricing_file=str(pricing_yaml),
    )
    assert cost == 18.0

    # Partial calculation
    cost_partial = calculate_cost(
        model="claude-sonnet-5-5",
        input_tokens=500_000,
        output_tokens=200_000,
        pricing_file=str(pricing_yaml),
    )
    # 0.5 * 3.0 + 0.2 * 15.0 = 1.5 + 3.0 = 4.5
    assert cost_partial == 4.5

    # Unknown model gives None
    cost_unknown = calculate_cost(
        model="unpriced-experimental-model",
        input_tokens=1000,
        output_tokens=1000,
        pricing_file=str(pricing_yaml),
    )
    assert cost_unknown is None

    # Malformed YAML (not a mapping) raises ValueError at load
    bad_yaml = tmp_path / "bad_pricing.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_pricing_registry(str(bad_yaml))

    # Syntax error in YAML raises ValueError at load
    bad_syntax_yaml = tmp_path / "syntax_err.yaml"
    bad_syntax_yaml.write_text(":: invalid yaml", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed pricing"):
        load_pricing_registry(str(bad_syntax_yaml))


def test_pricing_missing_file_raises_filenotfound(tmp_path):
    missing_file = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="Pricing file not found at"):
        load_pricing_registry(str(missing_file))


def test_pricing_works_from_different_working_directory(monkeypatch, tmp_path):
    """Package-relative default pricing resolution works even when cwd is elsewhere."""
    clear_pricing_cache()
    # Change working directory to an arbitrary temporary directory
    monkeypatch.chdir(tmp_path)

    # Default loading without argument should successfully resolve package pricing.yaml
    default_path = resolve_pricing_path()
    assert default_path.exists()

    # The committed default file is an empty mapping
    registry = load_pricing_registry()
    assert isinstance(registry, dict)

    # Tests that need prices write their own temporary file
    custom_pricing = tmp_path / "custom_pricing.yaml"
    custom_pricing.write_text(
        """
custom-model:
  input_per_mtok_usd: 1.5
  output_per_mtok_usd: 3.5
  source_url: "https://example.com/pricing"
  checked_on: "2026-10-01"
""",
        encoding="utf-8",
    )
    reg_custom = load_pricing_registry(str(custom_pricing))
    assert "custom-model" in reg_custom
    assert reg_custom["custom-model"].input_per_mtok_usd == 1.5


def test_pricing_file_is_read_once_across_many_calls(tmp_path):
    """Pricing registry is cached with lru_cache; file IO occurs once across many calls."""
    pricing_yaml = tmp_path / "cached_pricing.yaml"
    pricing_yaml.write_text(
        """
test-model:
  input_per_mtok_usd: 1.0
  output_per_mtok_usd: 2.0
  source_url: "https://example.com/pricing"
  checked_on: "2026-10-01"
""",
        encoding="utf-8",
    )
    clear_pricing_cache()

    with patch("yaml.safe_load", wraps=yaml.safe_load) as mock_yaml:
        for _ in range(10):
            reg = load_pricing_registry(str(pricing_yaml))
            assert "test-model" in reg

        # yaml.safe_load called once due to LRU caching
        assert mock_yaml.call_count == 1


# ==============================================================================
# Planner Trace Event Real Token Integration Test
# ==============================================================================

def test_planner_trace_event_tokens_equal_fake_llm_usage():
    from aegis.agents.critic import CriticRubricOutput
    from aegis.models.schemas import (
        AgentRole,
        ExecutionPlan,
        SpecialistOutput,
        SubTask,
        SynthesizerOutput,
    )
    from tests.fakes import fake_text

    plan = ExecutionPlan(
        task_id="t-trace-1",
        original_query="Check pump vibration",
        objective="Decompose vibration diagnostic",
        subtasks=[
            SubTask(
                id="st-1",
                title="Review documentation",
                description="Check guidelines",
                assigned_agent=AgentRole.RESEARCH,
            )
        ],
    )
    fake_llm = FakeLLM(
        responses=[
            fake_structured(
                plan,
                input_tokens=123,
                output_tokens=456,
                model="fake-planner-model",
                latency_ms=88.5,
                cost_usd=0.005,
            ),
            fake_text("Done."),
            SpecialistOutput(summary="Doc review done.", findings=[], limitations=[]),
            CriticRubricOutput(answers_question=8.0, uses_task_data=8.0, claims_verified=8.0, limitations_stated=8.0),
            SynthesizerOutput(title="Pump Report", final_finding="Pump check done.", key_conclusions=[]),
        ]
    )
    orch = AegisOrchestrator(llm_provider=fake_llm)
    state: AegisState = {
        "task_id": "t-trace-1",
        "query": "Check pump vibration",
        "has_dataset": False,
        "require_human_approval": False,
        "max_retries": 0,
        "retry_count": 0,
    }
    result = orch.run(state)

    trace = result["trace_events"]
    planner_finish_event = next(
        e for e in trace if e.agent_role == AgentRole.PLANNER and e.event_type == "agent_finish"
    )

    assert planner_finish_event.input_tokens == 123
    assert planner_finish_event.output_tokens == 456
    assert planner_finish_event.token_usage == 579
    assert planner_finish_event.model == "fake-planner-model"
    assert planner_finish_event.duration_ms == 88.5
    assert planner_finish_event.estimated_cost_usd == 0.005


# ==============================================================================
# Exception Propagation & Usage Accounting Tests
# ==============================================================================

def test_anthropic_structured_api_connection_error_propagates():
    import anthropic
    mock_client = MagicMock()
    mock_client.messages.parse.side_effect = anthropic.APIConnectionError(request=MagicMock())

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    with pytest.raises(anthropic.APIConnectionError):
        provider.structured(
            [Message(role="user", content=[TextBlock(text="Generate")])],
            schema=DummyOutput,
        )

    # Must fail immediately without making a second repair call
    assert mock_client.messages.parse.call_count == 1


def test_openai_structured_api_connection_error_propagates():
    import openai
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = openai.APIConnectionError(request=MagicMock())

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    with pytest.raises(openai.APIConnectionError):
        provider.structured(
            [Message(role="user", content=[TextBlock(text="Generate")])],
            schema=DummyOutput,
        )

    # Must fail immediately without making a second repair call
    assert mock_client.chat.completions.parse.call_count == 1


def test_anthropic_structured_type_error_propagates():
    mock_client = MagicMock()
    mock_client.messages.parse.side_effect = TypeError("Internal type mismatch")

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    with pytest.raises(TypeError, match="Internal type mismatch"):
        provider.structured(
            [Message(role="user", content=[TextBlock(text="Generate")])],
            schema=DummyOutput,
        )

    # Must propagate unchanged, no repair call made
    assert mock_client.messages.parse.call_count == 1


def test_openai_structured_type_error_propagates():
    mock_client = MagicMock()
    mock_client.chat.completions.parse.side_effect = TypeError("Internal type mismatch")

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    with pytest.raises(TypeError, match="Internal type mismatch"):
        provider.structured(
            [Message(role="user", content=[TextBlock(text="Generate")])],
            schema=DummyOutput,
        )

    # Must propagate unchanged, no repair call made
    assert mock_client.chat.completions.parse.call_count == 1


def test_structured_tool_type_error_propagates():
    """In tool mode, unexpected TypeError during validation propagates unchanged."""
    mock_client = MagicMock()
    tc_mock = MagicMock()
    tc_mock.id = "tc_err"
    tc_mock.function.name = "DummyOutput"
    tc_mock.function.arguments = json.dumps({"summary": "Valid JSON", "count": 5})

    choice = MagicMock()
    choice.message.content = None
    choice.message.tool_calls = [tc_mock]
    choice.finish_reason = "tool_calls"
    resp = MagicMock()
    resp.model = "gpt-4o"
    resp.choices = [choice]
    mock_client.chat.completions.create.return_value = resp

    settings = Settings(provider="openai", model="gpt-4o", structured_mode="tool")
    provider = OpenAIProvider(model="gpt-4o", api_key="fake", settings=settings, client=mock_client)

    with patch.object(DummyOutput, "model_validate", side_effect=TypeError("Unexpected model bug")):
        with pytest.raises(TypeError, match="Unexpected model bug"):
            provider.structured(
                [Message(role="user", content=[TextBlock(text="Generate")])],
                schema=DummyOutput,
            )

    # Only 1 attempt was made; not treated as a schema failure
    assert mock_client.chat.completions.create.call_count == 1


def test_anthropic_structured_usage_accumulates_when_first_fails_validation():
    mock_client = MagicMock()

    # Attempt 1: response arrives but model fails schema (no parsed output)
    resp1 = MagicMock()
    resp1.model = "claude-sonnet-5-5"
    resp1.content = [MagicMock(text="Broken JSON response", parsed_output=None)]
    resp1.usage.input_tokens = 85
    resp1.usage.output_tokens = 35

    # Attempt 2: repaired response succeeds
    parsed_block = MagicMock()
    parsed_block.parsed_output = DummyOutput(summary="Repaired valid", count=3)
    resp2 = MagicMock()
    resp2.model = "claude-sonnet-5-5"
    resp2.content = [parsed_block]
    resp2.usage.input_tokens = 115
    resp2.usage.output_tokens = 45

    mock_client.messages.parse.side_effect = [resp1, resp2]

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    # Critical: tokens from first attempt (85, 35) must not be lost
    assert result.usage.input_tokens == 200  # 85 + 115
    assert result.usage.output_tokens == 80  # 35 + 45


def test_openai_structured_usage_accumulates_when_first_fails_validation():
    mock_client = MagicMock()

    # Attempt 1: response arrives but parsed is None
    choice1 = MagicMock()
    choice1.message.parsed = None
    choice1.message.refusal = "Could not comply"
    choice1.message.content = "Failed output"
    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 90
    resp1.usage.completion_tokens = 40

    # Attempt 2: repaired response succeeds
    choice2 = MagicMock()
    choice2.message.parsed = DummyOutput(summary="Now valid", count=6)
    choice2.message.refusal = None
    resp2 = MagicMock()
    resp2.model = "gpt-4o"
    resp2.choices = [choice2]
    resp2.usage.prompt_tokens = 110
    resp2.usage.completion_tokens = 50

    mock_client.chat.completions.parse.side_effect = [resp1, resp2]

    provider = OpenAIProvider(model="gpt-4o", api_key="fake", client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    # Critical: prompt tokens and completion tokens from both attempts must be accumulated
    assert result.usage.input_tokens == 200  # 90 + 110
    assert result.usage.output_tokens == 90  # 40 + 50


def test_structured_tool_usage_accumulates_when_first_fails_validation():
    mock_client = MagicMock()

    # Attempt 1: tool call has count = 0 (fails DummyOutput ge=1)
    tc1 = MagicMock()
    tc1.id = "tc_1"
    tc1.function.name = "DummyOutput"
    tc1.function.arguments = json.dumps({"summary": "Bad count", "count": 0})
    resp1 = MagicMock()
    resp1.model = "gpt-4o"
    resp1.choices = [MagicMock(message=MagicMock(tool_calls=[tc1], content=None), finish_reason="tool_calls")]
    resp1.usage.prompt_tokens = 95
    resp1.usage.completion_tokens = 30

    # Attempt 2: repaired tool call succeeds
    tc2 = MagicMock()
    tc2.id = "tc_2"
    tc2.function.name = "DummyOutput"
    tc2.function.arguments = json.dumps({"summary": "Good count", "count": 2})
    resp2 = MagicMock()
    resp2.model = "gpt-4o"
    resp2.choices = [MagicMock(message=MagicMock(tool_calls=[tc2], content=None), finish_reason="tool_calls")]
    resp2.usage.prompt_tokens = 125
    resp2.usage.completion_tokens = 40

    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    settings = Settings(provider="openai", model="gpt-4o", structured_mode="tool")
    provider = OpenAIProvider(model="gpt-4o", api_key="fake", settings=settings, client=mock_client)
    result = provider.structured(
        [Message(role="user", content=[TextBlock(text="Generate")])],
        schema=DummyOutput,
    )

    assert result.attempts == 2
    assert result.usage.input_tokens == 220  # 95 + 125
    assert result.usage.output_tokens == 70  # 30 + 40

