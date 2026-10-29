"""Unit tests for aegis/llm: adapters, structured output, pricing, and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.llm.errors import LLMRequestError, StructuredOutputError
from aegis.llm.openai_provider import OpenAIProvider
from aegis.llm.pricing import calculate_cost, load_pricing_registry
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
    # Mock return object
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from Claude"
    mock_resp = MagicMock()
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
    assert res.model == "claude-sonnet-5-5"
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
    resp1.content = [tool_block]
    resp1.stop_reason = "tool_use"
    resp1.usage.input_tokens = 50
    resp1.usage.output_tokens = 20

    # Step 2: Final text
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Fluid level is normal."

    resp2 = MagicMock()
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
    assert res.model == "gpt-4o"

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
    resp1.choices = [choice1]
    resp1.usage.prompt_tokens = 40
    resp1.usage.completion_tokens = 25

    # Step 2: Final text response
    choice2 = MagicMock()
    choice2.message.content = "Query executed successfully."
    choice2.message.tool_calls = None
    choice2.finish_reason = "stop"

    resp2 = MagicMock()
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
# Structured Output & Repair Loop Tests
# ==============================================================================

def test_structured_output_valid_first_time():
    mock_client = MagicMock()
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = "tc_1"
    tool_block.name = "DummyOutput"
    tool_block.input = {"summary": "All systems nominal", "count": 5}

    resp = MagicMock()
    resp.content = [tool_block]
    resp.stop_reason = "tool_use"
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 30
    mock_client.messages.create.return_value = resp

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    result = provider.structured([Message(role="user", content=[TextBlock(text="Generate status")])], schema=DummyOutput)

    assert result.attempts == 1
    assert isinstance(result.value, DummyOutput)
    assert result.value.summary == "All systems nominal"
    assert result.value.count == 5
    assert result.usage.input_tokens == 100


def test_structured_output_invalid_then_repaired():
    mock_client = MagicMock()

    # Attempt 1 returns invalid count: 0 (< 1)
    b1 = MagicMock()
    b1.type = "tool_use"
    b1.id = "tc_1"
    b1.name = "DummyOutput"
    b1.input = {"summary": "Bad count", "count": 0}
    resp1 = MagicMock()
    resp1.content = [b1]
    resp1.stop_reason = "tool_use"
    resp1.usage.input_tokens = 100
    resp1.usage.output_tokens = 25

    # Attempt 2 returns valid count: 3
    b2 = MagicMock()
    b2.type = "tool_use"
    b2.id = "tc_2"
    b2.name = "DummyOutput"
    b2.input = {"summary": "Repaired count", "count": 3}
    resp2 = MagicMock()
    resp2.content = [b2]
    resp2.stop_reason = "tool_use"
    resp2.usage.input_tokens = 120
    resp2.usage.output_tokens = 30

    mock_client.messages.create.side_effect = [resp1, resp2]

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    result = provider.structured([Message(role="user", content=[TextBlock(text="Generate status")])], schema=DummyOutput)

    assert result.attempts == 2
    assert result.value.count == 3
    assert result.value.summary == "Repaired count"
    assert result.usage.input_tokens == 220
    assert result.usage.output_tokens == 55

    # Verify repair turn message contained the validation error
    second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
    user_repair_block = second_call_messages[2]["content"][0]
    assert user_repair_block["type"] == "tool_result"
    assert user_repair_block["is_error"] is True
    assert "Schema validation error" in user_repair_block["content"]


def test_structured_output_invalid_twice_raises_structured_output_error():
    mock_client = MagicMock()

    # Both attempts return invalid count 0
    b = MagicMock()
    b.type = "tool_use"
    b.id = "tc_err"
    b.name = "DummyOutput"
    b.input = {"summary": "Still bad", "count": 0}
    resp = MagicMock()
    resp.content = [b]
    resp.stop_reason = "tool_use"
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 20

    mock_client.messages.create.return_value = resp

    provider = AnthropicProvider(model="claude-sonnet-5-5", api_key="fake", client=mock_client)
    with pytest.raises(StructuredOutputError) as exc_info:
        provider.structured([Message(role="user", content=[TextBlock(text="Generate")])], schema=DummyOutput)

    assert "Structured output validation failed after repair turn" in str(exc_info.value)
    assert hasattr(exc_info.value, "raw_text")
    assert "Still bad" in exc_info.value.raw_text


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


# ==============================================================================
# Planner Trace Event Real Token Integration Test
# ==============================================================================

def test_planner_trace_event_tokens_equal_fake_llm_usage():
    from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask

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
            )
        ]
    )
    orch = AegisOrchestrator(llm_provider=fake_llm)
    state: AegisState = {
        "task_id": "t-trace-1",
        "query": "Check pump vibration",
        "has_dataset": False,
        "require_human_approval": False,
        "max_retries": 1,
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
