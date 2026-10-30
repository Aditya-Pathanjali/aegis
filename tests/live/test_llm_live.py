"""Live integration smoke test against Anthropic API (requires ANTHROPIC_API_KEY)."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from aegis.config import Settings
from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.llm.types import Message, TextBlock, ToolResultBlock, ToolSpec


class TinySchema(BaseModel):
    status: str
    code: int


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable is not set",
)
def test_anthropic_live_smoke():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("AEGIS_MODEL", "claude-sonnet-5-5")
    settings = Settings(provider="anthropic", model=model)
    provider = AnthropicProvider(model=model, api_key=api_key, settings=settings)

    # 1. Tool-call round trip
    tool_spec = ToolSpec(
        name="echo_tool",
        description="Echoes the provided message back.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    )

    user_msg = Message(role="user", content=[TextBlock(text="Call echo_tool with message='smoke-test'.")])
    resp1 = provider.chat([user_msg], tools=[tool_spec])

    assert len(resp1.tool_calls) >= 1
    assert resp1.usage.input_tokens > 0
    assert resp1.usage.output_tokens > 0
    tc = resp1.tool_calls[0]
    assert tc.name == "echo_tool"

    # Send tool result back
    history = [
        user_msg,
        Message(role="assistant", content=[tc]),
        Message(role="user", content=[ToolResultBlock(tool_call_id=tc.id, content="echoed: smoke-test")]),
    ]
    resp2 = provider.chat(history, tools=[tool_spec])
    assert resp2.text is not None and len(resp2.text) > 0
    assert resp2.usage.input_tokens > 0
    assert resp2.usage.output_tokens > 0

    # 2. Structured output call
    struct_prompt = Message(
        role="user",
        content=[TextBlock(text="Return a JSON with status 'operational' and code 200.")],
    )
    result = provider.structured([struct_prompt], schema=TinySchema)

    assert isinstance(result.value, TinySchema)
    assert result.value.status == "operational"
    assert result.value.code == 200
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    cost_str = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "unpriced"

    print(
        f"\n[LIVE SMOKE TEST RESULT]\n"
        f"model: {result.model}\n"
        f"tokens: in={result.usage.input_tokens}, out={result.usage.output_tokens}, total={result.usage.input_tokens + result.usage.output_tokens}\n"
        f"latency: {result.latency_ms:.2f} ms\n"
        f"cost: {cost_str}"
    )
