"""Unit tests for FakeLLM protocol implementation and helper functions."""

from __future__ import annotations

import pytest

from aegis.llm.types import Message, TextBlock
from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask
from tests.fakes import FakeLLM, fake_structured, fake_text, fake_tool_call


def test_fake_llm_records_requests():
    llm = FakeLLM(responses=[fake_text("Test response", input_tokens=15, output_tokens=25)])
    msg = Message(role="user", content=[TextBlock(text="Explain analysis")])
    resp = llm.chat([msg], system="Be concise")
    assert resp.text == "Test response"
    assert resp.usage.input_tokens == 15
    assert resp.usage.output_tokens == 25
    assert len(llm.calls) == 1
    assert llm.calls[0]["method"] == "chat"
    assert llm.calls[0]["messages"] == [msg]
    assert llm.calls[0]["system"] == "Be concise"


def test_fake_llm_queued_responses():
    llm = FakeLLM(responses=["First queued response", fake_tool_call("rag_search", {"query": "test"})])
    msg1 = Message(role="user", content=[TextBlock(text="Query 1")])
    msg2 = Message(role="user", content=[TextBlock(text="Query 2")])

    resp1 = llm.chat([msg1])
    assert resp1.text == "First queued response"

    resp2 = llm.chat([msg2])
    assert len(resp2.tool_calls) == 1
    assert resp2.tool_calls[0].name == "rag_search"

    # When queue is empty, must raise AssertionError
    with pytest.raises(AssertionError, match="FakeLLM: no scripted response left"):
        llm.chat([msg1])


def test_fake_llm_structured_output():
    plan = ExecutionPlan(
        task_id="p-1",
        original_query="Test query",
        objective="Test objective",
        subtasks=[
            SubTask(
                id="st-1",
                title="Subtask 1",
                description="Desc 1",
                assigned_agent=AgentRole.RESEARCH,
            )
        ],
    )
    llm = FakeLLM(responses=[fake_structured(plan, input_tokens=100, output_tokens=50)])
    msg = Message(role="user", content=[TextBlock(text="Plan task")])
    res = llm.structured([msg], schema=ExecutionPlan)
    assert res.value.task_id == "p-1"
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 50
    assert len(llm.calls) == 1
    assert llm.calls[0]["schema"] == ExecutionPlan

    # When empty, raises AssertionError
    with pytest.raises(AssertionError, match="FakeLLM: no scripted response left"):
        llm.structured([msg], schema=ExecutionPlan)
