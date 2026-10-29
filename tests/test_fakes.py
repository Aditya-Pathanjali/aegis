"""Unit tests for FakeLLM."""

import pytest

from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask
from tests.fakes import FakeLLM


def test_fake_llm_records_requests():
    llm = FakeLLM(responses=["Test response"])
    text = llm.generate_text("Explain analysis", system_prompt="Be concise")
    assert text == "Test response"
    assert len(llm.requests) == 1
    assert llm.requests[0]["method"] == "generate_text"
    assert llm.requests[0]["prompt"] == "Explain analysis"
    assert llm.requests[0]["system_prompt"] == "Be concise"


def test_fake_llm_queued_responses():
    llm = FakeLLM(responses=["First queued response", "Second queued response"])

    assert llm.generate_text("Query 1") == "First queued response"
    assert llm.generate_text("Query 2") == "Second queued response"

    # When queue is empty, must raise AssertionError
    with pytest.raises(AssertionError, match="FakeLLM: no scripted response left"):
        llm.generate_text("Query 3")


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
    llm = FakeLLM(responses=[plan])
    result = llm.generate_structured("Plan task", schema=ExecutionPlan)
    assert isinstance(result, ExecutionPlan)
    assert result.task_id == "p-1"
    assert len(result.subtasks) == 1
    assert len(llm.requests) == 1
    assert llm.requests[0]["schema"] == ExecutionPlan

    # When empty, raises AssertionError
    with pytest.raises(AssertionError, match="FakeLLM: no scripted response left"):
        llm.generate_structured("Another plan", schema=ExecutionPlan)
