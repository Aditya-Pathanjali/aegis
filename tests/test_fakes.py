"""Unit tests for FakeLLM."""

from aegis.models.schemas import ExecutionPlan
from tests.fakes import FakeLLM


def test_fake_llm_records_requests():
    llm = FakeLLM()
    text = llm.generate_text("Explain vibration analysis", system_prompt="Be concise")
    assert isinstance(text, str)
    assert len(llm.requests) == 1
    assert llm.requests[0]["method"] == "generate_text"
    assert llm.requests[0]["prompt"] == "Explain vibration analysis"
    assert llm.requests[0]["system_prompt"] == "Be concise"


def test_fake_llm_queued_responses():
    llm = FakeLLM()
    llm.queue_response("First queued response")
    llm.queue_response("Second queued response")

    assert llm.generate_text("Query 1") == "First queued response"
    assert llm.generate_text("Query 2") == "Second queued response"
    # When queue is empty, returns fallback
    fallback = llm.generate_text("Query 3")
    assert "FakeLLM" in fallback


def test_fake_llm_structured_output():
    llm = FakeLLM()
    plan = llm.generate_structured("Plan with sensor data", schema=ExecutionPlan)
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.subtasks) >= 2
    assert len(llm.requests) == 1
    assert llm.requests[0]["schema"] == ExecutionPlan
