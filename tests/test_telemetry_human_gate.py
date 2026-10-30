"""Tests for PR 7: Telemetry, accounting, human review gate with interrupt, and SSE streaming."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from langgraph.types import Command

from aegis.agents.critic import CriticRubricOutput
from aegis.api.main import app
from aegis.llm.pricing import clear_pricing_cache
from aegis.models.schemas import (
    AgentRole,
    ExecutionPlan,
    HumanApprovalRequest,
    SpecialistOutput,
    SubTask,
    SynthesizerOutput,
    TaskStatus,
    TraceEvent,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.services.orchestrator_service import get_orchestrator_service
from tests.fakes import FakeLLM, fake_structured, fake_text


def _build_test_plan(task_id: str = "t-gate-test") -> ExecutionPlan:
    return ExecutionPlan(
        task_id=task_id,
        original_query="Audit compressor bearing vibration",
        objective="Inspect bearing telemetry and recommend maintenance schedule.",
        subtasks=[
            SubTask(
                id="st-1",
                title="Bearing Literature Research",
                description="Profile vibration standards and check thresholds.",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            )
        ],
    )


def test_pause_and_resume_with_command():
    """Verify pause and resume flow through the human gate using LangGraph Command(resume=...)."""
    plan = _build_test_plan("t-pause-cmd")
    spec_out = SpecialistOutput(
        summary="Bearing vibration elevated.",
        findings=[],
        limitations=["Data restricted to last 24h."],
    )
    critic_rubric = CriticRubricOutput(
        answers_question=8.0,
        uses_task_data=8.0,
        claims_verified=8.0,
        limitations_stated=8.0,
    )
    synth_out = SynthesizerOutput(
        title="Bearing Assessment",
        final_finding="Bearing needs replacement.",
        key_conclusions=["Vibration peak verified."],
        actionable_recommendations=["Schedule overhaul in 48 hours."],
    )

    fake_llm = FakeLLM(responses=[plan, fake_text("Analysis complete."), spec_out, critic_rubric, synth_out])
    orchestrator = AegisOrchestrator(llm_provider=fake_llm)

    initial_state: AegisState = {
        "task_id": "t-pause-cmd",
        "query": "Audit compressor bearing vibration",
        "require_human_approval": True,
        "max_retries": 1,
        "retry_count": 0,
    }

    # 1. Run workflow: must pause at human gate with interrupt
    paused_state = orchestrator.run(initial_state)

    assert paused_state["status"] == TaskStatus.WAITING_APPROVAL
    approval_req = paused_state.get("human_approval")
    assert approval_req is not None
    assert approval_req.status == "pending"
    assert approval_req.required is True
    assert "Schedule overhaul in 48 hours." in approval_req.proposed_actions

    # 2. Resume workflow with LangGraph Command(resume={"action": "approve", "feedback": "Approved by engineering lead"})
    config = {"configurable": {"thread_id": "t-pause-cmd"}}
    resumed_state = orchestrator.graph.invoke(
        Command(resume={"action": "approve", "feedback": "Approved by engineering lead"}),
        config=config,
    )

    assert resumed_state["status"] == TaskStatus.COMPLETED
    assert resumed_state["human_approval"].status == "approved"
    assert resumed_state["human_approval"].feedback == "Approved by engineering lead"


def test_reject_approval_attempt_when_task_not_waiting_409():
    """Verify POST /api/tasks/{task_id}/approve returns 409 Conflict when not in WAITING_APPROVAL."""
    client = TestClient(app)
    service = get_orchestrator_service()

    # Case 1: Task completed (not waiting)
    completed_task_id = "task-completed-test"
    service.state_store.save_state(
        completed_task_id,
        {
            "task_id": completed_task_id,
            "query": "Test query",
            "status": TaskStatus.COMPLETED,
            "trace_events": [],
            "errors": [],
        },
    )

    resp = client.post(
        f"/api/tasks/{completed_task_id}/approve",
        json={"action": "approve", "feedback": "Too late"},
    )
    assert resp.status_code == 409
    assert "WAITING_APPROVAL" in resp.json()["detail"]

    # Also test /approval alias
    resp_alias = client.post(
        f"/api/tasks/{completed_task_id}/approval",
        json={"action": "approve", "feedback": "Too late"},
    )
    assert resp_alias.status_code == 409

    # Case 2: Non-existent task returns 404
    resp_404 = client.post(
        "/api/tasks/non-existent-task-id-999/approve",
        json={"action": "approve"},
    )
    assert resp_404.status_code == 404

    # Case 3: Task waiting for approval succeeds (200)
    waiting_task_id = "task-waiting-test"
    service.state_store.save_state(
        waiting_task_id,
        {
            "task_id": waiting_task_id,
            "query": "Pending query",
            "status": TaskStatus.WAITING_APPROVAL,
            "human_approval": HumanApprovalRequest(task_id=waiting_task_id, required=True, status="pending"),
            "trace_events": [],
            "errors": [],
        },
    )

    resp_ok = client.post(
        f"/api/tasks/{waiting_task_id}/approve",
        json={"action": "approve", "feedback": "Approved"},
    )
    assert resp_ok.status_code == 200
    data = resp_ok.json()
    assert data["status"] == "completed"
    assert data["approval_request"]["status"] == "approved"


def test_sse_stream_delivers_in_order_with_no_duplicates_on_reconnect():
    """Verify GET /api/tasks/{task_id}/events streams in order, deduplicates, and unregisters on disconnect."""
    client = TestClient(app)
    service = get_orchestrator_service()
    task_id = "task-sse-test-1"

    # Pre-populate state store with initial historical events
    ev1 = TraceEvent(event_id="ev-001", task_id=task_id, event_type="agent_start", message="Start 1")
    ev2 = TraceEvent(event_id="ev-002", task_id=task_id, event_type="tool_call", message="Tool 1")
    service.state_store.save_state(
        task_id,
        {
            "task_id": task_id,
            "query": "SSE test query",
            "status": TaskStatus.EXECUTING,
            "trace_events": [ev1, ev2],
            "errors": [],
        },
    )

    # First connection: client connects, reads historical events
    received_ids_pass1 = []
    with client.stream("GET", f"/api/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if "event_id" in payload:
                    received_ids_pass1.append(payload["event_id"])
                if len(received_ids_pass1) == 2:
                    break  # Disconnect after 2 events

    assert received_ids_pass1 == ["ev-001", "ev-002"]

    # Now add more events and mark task completed
    ev3 = TraceEvent(event_id="ev-003", task_id=task_id, event_type="agent_finish", message="Finish 1")
    state = service.state_store.get_state(task_id)
    state["trace_events"].append(ev3)
    state["status"] = TaskStatus.COMPLETED
    service.state_store.save_state(task_id, state)

    # Reconnect: client must get all 3 events without duplicate IDs
    received_ids_pass2 = []
    with client.stream("GET", f"/api/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if "event_id" in payload:
                    received_ids_pass2.append(payload["event_id"])
                elif payload.get("event_type") == "stream_end":
                    break

    assert received_ids_pass2 == ["ev-001", "ev-002", "ev-003"]
    # Verify no duplicates
    assert len(received_ids_pass2) == len(set(received_ids_pass2))

    # Verify listeners cleaned up on disconnect
    assert task_id not in service._subscribers or len(service._subscribers[task_id]) == 0


def test_trace_token_counts_match_fake_llm_and_cost_null_for_unpriced_model(tmp_path: Path):
    """Verify TraceEvent tokens match FakeLLM usage, and costs are null when unpriced vs computed when priced."""
    plan = _build_test_plan("t-tokens-test")
    spec_out = SpecialistOutput(summary="Analysis complete.", findings=[], limitations=[])
    critic_rubric = CriticRubricOutput(
        answers_question=9.0, uses_task_data=9.0, claims_verified=9.0, limitations_stated=9.0
    )
    synth_out = SynthesizerOutput(
        title="Unpriced Model Assessment",
        final_finding="Normal operations.",
        key_conclusions=[],
        actionable_recommendations=[],
    )

    # Fake responses with known usage on an unpriced model
    unpriced_model = "custom-unpriced-llm-v1"
    r1 = fake_structured(plan, input_tokens=100, output_tokens=50, model=unpriced_model)
    r2 = fake_text("Working...", input_tokens=100, output_tokens=50, model=unpriced_model)
    r3 = fake_structured(spec_out, input_tokens=100, output_tokens=50, model=unpriced_model)
    r4 = fake_structured(critic_rubric, input_tokens=100, output_tokens=50, model=unpriced_model)
    r5 = fake_structured(synth_out, input_tokens=100, output_tokens=50, model=unpriced_model)

    fake_llm = FakeLLM(
        responses=[r1, r2, r3, r4, r5],
        model=unpriced_model,
    )

    orch = AegisOrchestrator(llm_provider=fake_llm)
    state = orch.run(
        {
            "task_id": "t-tokens-test",
            "query": "Check telemetry",
            "require_human_approval": False,
            "max_retries": 1,
            "retry_count": 0,
        }
    )

    report = state.get("final_report")
    assert report is not None

    # Verify TraceEvent tokens match FakeLLM usage
    llm_traces = [t for t in report.agent_execution_trace if t.call_count > 0]
    assert len(llm_traces) >= 4  # Planner, Specialist, Critic, Synthesizer

    for t in llm_traces:
        assert t.input_tokens > 0
        assert t.output_tokens > 0
        assert t.token_usage == t.input_tokens + t.output_tokens
        assert t.cost_usd is None  # Unpriced model MUST have null cost

    # In FinalReport, total_cost_usd is null because the model is unpriced
    assert report.total_cost_usd is None
    assert report.total_tokens == sum(t.token_usage for t in report.agent_execution_trace)

    # Now verify priced model computes non-null cost
    custom_pricing = {
        "priced-test-model": {
            "input_per_mtok_usd": 4.0,
            "output_per_mtok_usd": 16.0,
            "source_url": "https://example.com/pricing",
            "checked_on": "2026-10-30",
        }
    }
    pricing_file = tmp_path / "pricing.yaml"
    import yaml

    pricing_file.write_text(yaml.safe_dump(custom_pricing), encoding="utf-8")
    clear_pricing_cache()

    from aegis.llm import pricing

    orig_default = pricing.DEFAULT_PRICING_FILE
    pricing.DEFAULT_PRICING_FILE = pricing_file
    try:
        p_model = "priced-test-model"
        p1 = fake_structured(plan, input_tokens=100, output_tokens=50, model=p_model)
        p2 = fake_text("Working...", input_tokens=100, output_tokens=50, model=p_model)
        p3 = fake_structured(spec_out, input_tokens=100, output_tokens=50, model=p_model)
        p4 = fake_structured(critic_rubric, input_tokens=100, output_tokens=50, model=p_model)
        p5 = fake_structured(synth_out, input_tokens=100, output_tokens=50, model=p_model)

        priced_fake_llm = FakeLLM(
            responses=[p1, p2, p3, p4, p5],
            model=p_model,
        )
        orch_priced = AegisOrchestrator(llm_provider=priced_fake_llm)
        state_priced = orch_priced.run(
            {
                "task_id": "t-priced-test",
                "query": "Check telemetry priced",
                "require_human_approval": False,
                "max_retries": 1,
                "retry_count": 0,
            }
        )
        report_priced = state_priced["final_report"]
        assert report_priced.total_cost_usd is not None
        assert report_priced.total_cost_usd > 0.0

        for t in [tr for tr in report_priced.agent_execution_trace if tr.call_count > 0]:
            assert t.cost_usd is not None
            expected_cost = round((t.input_tokens * 4.0 + t.output_tokens * 16.0) / 1_000_000.0, 6)
            assert t.cost_usd == expected_cost
    finally:
        pricing.DEFAULT_PRICING_FILE = orig_default
        clear_pricing_cache()
