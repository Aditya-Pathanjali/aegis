"""Integration tests for LangGraph state machine orchestration and human review gates."""

import time

from aegis.models.schemas import ApprovalAction, TaskStatus, TaskSubmission
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.services.orchestrator_service import OrchestratorService
from tests.fakes import FakeLLM


def test_orchestrator_complete_run():
    orchestrator = AegisOrchestrator(llm_provider=FakeLLM())
    initial_state: AegisState = {
        "task_id": "test-orch-1",
        "query": "Evaluate predictive maintenance feasibility for centrifugal pump using sensor telemetry.",
        "has_dataset": True,
        "require_human_approval": False,
        "max_retries": 2,
        "retry_count": 0,
    }
    result = orchestrator.run(initial_state)

    assert result["status"] == TaskStatus.COMPLETED
    assert result["final_report"] is not None
    assert result["final_report"].executive_summary.overall_confidence > 0.8
    assert len(result["specialist_results"]) >= 2
    assert result["verification_result"] is not None
    assert result["critic_evaluation"].quality_pass is True


def test_human_gate_triggering_and_approval():
    service = OrchestratorService(llm_provider=FakeLLM())
    task_id = service.submit_task(
        TaskSubmission(
            query="Analyze bearing wear with mandatory human sign-off required.",
            require_human_approval=True,
            max_retries=1,
        )
    )

    # Poll briefly for task to reach waiting_approval
    for _ in range(30):
        resp = service.state_store.get_response(task_id)
        if resp and resp.status == TaskStatus.WAITING_APPROVAL:
            break
        time.sleep(0.1)

    resp = service.state_store.get_response(task_id)
    assert resp is not None
    assert resp.status == TaskStatus.WAITING_APPROVAL
    assert resp.approval_request is not None
    assert resp.approval_request.status == "pending"

    # Operator approves
    updated = service.resolve_human_approval(
        task_id=task_id,
        action=ApprovalAction.APPROVE,
        feedback="Overhaul authorized for next scheduled shift.",
    )
    assert updated.status == TaskStatus.COMPLETED
    assert updated.approval_request.status == "approved"
