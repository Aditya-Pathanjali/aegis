"""Integration tests for LangGraph state machine orchestration and human review gates."""

import time

from aegis.agents.critic import CriticRubricOutput
from aegis.agents.verification import JudgeVerdict
from aegis.models.schemas import (
    AgentRole,
    ApprovalAction,
    EvidenceRef,
    ExecutionPlan,
    Finding,
    SpecialistOutput,
    SubTask,
    SynthesizerOutput,
    TaskStatus,
    TaskSubmission,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.services.orchestrator_service import OrchestratorService
from aegis.tools.context import TaskContext
from tests.fakes import FakeLLM, fake_text, fake_tool_call


def test_orchestrator_complete_run():
    plan = ExecutionPlan(
        task_id="test-orch-1",
        original_query="Evaluate equipment maintenance feasibility.",
        objective="Assess feasibility using research and data tools.",
        subtasks=[
            SubTask(
                id="subtask-1",
                title="Literature Research",
                description="Search literature guidelines",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            ),
            SubTask(
                id="subtask-2",
                title="Data Profiling",
                description="Analyze telemetry dataset",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query"],
            ),
        ],
    )
    call_id_1 = "call-rag-1"
    tool_resp_1 = fake_tool_call(name="rag_search", args={"query": "guidelines"}, tool_call_id=call_id_1)
    res_1 = SpecialistOutput(
        summary="Research done.",
        findings=[Finding(statement="Literature review of guidelines completed.", evidence=[EvidenceRef(kind="tool_call", ref_id=call_id_1)])],
        limitations=[],
    )
    call_id_2 = "call-sql-2"
    tool_resp_2 = fake_tool_call(name="sql_query", args={"sql": "SELECT 3.2"}, tool_call_id=call_id_2)
    res_2 = SpecialistOutput(
        summary="Data analysis done.",
        findings=[Finding(statement="Peak vibration measured 3.2 mm/s.", evidence=[EvidenceRef(kind="tool_call", ref_id=call_id_2)])],
        limitations=[],
    )
    judge_1 = JudgeVerdict(verdict="supported", quote_span="guidelines", rationale="Matches RAG")
    judge_2 = JudgeVerdict(verdict="supported", quote_span="3.2", rationale="Matches SQL")
    critic_rubric = CriticRubricOutput(
        answers_question=8.0, uses_task_data=8.0, claims_verified=8.0, limitations_stated=8.0
    )
    synth_draft = SynthesizerOutput(
        title="Feasibility Assessment",
        final_finding="Feasible maintenance schedule confirmed.",
        key_conclusions=["All subsystems operational."],
        actionable_recommendations=["Proceed with planned maintenance."],
    )
    fake_llm = FakeLLM(responses=[plan, tool_resp_1, fake_text("Done."), res_1, tool_resp_2, fake_text("Done."), res_2, judge_1, judge_2, critic_rubric, synth_draft])
    task_context = TaskContext(task_id="test-orch-1")
    orchestrator = AegisOrchestrator(llm_provider=fake_llm, task_context=task_context)
    initial_state: AegisState = {
        "task_id": "test-orch-1",
        "query": "Evaluate equipment maintenance feasibility.",
        "has_dataset": True,
        "require_human_approval": False,
        "max_retries": 2,
        "retry_count": 0,
    }
    result = orchestrator.run(initial_state)

    assert result["status"] == TaskStatus.COMPLETED
    assert result["final_report"] is not None
    assert len(result["specialist_results"]) == 2
    assert result["verification_result"].status == "passed"
    assert result["critic_evaluation"].quality_pass is True


def test_human_gate_triggering_and_approval():
    plan = ExecutionPlan(
        task_id="test-gate-1",
        original_query="Analyze equipment with mandatory sign-off.",
        objective="Analyze equipment with human review gate.",
        subtasks=[
            SubTask(
                id="subtask-1",
                title="Literature Research",
                description="Search literature guidelines",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            )
        ],
    )
    res_1 = SpecialistOutput(summary="Research done.", findings=[], limitations=[])
    critic_rubric = CriticRubricOutput(
        answers_question=8.0, uses_task_data=8.0, claims_verified=8.0, limitations_stated=8.0
    )
    synth_draft = SynthesizerOutput(
        title="Gate Assessment",
        final_finding="Equipment sign-off needed.",
        key_conclusions=[],
        actionable_recommendations=[],
    )
    service = OrchestratorService(
        llm_provider=FakeLLM(responses=[plan, fake_text("Done."), res_1, critic_rubric, synth_draft])
    )
    task_id = service.submit_task(
        TaskSubmission(
            query="Analyze equipment with mandatory sign-off.",
            require_human_approval=True,
            max_retries=0,
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
    assert resp.approval_request.proposed_actions == []

    # Operator approves
    updated = service.resolve_human_approval(
        task_id=task_id,
        action=ApprovalAction.APPROVE,
        feedback="Authorization granted.",
    )
    assert updated.status == TaskStatus.COMPLETED
    assert updated.approval_request.status == "approved"
