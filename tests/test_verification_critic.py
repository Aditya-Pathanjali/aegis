"""Tests for PR 6 verification, critic rubric evaluation, targeted retry, and synthesis."""

from __future__ import annotations

import pandas as pd
import pydantic
import pytest

from aegis.agents.critic import CriticAgent, CriticRubricOutput
from aegis.agents.verification import JudgeVerdict, VerificationAgent
from aegis.config import Settings
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    CriticEvaluation,
    EvidenceRef,
    ExecutionPlan,
    FinalReport,
    Finding,
    SpecialistOutput,
    SubTask,
    SynthesizerOutput,
    TaskStatus,
    ToolCallRecord,
    VerificationMetrics,
    VerificationResult,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.tools.context import TaskContext
from tests.fakes import FakeLLM, fake_text, fake_tool_call


def test_all_combinations_of_numeric_mismatch_and_judge_verdict():
    """Tests all 6 combinations of (numeric mismatch yes/no) x (judge verdict)."""
    call_id = "call-matrix-1"
    evidence_text = "Operational temperature measured at 75.0 C with velocity RMS 2.5 mm/s."
    tool_rec = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={},
        output={"text": evidence_text},
        execution_time_ms=5.0,
        success=True,
    )

    test_matrix = [
        # (claim, judge_verdict, quote, expected_status, expected_is_supported)
        # 1. No numeric mismatch, judge says supported
        ("Temperature reached 75.0 C.", "supported", "75.0 C", "supported", True),
        # 2. No numeric mismatch, judge says contradicted
        ("Temperature reached 75.0 C.", "contradicted", "75.0 C", "contradicted", False),
        # 3. No numeric mismatch, judge says not_enough_info
        ("Temperature reached 75.0 C.", "not_enough_info", "", "unsupported", False),
        # 4. Numeric mismatch (claim has 99.0 which is not in evidence), judge says supported
        ("Temperature reached 99.0 C.", "supported", "75.0 C", "unsupported", False),
        # 5. Numeric mismatch, judge says contradicted
        ("Temperature reached 99.0 C.", "contradicted", "75.0 C", "contradicted", False),
        # 6. Numeric mismatch, judge says not_enough_info
        ("Temperature reached 99.0 C.", "not_enough_info", "", "unsupported", False),
    ]

    for claim_text, verdict, quote, expected_status, expected_is_supported in test_matrix:
        finding = Finding(
            statement=claim_text,
            evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
        )
        spec_res = AgentExecutionResult(
            subtask_id="st-test",
            agent_role=AgentRole.DATA_ANALYST,
            output_summary="Matrix test",
            findings=[finding],
            tool_calls=[tool_rec],
        )
        fake_judge = JudgeVerdict(verdict=verdict, quote_span=quote, rationale="Matrix test")
        fake_llm = FakeLLM(responses=[fake_judge])
        verifier = VerificationAgent(llm_provider=fake_llm)

        v_res = verifier.execute(specialist_results=[spec_res])
        assert len(v_res.claims) == 1
        claim_res = v_res.claims[0]
        assert claim_res.status == expected_status, f"Failed for {claim_text} + {verdict}"
        assert claim_res.is_supported is expected_is_supported, f"Failed is_supported for {claim_text} + {verdict}"


def test_empty_findings_fails_verification_and_triggers_critic_and_human_gate():
    """Empty report (zero findings) has None metrics, quality_pass=False, and triggers human gate / critic failure."""
    verifier = VerificationAgent(llm_provider=FakeLLM())
    spec_res = AgentExecutionResult(
        subtask_id="st-empty",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Empty findings",
        findings=[],
    )

    v_res = verifier.execute(specialist_results=[spec_res])
    assert v_res.metrics.supported_ratio is None
    assert v_res.metrics.citation_coverage is None
    assert v_res.metrics.quality_pass is False
    assert v_res.audit_notes == "no findings to verify"

    # Critic must evaluate quality_pass as False
    critic = CriticAgent(llm_provider=FakeLLM(responses=[
        CriticRubricOutput(answers_question=8.0, uses_task_data=8.0, claims_verified=8.0, limitations_stated=8.0)
    ]))
    c_eval = critic.execute(specialist_results=[spec_res], verification_result=v_res)
    assert c_eval.quality_pass is False

    # Human review gate must treat quality failure as high risk
    orch = AegisOrchestrator(llm_provider=FakeLLM())
    gate_state: AegisState = {
        "task_id": "test-empty-gate",
        "query": "Test query",
        "verification_result": v_res,
        "critic_evaluation": c_eval,
        "require_human_approval": True,
    }
    gate_out = orch._human_gate_node(gate_state)
    approval = gate_out["human_approval"]
    assert approval.risk_level == "high"
    assert "quality failure" in approval.reason.lower() or "no verified findings" in approval.reason.lower()


def test_tool_name_citation_does_not_resolve():
    """Citing tool name (e.g. 'sql_query') instead of call_id must NOT resolve to evidence."""
    call_id = "real-call-id-123"
    tool_rec = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={},
        output={"pressure": 120.0},
        execution_time_ms=5.0,
        success=True,
    )
    # Finding cites tool_name "sql_query" as ref_id instead of "real-call-id-123"
    finding = Finding(
        statement="Pressure was 120.0.",
        evidence=[EvidenceRef(kind="tool_call", ref_id="sql_query")],
    )
    spec_res = AgentExecutionResult(
        subtask_id="st-cite",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Tool cite test",
        findings=[finding],
        tool_calls=[tool_rec],
    )

    verifier = VerificationAgent(llm_provider=FakeLLM())
    v_res = verifier.execute(specialist_results=[spec_res])

    assert len(v_res.claims) == 1
    assert v_res.claims[0].status == "no_evidence"
    assert v_res.claims[0].is_supported is False


def test_claim_verification_schema_requires_status():
    """ClaimVerification requires status (no default) and derives is_supported."""
    from aegis.models.schemas import ClaimVerification

    # Missing status raises ValidationError
    with pytest.raises(pydantic.ValidationError):
        ClaimVerification(claim="Test claim")

    # Provided status derives is_supported
    cv_sup = ClaimVerification(claim="Valid claim", status="supported")
    assert cv_sup.is_supported is True

    cv_unsup = ClaimVerification(claim="Invalid claim", status="unsupported")
    assert cv_unsup.is_supported is False

    cv_contra = ClaimVerification(claim="Contradicted claim", status="contradicted")
    assert cv_contra.is_supported is False
    """Deterministic numeric check extracts claim numbers and flags mismatch if not in evidence."""
    call_id = "call-telemetry-1"
    tool_record = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={"query": "SELECT max(pressure) FROM dataset"},
        output={"max_pressure": 320.5, "min_pressure": 110.0},
        execution_time_ms=10.0,
        success=True,
    )
    finding = Finding(
        statement="Max operating pressure reached 480.0 psi under peak load.",
        evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
    )
    spec_result = AgentExecutionResult(
        subtask_id="st-da",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Pressure analysis",
        findings=[finding],
        tool_calls=[tool_record],
    )

    # Even if LLM judge claims supported, numeric mismatch must override
    fake_judge_resp = JudgeVerdict(
        verdict="supported",
        quote_span="320.5",
        rationale="Numeric report",
    )
    fake_llm = FakeLLM(responses=[fake_judge_resp])
    verifier = VerificationAgent(llm_provider=fake_llm)

    res: VerificationResult = verifier.execute(
        specialist_results=[spec_result],
        original_query="What is max pressure?",
    )

    assert len(res.claims) == 1
    claim = res.claims[0]
    assert claim.numeric_mismatch is True
    assert claim.status == "unsupported"
    assert claim.is_supported is False
    assert res.metrics.numeric_mismatches == 1
    assert res.metrics.quality_pass is False
    assert res.status == "needs_revision"


def test_quote_span_not_in_evidence_downgraded():
    """If quote span cannot be found verbatim in evidence, verdict downgrades to not_enough_info / unsupported."""
    call_id = "call-rag-1"
    evidence_content = "ISO vibration criteria specifies vibration category A and B limits."
    tool_record = ToolCallRecord(
        call_id=call_id,
        tool_name="rag_search",
        arguments={"query": "ISO criteria"},
        output={"evidence": [{"chunk_id": "chunk-iso-1", "content": evidence_content}]},
        execution_time_ms=15.0,
        success=True,
    )
    finding = Finding(
        statement="Category A vibration limit is confirmed by ISO standards.",
        evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
    )
    spec_result = AgentExecutionResult(
        subtask_id="st-res",
        agent_role=AgentRole.RESEARCH,
        output_summary="ISO research",
        findings=[finding],
        tool_calls=[tool_record],
    )

    # Judge returns a fabricated quote span not found verbatim in evidence
    fabricated_quote = "Category A is fully verified and endorsed across all machinery classes."
    fake_judge_resp = JudgeVerdict(
        verdict="supported",
        quote_span=fabricated_quote,
        rationale="Looks verified",
    )
    fake_llm = FakeLLM(responses=[fake_judge_resp])
    verifier = VerificationAgent(llm_provider=fake_llm)

    res: VerificationResult = verifier.execute(
        specialist_results=[spec_result],
        original_query="Verify Category A criteria",
    )

    assert len(res.claims) == 1
    claim = res.claims[0]
    assert claim.status == "unsupported"
    assert claim.is_supported is False
    assert "downgraded" in claim.notes


def test_critic_deterministic_quality_pass():
    """Critic quality_pass is computed deterministically in code from scores and verification pass."""
    fake_llm = FakeLLM()
    critic = CriticAgent(llm_provider=fake_llm)

    # Verification passed, but rubric score is below 6.0 -> quality_pass must be False
    rubric_low = CriticRubricOutput(
        answers_question=5.0,  # below 6.0
        uses_task_data=8.0,
        claims_verified=8.0,
        limitations_stated=8.0,
        retry_subtask_ids=[],
        replan=False,
        guidance="Inadequate answer to prompt",
    )
    fake_llm.queue_response(rubric_low)

    v_pass = VerificationResult(
        status="passed",
        claims=[],
        metrics=VerificationMetrics(quality_pass=True, supported=1, supported_ratio=1.0),
    )
    eval_res: CriticEvaluation = critic.execute(
        specialist_results=[],
        verification_result=v_pass,
    )
    assert eval_res.quality_pass is False

    # Verification passed and all scores >= 6.0, no retries -> quality_pass is True
    rubric_high = CriticRubricOutput(
        answers_question=8.0,
        uses_task_data=8.5,
        claims_verified=9.0,
        limitations_stated=7.5,
        retry_subtask_ids=[],
        replan=False,
        guidance="All standards met",
    )
    fake_llm.queue_response(rubric_high)

    eval_res_pass: CriticEvaluation = critic.execute(
        specialist_results=[],
        verification_result=v_pass,
    )
    assert eval_res_pass.quality_pass is True


def test_retry_reexecutes_only_targeted_subtask_and_preserves_others():
    """Retry re-executes only the targeted subtask; other subtasks' results are preserved."""
    plan = ExecutionPlan(
        task_id="t-targeted-retry",
        objective="Inspect cooling system",
        subtasks=[
            SubTask(
                id="st-1",
                description="Literature search on cooling thresholds",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            ),
            SubTask(
                id="st-2",
                description="Telemetry profiling for coolant temperature",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query"],
                dependencies=["st-1"],
            ),
        ],
    )

    # Initial specialist responses
    st1_out_initial = SpecialistOutput(
        summary="Literature threshold is 85 C.",
        findings=[],
    )
    st2_out_initial = SpecialistOutput(
        summary="Initial coolant temperature was inconclusive.",
        findings=[Finding(statement="Sensor readings incomplete.")],  # unverified -> fails verification
    )

    # Critic requests retry ONLY on st-2
    critic_request_retry = CriticRubricOutput(
        answers_question=7.0,
        uses_task_data=6.0,
        claims_verified=6.0,
        limitations_stated=7.0,
        retry_subtask_ids=["st-2"],  # Targeted retry on st-2 only
        replan=False,
        guidance="Re-run SQL query filtering valid temperature records.",
    )

    # Targeted retry of st-2 executes tool query and returns verified finding
    call_id_st2 = "call-st2-query"
    tool_resp_st2 = fake_tool_call(
        name="sql_query",
        args={"sql": "SELECT 82.0 as temp"},
        tool_call_id=call_id_st2,
    )
    st2_out_retry = SpecialistOutput(
        summary="Coolant temperature peaked at 82 C.",
        findings=[
            Finding(
                statement="Coolant temperature peaked at 82.0 C.",
                evidence=[EvidenceRef(kind="tool_call", ref_id=call_id_st2)],
            )
        ],
    )
    judge_pass = JudgeVerdict(
        verdict="supported",
        quote_span="82.0",
        rationale="Matches SQL output",
    )

    # Critic passes
    critic_pass = CriticRubricOutput(
        answers_question=9.0,
        uses_task_data=9.0,
        claims_verified=9.0,
        limitations_stated=8.5,
        retry_subtask_ids=[],
        replan=False,
        guidance="Verified successfully.",
    )

    # Synthesizer final draft
    synth_draft = SynthesizerOutput(
        title="Cooling System Assessment",
        final_finding="Coolant temperature remained within safe limits.",
        key_conclusions=["Temperature peak was 82.0 C, below 85 C limit."],
        actionable_recommendations=["Continue routine scheduled inspection."],
    )

    fake_llm = FakeLLM(
        responses=[
            plan,
            fake_text("Searching docs"),
            st1_out_initial,
            fake_text("Running telemetry SQL"),
            st2_out_initial,
            critic_request_retry,
            # Targeted retry of st-2 ONLY (st-1 is NOT called again!)
            tool_resp_st2,
            fake_text("Re-running SQL with filter"),
            st2_out_retry,
            judge_pass,
            critic_pass,
            synth_draft,
        ]
    )

    orch = AegisOrchestrator(llm_provider=fake_llm)
    state: AegisState = {
        "task_id": "t-targeted-retry",
        "query": "Assess coolant temperature",
        "has_dataset": True,
        "require_human_approval": False,
        "max_retries": 2,
        "retry_count": 0,
    }

    result = orch.run(state)

    assert result["status"] == TaskStatus.COMPLETED
    spec_results = result["specialist_results"]
    assert "st-1" in spec_results
    assert "st-2" in spec_results
    # st-1 preserved original findings
    assert spec_results["st-1"].output_summary == "Literature threshold is 85 C."
    # st-2 was updated with retry findings
    assert spec_results["st-2"].output_summary == "Coolant temperature peaked at 82 C."


def test_retry_cap_reached_routes_to_synthesis():
    """When retry cap is reached, orchestrator proceeds directly to synthesis with limitations."""
    plan = ExecutionPlan(
        task_id="t-cap",
        objective="Inspect unit",
        subtasks=[
            SubTask(
                id="st-1",
                description="Check metrics",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            )
        ],
    )
    spec_out = SpecialistOutput(summary="Done", findings=[Finding(statement="Claim.")])

    critic_fail_1 = CriticRubricOutput(
        answers_question=5.0,
        uses_task_data=5.0,
        claims_verified=5.0,
        limitations_stated=5.0,
        retry_subtask_ids=["st-1"],
        replan=False,
        guidance="Retry 1",
    )
    critic_fail_2 = CriticRubricOutput(
        answers_question=5.0,
        uses_task_data=5.0,
        claims_verified=5.0,
        limitations_stated=5.0,
        retry_subtask_ids=["st-1"],
        replan=False,
        guidance="Retry 2",
    )
    synth_draft = SynthesizerOutput(
        title="Cap Reached Report",
        final_finding="Inconclusive telemetry",
        key_conclusions=[],
        actionable_recommendations=[],
    )

    fake_llm = FakeLLM(
        responses=[
            plan,
            fake_text("Pass 0"),
            spec_out,
            critic_fail_1,
            # Retry 1
            fake_text("Pass 1"),
            spec_out,
            critic_fail_2,
            # Cap of 1 reached -> proceeds to synthesis
            synth_draft,
        ]
    )

    orch = AegisOrchestrator(llm_provider=fake_llm)
    state: AegisState = {
        "task_id": "t-cap",
        "query": "Check metrics",
        "has_dataset": False,
        "require_human_approval": False,
        "max_retries": 1,  # Cap is 1 retry
        "retry_count": 0,
    }

    result = orch.run(state)
    if result["status"] == TaskStatus.WAITING_APPROVAL:
        result = orch.resume("t-cap", {"action": "approve"})
    assert result["status"] == TaskStatus.COMPLETED
    assert result["final_report"] is not None
    assert any("Maximum retry limit reached" in lim for lim in result["final_report"].limitations)


def test_different_questions_produce_different_final_reports():
    """Different questions produce different final reports: verified end-to-end."""
    plan_a = ExecutionPlan(
        task_id="t-qa",
        objective="Analyze bearing vibration",
        subtasks=[SubTask(id="st-a", description="Vibration analysis", assigned_agent=AgentRole.RESEARCH)],
    )
    spec_a = SpecialistOutput(
        summary="Bearing velocity is nominal.",
        findings=[],
    )
    critic_a = CriticRubricOutput(
        answers_question=9.0, uses_task_data=9.0, claims_verified=9.0, limitations_stated=9.0
    )
    synth_a = SynthesizerOutput(
        title="Bearing Vibration Assessment",
        final_finding="Bearings operating within Category A criteria.",
        key_conclusions=["Velocity RMS 2.1 mm/s is acceptable."],
        actionable_recommendations=["Standard monitoring interval."],
    )

    plan_b = ExecutionPlan(
        task_id="t-qb",
        objective="Assess heat exchanger thermal efficiency",
        subtasks=[SubTask(id="st-b", description="Thermal efficiency", assigned_agent=AgentRole.RESEARCH)],
    )
    spec_b = SpecialistOutput(
        summary="Heat exchanger fouling detected.",
        findings=[],
    )
    critic_b = CriticRubricOutput(
        answers_question=8.5, uses_task_data=8.5, claims_verified=8.5, limitations_stated=8.5
    )
    synth_b = SynthesizerOutput(
        title="Heat Exchanger Thermal Diagnostic",
        final_finding="Thermal fouling coefficient indicates severe scaling.",
        key_conclusions=["Heat transfer efficiency down by 28%."],
        actionable_recommendations=["Chemical flushing required immediately."],
    )

    fake_llm_a = FakeLLM(responses=[plan_a, fake_text("A"), spec_a, critic_a, synth_a])
    fake_llm_b = FakeLLM(responses=[plan_b, fake_text("B"), spec_b, critic_b, synth_b])

    orch_a = AegisOrchestrator(llm_provider=fake_llm_a)
    orch_b = AegisOrchestrator(llm_provider=fake_llm_b)

    res_a = orch_a.run({"task_id": "t-qa", "query": "Inspect bearing vibration", "require_human_approval": False, "max_retries": 0})
    res_b = orch_b.run({"task_id": "t-qb", "query": "Assess heat exchanger", "require_human_approval": False, "max_retries": 0})

    rep_a = res_a["final_report"]
    rep_b = res_b["final_report"]

    assert rep_a.title != rep_b.title
    assert rep_a.executive_summary.final_finding != rep_b.executive_summary.final_finding
    assert rep_a.executive_summary.key_conclusions != rep_b.executive_summary.key_conclusions
    assert rep_a.executive_summary.actionable_recommendations != rep_b.executive_summary.actionable_recommendations
    assert rep_a.executive_summary.overall_confidence is None
    assert rep_b.executive_summary.overall_confidence is None
    assert rep_a.executive_summary.business_impact == ""
    assert rep_b.executive_summary.business_impact == ""


def test_judge_model_set_builds_judge_with_that_model_and_counts_calls(monkeypatch):
    """When judge_model is set, the judge is built with that model and the task's provider,
    and its calls are counted under that model."""
    task_settings = Settings(provider="openai", model="gpt-4o", judge_model="gpt-4o-mini")
    task_llm = FakeLLM(model="gpt-4o", settings=task_settings)

    call_id = "call-judge-1"
    evidence_text = "Operational vibration RMS is 2.1 mm/s."
    tool_rec = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={},
        output={"text": evidence_text},
        execution_time_ms=5.0,
        success=True,
    )
    finding = Finding(
        statement="Operational vibration RMS is 2.1 mm/s.",
        evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
    )
    spec_res = AgentExecutionResult(
        subtask_id="st-test",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Audit output",
        findings=[finding],
        tool_calls=[tool_rec],
    )

    fake_judge_llm = FakeLLM(
        responses=[JudgeVerdict(verdict="supported", quote_span="2.1 mm/s", rationale="Matches evidence")],
        model="gpt-4o-mini",
    )

    created_with_settings = []

    def mock_get_llm_provider(settings=None):
        created_with_settings.append(settings)
        return fake_judge_llm

    monkeypatch.setattr("aegis.llm.get_llm_provider", mock_get_llm_provider)

    verifier = VerificationAgent(llm_provider=task_llm)
    v_res = verifier.execute(specialist_results=[spec_res], original_query="Check vibration")

    # Built with that model and the task's provider
    assert len(created_with_settings) == 1
    assert created_with_settings[0].provider == "openai"
    assert created_with_settings[0].model == "gpt-4o-mini"

    # Judge called, task LLM not called
    assert len(fake_judge_llm.calls) == 1
    assert len(task_llm.calls) == 0

    # Calls counted under that model
    assert v_res.metadata["model"] == "gpt-4o-mini"
    assert v_res.metadata["judge_model"] == "gpt-4o-mini"
    assert v_res.metadata["call_count"] == 1


def test_judge_model_unset_uses_self_llm():
    """When judge_model is unset, the judge is self.llm."""
    task_settings = Settings(provider="openai", model="gpt-4o", judge_model=None)
    fake_judge_verdict = JudgeVerdict(verdict="supported", quote_span="2.1 mm/s", rationale="Matches evidence")
    task_llm = FakeLLM(responses=[fake_judge_verdict], model="gpt-4o", settings=task_settings)

    verifier = VerificationAgent(llm_provider=task_llm)
    judge_provider, judge_model_name = verifier._get_judge_llm()

    assert judge_provider is task_llm
    assert judge_model_name == "gpt-4o"

    call_id = "call-judge-2"
    tool_rec = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={},
        output={"text": "Operational vibration RMS is 2.1 mm/s."},
        execution_time_ms=5.0,
        success=True,
    )
    finding = Finding(
        statement="Operational vibration RMS is 2.1 mm/s.",
        evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
    )
    spec_res = AgentExecutionResult(
        subtask_id="st-test",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Audit output",
        findings=[finding],
        tool_calls=[tool_rec],
    )

    v_res = verifier.execute(specialist_results=[spec_res], original_query="Check vibration")
    assert len(task_llm.calls) == 1
    assert v_res.metadata["model"] == "gpt-4o"
    assert v_res.metadata["judge_model"] == "gpt-4o"
    assert v_res.metadata["call_count"] == 1


def test_end_to_end_mocked_graph_run_with_judge_model_set_reaches_final_report(monkeypatch):
    """End-to-end mocked graph run with judge_model set reaches FinalReport and uses judge_model."""
    task_settings = Settings(provider="openai", model="gpt-4o", judge_model="gpt-4o-judge")

    df = pd.DataFrame({"sensor_1": [10.0, 20.0, 30.0], "vibration": [2.1, 2.1, 2.1]})
    task_ctx = TaskContext(dataset=df)

    call_id = "call-da-e2e-1"
    plan = ExecutionPlan(
        task_id="t-judge-e2e",
        objective="Inspect vibration anomalies",
        subtasks=[
            SubTask(
                id="st-1",
                description="Query dataset metrics",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query"],
            )
        ],
    )
    tool_resp = fake_tool_call(
        name="sql_query",
        args={"query": "SELECT AVG(vibration) as avg_vib FROM dataset;"},
        tool_call_id=call_id,
    )
    spec_text = fake_text("Vibration average is 2.1 mm/s.")
    spec_out = SpecialistOutput(
        summary="Analysis complete.",
        findings=[
            Finding(
                statement="Dataset vibration average is 2.1 mm/s.",
                evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
            )
        ],
        limitations=[],
    )
    critic_pass = CriticRubricOutput(
        answers_question=9.0,
        uses_task_data=9.0,
        claims_verified=9.0,
        limitations_stated=9.0,
    )
    synth_report = SynthesizerOutput(
        title="Vibration Diagnostic Report",
        final_finding="Vibration is within acceptable ISO tolerances.",
        key_conclusions=["2.1 mm/s average velocity RMS."],
        actionable_recommendations=["Maintain standard monitoring."],
    )

    fake_task_llm = FakeLLM(
        responses=[
            plan,
            tool_resp,
            spec_text,
            spec_out,
            critic_pass,
            synth_report,
        ],
        model="gpt-4o",
        settings=task_settings,
    )

    fake_judge_llm = FakeLLM(
        responses=[
            JudgeVerdict(verdict="supported", quote_span="2.1", rationale="Query result matches claim"),
        ],
        model="gpt-4o-judge",
    )

    created_judges = []

    def mock_get_llm_provider(settings=None):
        created_judges.append(settings)
        return fake_judge_llm

    monkeypatch.setattr("aegis.llm.get_llm_provider", mock_get_llm_provider)

    try:
        orch = AegisOrchestrator(llm_provider=fake_task_llm, task_context=task_ctx)
        result = orch.run({
            "task_id": "t-judge-e2e",
            "query": "Check vibration metrics",
            "has_dataset": True,
            "require_human_approval": False,
            "max_retries": 0,
        })

        assert result["status"] == TaskStatus.COMPLETED
        assert result["final_report"] is not None
        assert isinstance(result["final_report"], FinalReport)

        # Verification used judge_llm
        assert len(created_judges) == 1
        assert created_judges[0].model == "gpt-4o-judge"
        assert created_judges[0].provider == "openai"
        assert len(fake_judge_llm.calls) == 1

        # Check trace events recorded judge model
        verify_events = [e for e in result["trace_events"] if e.event_type == "verify" and e.model == "gpt-4o-judge"]
        assert len(verify_events) >= 1
        assert verify_events[0].call_count == 1
    finally:
        task_ctx.cleanup()

