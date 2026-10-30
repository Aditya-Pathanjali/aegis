"""Unit tests for individual specialist agents."""

from aegis.agents.critic import CriticAgent
from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.agents.research import ResearchAgent
from aegis.agents.synthesizer import SynthesizerAgent
from aegis.agents.verification import VerificationAgent
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    CriticEvaluation,
    ExecutionPlan,
    Finding,
    SpecialistOutput,
    SubTask,
    VerificationMetrics,
    VerificationResult,
)
from tests.fakes import FakeLLM, fake_text


def test_planner_agent():
    expected_plan = ExecutionPlan(
        task_id="plan-1",
        original_query="Analyse equipment data",
        objective="Analyze system telemetry and research guidelines",
        subtasks=[
            SubTask(
                id="st-1",
                title="Engineering Literature Review",
                description="Review operating guidelines",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            ),
            SubTask(
                id="st-2",
                title="Data Profiling",
                description="Profile dataset statistics",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query"],
            ),
        ],
    )
    fake_llm = FakeLLM(responses=[expected_plan])
    planner = PlannerAgent(llm_provider=fake_llm)
    plan = planner.execute(
        query="Analyse equipment data",
        has_dataset=True,
    )
    assert len(plan.subtasks) == 2
    roles = [st.assigned_agent for st in plan.subtasks]
    assert AgentRole.RESEARCH in roles
    assert AgentRole.DATA_ANALYST in roles


def test_research_agent():
    out = SpecialistOutput(
        summary="Retrieved engineering standards for vibration limits.",
        findings=[Finding(statement="Vibration threshold standard defined.", evidence=[])],
        limitations=[],
    )
    fake_llm = FakeLLM(responses=[fake_text("Done searching."), out])
    researcher = ResearchAgent(llm_provider=fake_llm)
    st = SubTask(
        id="st-res",
        title="Literature Search",
        description="Lookup engineering guidelines",
        assigned_agent=AgentRole.RESEARCH,
    )
    res = researcher.execute(subtask=st)
    assert res.output_summary == "Retrieved engineering standards for vibration limits."
    assert len(res.findings) == 1


def test_data_analyst_agent():
    out = SpecialistOutput(
        summary="Data profiling complete.",
        findings=[Finding(statement="Dataset has 10 columns.", evidence=[])],
        limitations=[],
    )
    fake_llm = FakeLLM(responses=[fake_text("Done profiling."), out])
    analyst = DataAnalystAgent(llm_provider=fake_llm)
    st = SubTask(
        id="st-data",
        title="Data Analysis",
        description="Compute summary stats",
        assigned_agent=AgentRole.DATA_ANALYST,
    )
    res = analyst.execute(subtask=st)
    assert res.output_summary == "Data profiling complete."
    assert len(res.findings) == 1


def test_verification_agent():
    verifier = VerificationAgent(llm_provider=FakeLLM())
    out = SpecialistOutput(
        summary="Data analysis output",
        findings=[],
        limitations=[],
    )
    analyst = DataAnalystAgent(llm_provider=FakeLLM(responses=[fake_text("ok"), out]))
    res = analyst.execute(
        subtask=SubTask(id="st", title="test", description="test", assigned_agent=AgentRole.DATA_ANALYST)
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze equipment")
    # Per Blocker 4: empty findings has status needs_revision, quality_pass False, None metrics
    assert v_res.status == "needs_revision"
    assert v_res.metrics.supported_ratio is None
    assert v_res.metrics.quality_pass is False
    assert v_res.claims == []


def test_critic_agent():
    from aegis.agents.critic import CriticRubricOutput
    from aegis.agents.verification import JudgeVerdict
    from aegis.models.schemas import EvidenceRef, ToolCallRecord

    rubric = CriticRubricOutput(
        answers_question=8.0,
        uses_task_data=8.0,
        claims_verified=8.0,
        limitations_stated=8.0,
        retry_subtask_ids=[],
        replan=False,
        guidance="Well substantiated analysis.",
    )
    judge_pass = JudgeVerdict(verdict="supported", quote_span="12.5", rationale="Matches SQL")
    critic = CriticAgent(llm_provider=FakeLLM(responses=[rubric]))
    verifier = VerificationAgent(llm_provider=FakeLLM(responses=[judge_pass]))
    call_id = "call-analyst-1"
    tool_rec = ToolCallRecord(
        call_id=call_id,
        tool_name="sql_query",
        arguments={},
        output={"pressure": 12.5},
        execution_time_ms=5.0,
        success=True,
    )
    finding = Finding(
        statement="Hydraulic pressure was 12.5 bar.",
        evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
    )
    res = AgentExecutionResult(
        subtask_id="st",
        agent_role=AgentRole.DATA_ANALYST,
        output_summary="Data analysis output",
        findings=[finding],
        tool_calls=[tool_rec],
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze")
    assert v_res.metrics.quality_pass is True
    eval_res = critic.execute(specialist_results=[res], verification_result=v_res)
    assert eval_res.quality_pass is True
    assert eval_res.score == 8.0
    assert eval_res.guidance == "Well substantiated analysis."


def test_synthesizer_agent():
    from aegis.models.schemas import SynthesizerOutput

    draft = SynthesizerOutput(
        title="Equipment Health Inspection",
        final_finding="Telemetry and research confirm operational integrity.",
        key_conclusions=["All measured parameters within tolerance."],
        actionable_recommendations=["Maintain regular inspection cadence."],
    )
    synthesizer = SynthesizerAgent(llm_provider=FakeLLM(responses=[draft]))
    report = synthesizer.execute(
        task_id="t-synth",
        query="Inspect system",
        specialist_results=[],
        verification_result=VerificationResult(
            status="passed",
            claims=[],
            metrics=VerificationMetrics(quality_pass=True),
            unsupported_claims=[],
        ),
        critic_evaluation=CriticEvaluation(
            quality_pass=True,
            score=8.0,
            guidance="Approved",
        ),
        trace_events=[],
    )
    assert report.executive_summary.final_finding == "Telemetry and research confirm operational integrity."
    assert len(report.executive_summary.key_conclusions) == 1
    assert len(report.executive_summary.actionable_recommendations) == 1
    assert report.executive_summary.overall_confidence is None
    assert report.executive_summary.business_impact == ""
