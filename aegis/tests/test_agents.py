"""Unit tests for individual specialist agents."""

from aegis.agents.critic import CriticAgent
from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.agents.research import ResearchAgent
from aegis.agents.synthesizer import SynthesizerAgent
from aegis.agents.verification import VerificationAgent
from aegis.models.schemas import (
    AgentRole,
    CriticEvaluation,
    ExecutionPlan,
    SubTask,
    VerificationMetrics,
    VerificationResult,
)
from tests.fakes import FakeLLM


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
            ),
            SubTask(
                id="st-2",
                title="Data Profiling",
                description="Profile dataset statistics",
                assigned_agent=AgentRole.DATA_ANALYST,
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
    researcher = ResearchAgent(llm_provider=FakeLLM())
    st = SubTask(
        id="st-res",
        title="Literature Search",
        description="Lookup engineering guidelines",
        assigned_agent=AgentRole.RESEARCH,
    )
    res = researcher.execute(subtask=st)
    assert res.output_summary == "Research agent not implemented yet (PR 5)"
    assert res.detailed_findings == []
    assert res.metadata["limitations"] == ["Research agent not implemented yet (PR 5)"]


def test_data_analyst_agent():
    analyst = DataAnalystAgent(llm_provider=FakeLLM())
    st = SubTask(
        id="st-data",
        title="Data Analysis",
        description="Compute summary stats",
        assigned_agent=AgentRole.DATA_ANALYST,
    )
    res = analyst.execute(subtask=st)
    assert res.output_summary == "Data analyst agent not implemented yet (PR 5)"
    assert res.detailed_findings == []
    assert res.data_tables == []
    assert res.metadata["limitations"] == ["Data analyst agent not implemented yet (PR 5)"]


def test_verification_agent():
    verifier = VerificationAgent(llm_provider=FakeLLM())
    analyst = DataAnalystAgent(llm_provider=FakeLLM())
    res = analyst.execute(
        subtask=SubTask(id="st", title="test", description="test", assigned_agent=AgentRole.DATA_ANALYST)
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze equipment")
    assert v_res.status == "not_implemented"
    assert v_res.metrics.overall_reliability == 0.0
    assert v_res.claims == []


def test_critic_agent():
    critic = CriticAgent(llm_provider=FakeLLM())
    verifier = VerificationAgent(llm_provider=FakeLLM())
    analyst = DataAnalystAgent(llm_provider=FakeLLM())
    res = analyst.execute(
        subtask=SubTask(id="st", title="test", description="test", assigned_agent=AgentRole.DATA_ANALYST)
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze")
    eval_res = critic.execute(specialist_results=[res], verification_result=v_res)
    assert eval_res.quality_pass is True
    assert eval_res.critique_summary == "Critic not implemented yet (PR 6)"
    assert eval_res.score == 0.0


def test_synthesizer_agent():
    synthesizer = SynthesizerAgent(llm_provider=FakeLLM())
    report = synthesizer.execute(
        task_id="t-synth",
        query="Inspect system",
        specialist_results=[],
        verification_result=VerificationResult(
            status="not_implemented",
            claims=[],
            metrics=VerificationMetrics(
                evidence_coverage=0.0,
                source_quality=0.0,
                claim_consistency=0.0,
                temporal_validity=0.0,
                hallucination_risk=0.0,
                overall_reliability=0.0,
            ),
            unsupported_claims=[],
        ),
        critic_evaluation=CriticEvaluation(
            quality_pass=True,
            score=0.0,
            issues_found=[],
            missing_evidence=[],
            weak_reasoning=[],
            contradictions=[],
            critique_summary="Critic not implemented yet (PR 6)",
        ),
        trace_events=[],
    )
    assert report.executive_summary.final_finding == ""
    assert report.executive_summary.key_conclusions == []
    assert report.executive_summary.actionable_recommendations == []
    assert report.limitations == ["Synthesizer agent not implemented yet (PR 5/6)"]
