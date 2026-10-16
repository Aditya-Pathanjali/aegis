"""Unit tests for individual specialist agents."""

from aegis.agents.coding import CodingAgent
from aegis.agents.critic import CriticAgent
from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.agents.research import ResearchAgent
from aegis.agents.synthesizer import SynthesizerAgent
from aegis.agents.verification import VerificationAgent
from aegis.models.schemas import AgentRole, SubTask


def test_planner_agent():
    planner = PlannerAgent()
    plan = planner.execute(
        query="Analyse whether predictive maintenance would be suitable for this equipment using sensor data.",
        has_dataset=True,
    )
    assert len(plan.subtasks) >= 2
    roles = [st.assigned_agent for st in plan.subtasks]
    assert AgentRole.RESEARCH in roles or AgentRole.DATA_ANALYST in roles


def test_research_agent():
    researcher = ResearchAgent()
    st = SubTask(
        id="st-res",
        title="ISO 10816 Vibration Limits",
        description="Lookup vibration severity zones",
        assigned_agent=AgentRole.RESEARCH,
    )
    res = researcher.execute(subtask=st)
    assert res.confidence_score >= 0.85
    assert len(res.detailed_findings) > 0
    assert len(res.tool_calls) == 2


def test_data_analyst_agent():
    analyst = DataAnalystAgent()
    st = SubTask(
        id="st-data",
        title="Analyze Sensor Telemetry",
        description="Compute summary stats and Pearson correlation",
        assigned_agent=AgentRole.DATA_ANALYST,
    )
    res = analyst.execute(subtask=st)
    assert res.confidence_score >= 0.90
    assert len(res.data_tables) >= 2
    assert any("Correlation" in dt.get("table_name", "") for dt in res.data_tables)


def test_coding_agent():
    coder = CodingAgent()
    st = SubTask(
        id="st-code",
        title="Exponential Wear Regression",
        description="Fit degradation curve",
        assigned_agent=AgentRole.CODING,
    )
    res = coder.execute(subtask=st)
    assert len(res.code_snippets) > 0
    assert "rul_days" in res.metadata


def test_verification_agent():
    verifier = VerificationAgent()
    analyst = DataAnalystAgent()
    res = analyst.execute(
        subtask=SubTask(id="st", title="test", description="test", assigned_agent=AgentRole.DATA_ANALYST)
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze predictive maintenance")
    assert v_res.metrics.overall_reliability > 80.0
    assert v_res.metrics.hallucination_risk < 15.0
    assert v_res.status in ["passed", "needs_revision"]


def test_critic_agent():
    critic = CriticAgent()
    verifier = VerificationAgent()
    analyst = DataAnalystAgent()
    res = analyst.execute(
        subtask=SubTask(id="st", title="test", description="test", assigned_agent=AgentRole.DATA_ANALYST)
    )
    v_res = verifier.execute(specialist_results=[res], original_query="Analyze")
    eval_res = critic.execute(specialist_results=[res], verification_result=v_res)
    assert eval_res.quality_pass is True
    assert eval_res.score >= 8.0
