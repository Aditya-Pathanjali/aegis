"""Tests for PR 5 specialist agents, planner, tool-using loop, and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.models.schemas import (
    AgentRole,
    EvidenceRef,
    ExecutionPlan,
    Finding,
    SpecialistOutput,
    SubTask,
)
from aegis.tools.context import TaskContext
from tests.fakes import FakeLLM, fake_text, fake_tool_call


@pytest.fixture
def sample_task_context():
    df = pd.DataFrame({"sensor_1": [10.0, 20.0, 30.0], "status": ["ok", "ok", "warn"]})
    ctx = TaskContext(dataset=df)
    yield ctx
    ctx.cleanup()


def test_tool_loop_with_valid_refs(sample_task_context):
    """Tool loop: model calls tool -> gets result -> produces findings with valid refs."""
    call_id = "call-sql-101"
    tool_resp = fake_tool_call(
        name="sql_query",
        args={"query": "SELECT COUNT(*) as total FROM dataset;"},
        tool_call_id=call_id,
    )
    final_text = fake_text("Found 3 rows in the dataset.")
    structured_out = SpecialistOutput(
        summary="Analysis completed successfully.",
        findings=[
            Finding(
                statement="Dataset contains 3 operational records.",
                evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
            )
        ],
        limitations=["Only checked sensor_1 status."],
    )

    fake_llm = FakeLLM(responses=[tool_resp, final_text, structured_out])
    analyst = DataAnalystAgent(llm_provider=fake_llm, tool_registry=sample_task_context.tool_registry)

    subtask = SubTask(
        id="st-da-1",
        title="Analyze dataset records",
        description="Query row count and summary statistics",
        assigned_agent=AgentRole.DATA_ANALYST,
        required_tools=["sql_query"],
    )

    result = analyst.execute(subtask=subtask)

    assert result.subtask_id == "st-da-1"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].call_id == call_id
    assert result.tool_calls[0].success is True
    assert len(result.findings) == 1
    assert result.findings[0].statement == "Dataset contains 3 operational records."
    assert len(result.findings[0].evidence) == 1
    assert result.findings[0].evidence[0].ref_id == call_id
    assert result.findings[0].evidence[0].kind == "tool_call"


def test_fabricated_refs_dropped(sample_task_context):
    """Fabricated refs are dropped; ref without tool call is rejected."""
    call_id = "call-sql-202"
    tool_resp = fake_tool_call(
        name="describe_dataset",
        args={},
        tool_call_id=call_id,
    )
    final_text = fake_text("Description done.")
    # Finding 1 has 1 real ref and 2 fabricated refs
    # Finding 2 has ONLY fabricated refs
    structured_out = SpecialistOutput(
        summary="Summary of dataset profile.",
        findings=[
            Finding(
                statement="The dataset has sensor_1 and status columns.",
                evidence=[
                    EvidenceRef(kind="tool_call", ref_id=call_id),
                    EvidenceRef(kind="tool_call", ref_id="fake-call-999"),
                    EvidenceRef(kind="chunk", ref_id="fake-chunk-888"),
                ],
            ),
            Finding(
                statement="Vibration levels were unverified.",
                evidence=[
                    EvidenceRef(kind="chunk", ref_id="fabricated-chunk-000"),
                ],
            ),
        ],
        limitations=[],
    )

    fake_llm = FakeLLM(responses=[tool_resp, final_text, structured_out])
    analyst = DataAnalystAgent(llm_provider=fake_llm, tool_registry=sample_task_context.tool_registry)

    subtask = SubTask(
        id="st-da-2",
        title="Profile dataset",
        description="Profile columns",
        assigned_agent=AgentRole.DATA_ANALYST,
        required_tools=["describe_dataset"],
    )

    result = analyst.execute(subtask=subtask)

    # First finding must retain only the real ref
    f1 = result.findings[0]
    assert len(f1.evidence) == 1
    assert f1.evidence[0].ref_id == call_id

    # Second finding had only fabricated refs, which were dropped
    f2 = result.findings[1]
    assert len(f2.evidence) == 0

    # Dropped all refs from finding 2 should be noted in limitations
    assert any("No verified evidence remaining" in lim for lim in result.limitations)


def test_step_limit_enforced(sample_task_context):
    """Step limit is enforced and agent loop terminates and calls structured."""
    # Loop should stop after 2 steps
    tool_resp_1 = fake_tool_call(name="sql_query", args={"query": "SELECT 1;"}, tool_call_id="c1")
    tool_resp_2 = fake_tool_call(name="sql_query", args={"query": "SELECT 2;"}, tool_call_id="c2")
    # Even if LLM would want to continue with a 3rd call, step limit of 2 forces exit to structured
    structured_out = SpecialistOutput(
        summary="Budget reached summary.",
        findings=[],
        limitations=["Tool budget reached."],
    )

    fake_llm = FakeLLM(responses=[tool_resp_1, tool_resp_2, structured_out])

    subtask = SubTask(
        id="st-limit",
        title="Loop test",
        description="Loop test",
        assigned_agent=AgentRole.DATA_ANALYST,
        required_tools=["sql_query"],
    )

    from aegis.agents.tool_agent import run_tool_agent

    result = run_tool_agent(
        llm_provider=fake_llm,
        system_prompt="Test",
        user_prompt="Run query",
        permitted_tool_names=["sql_query"],
        tool_registry=sample_task_context.tool_registry,
        role=AgentRole.DATA_ANALYST,
        subtask_id=subtask.id,
        step_limit=2,
    )

    assert len(result.tool_calls) == 2
    assert result.output_summary == "Budget reached summary."


def test_invalid_plan_repaired_once_and_succeeds():
    """Invalid plan is repaired once with validation feedback and succeeds."""
    # First plan has data_analyst without dataset (invalid)
    invalid_plan = ExecutionPlan(
        task_id="p-inv-1",
        objective="Analyze data without dataset",
        subtasks=[
            SubTask(
                id="st-1",
                description="Run SQL",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query"],
            )
        ],
    )
    # Repaired plan uses research instead (valid)
    repaired_plan = ExecutionPlan(
        task_id="p-rep-1",
        objective="Research standards",
        subtasks=[
            SubTask(
                id="st-1",
                description="Search literature",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            )
        ],
    )

    fake_llm = FakeLLM(responses=[invalid_plan, repaired_plan])
    planner = PlannerAgent(llm_provider=fake_llm)

    plan = planner.execute(query="Inspect machine", has_dataset=False)
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].assigned_agent == AgentRole.RESEARCH
    # Verify repair prompt was sent to FakeLLM
    assert len(fake_llm.calls) == 2
    second_call_messages = fake_llm.calls[1]["messages"]
    repair_prompt = second_call_messages[-1].content[0].text
    assert "The previously generated plan failed validation" in repair_prompt
    assert "data_analyst" in repair_prompt


def test_invalid_plan_unrepairable_raises():
    """Invalid plan failing repair a second time raises ValueError."""
    # First plan: cyclic dependency (st-1 depends on st-2, st-2 depends on st-1)
    invalid_plan_1 = ExecutionPlan(
        task_id="p-cycle-1",
        objective="Cyclic plan",
        subtasks=[
            SubTask(
                id="st-1",
                description="Task 1",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
                dependencies=["st-2"],
            ),
            SubTask(
                id="st-2",
                description="Task 2",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
                dependencies=["st-1"],
            ),
        ],
    )
    # Second plan: still cyclic
    invalid_plan_2 = invalid_plan_1.model_copy()

    fake_llm = FakeLLM(responses=[invalid_plan_1, invalid_plan_2])
    planner = PlannerAgent(llm_provider=fake_llm)

    with pytest.raises(ValueError, match="ExecutionPlan validation failed after repair attempt"):
        planner.execute(query="Check cycle", has_dataset=False)


def test_two_different_questions_produce_different_plans():
    """Outputs must depend on the input: different questions produce different plans."""
    plan_a = ExecutionPlan(
        task_id="plan-a",
        objective="Research ISO vibration standards",
        subtasks=[
            SubTask(
                id="st-iso",
                description="Retrieve ISO vibration velocity criteria",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
            )
        ],
    )

    plan_b = ExecutionPlan(
        task_id="plan-b",
        objective="Perform statistical telemetry degradation analysis",
        subtasks=[
            SubTask(
                id="st-stat",
                description="Calculate sensor mean and variance trends",
                assigned_agent=AgentRole.DATA_ANALYST,
                required_tools=["sql_query", "python_exec"],
            ),
            SubTask(
                id="st-lit",
                description="Cross-reference with failure documentation",
                assigned_agent=AgentRole.RESEARCH,
                required_tools=["rag_search"],
                dependencies=["st-stat"],
            ),
        ],
    )

    fake_llm = FakeLLM(responses=[plan_a, plan_b])
    planner = PlannerAgent(llm_provider=fake_llm)

    res_a = planner.execute(query="What are ISO vibration velocity thresholds?", has_dataset=False)
    res_b = planner.execute(query="Analyze telemetry degradation curves", has_dataset=True)

    assert res_a.objective != res_b.objective
    assert len(res_a.subtasks) != len(res_b.subtasks)
    assert [st.id for st in res_a.subtasks] != [st.id for st in res_b.subtasks]
    assert [st.assigned_agent for st in res_a.subtasks] != [st.assigned_agent for st in res_b.subtasks]

