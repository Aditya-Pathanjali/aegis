"""Offline verification tests for C-MAPSS workflow using FakeLLM and real tool execution."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from aegis.agents.critic import CriticRubricOutput
from aegis.agents.verification import JudgeVerdict
from aegis.models.llm_outputs import (
    AgentRole as LLMAgentRole,
)
from aegis.models.llm_outputs import (
    PlannedSubtask,
    PlanOutput,
    SynthesizerOutput,
)
from aegis.models.schemas import (
    EvidenceRef,
    Finding,
    SpecialistOutput,
    TaskStatus,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry
from tests.fakes import FakeLLM, fake_text, fake_tool_call
from tests.live.test_live_workflow import assert_live_workflow_result


def _build_cmapss_fake_llm(include_192_in_report: bool = True) -> FakeLLM:
    """Builds a FakeLLM instance scripted for a valid C-MAPSS workflow run."""
    plan = PlanOutput(
        objective="Analyze engine 1 operational degradation and maximum time cycles in C-MAPSS dataset",
        rationale="Literature review and empirical SQL query on dataset",
        subtasks=[
            PlannedSubtask(
                id="st-res-1",
                description="Review literature on turbofan degradation indicators",
                role=LLMAgentRole.RESEARCH,
                tools=["rag_search"],
                depends_on=[],
            ),
            PlannedSubtask(
                id="st-da-1",
                description="Query maximum time cycles and degradation metrics for engine 1",
                role=LLMAgentRole.DATA_ANALYST,
                tools=["sql_query"],
                depends_on=["st-res-1"],
            ),
        ],
    )

    # Research specialist: literature synthesis
    res_text = fake_text("Reviewed literature: C-MAPSS turbofan degradation exhibits progressive sensor drift.")
    res_output = SpecialistOutput(
        summary="Turbofan engines exhibit observable sensor drift prior to run-to-failure.",
        findings=[],
        limitations=["Empirical run-to-failure cycles depend on operational regime."],
    )

    # Data analyst specialist: executes real SQL query on CSV
    call_id = "call-sql-e1"
    da_tool_call = fake_tool_call(
        name="sql_query",
        args={
            "query": "SELECT MAX(time_cycles) AS max_cycles, MIN(time_cycles) AS min_cycles, COUNT(*) AS total_records FROM dataset WHERE unit_number = 1;"
        },
        tool_call_id=call_id,
    )
    da_text = fake_text("Engine 1 ran for a maximum of 192 time cycles across 192 recorded operational intervals.")
    da_output = SpecialistOutput(
        summary="Engine 1 degradation analysis completed.",
        findings=[
            Finding(
                statement="Engine 1 observed a maximum of 192 time cycles in the dataset.",
                evidence=[EvidenceRef(kind="tool_call", ref_id=call_id)],
            )
        ],
        limitations=["Sensor degradation patterns analyzed for unit 1 only."],
    )

    # Verification judge verdict
    judge_verdict = JudgeVerdict(
        verdict="supported",
        quote_span="192",
        rationale="SQL query execution results confirm maximum time_cycles for unit 1 is 192.",
    )

    # Critic evaluation
    critic_output = CriticRubricOutput(
        answers_question=9.0,
        uses_task_data=9.0,
        claims_verified=9.0,
        limitations_stated=9.0,
    )

    # Synthesizer final output
    if include_192_in_report:
        synth_output = SynthesizerOutput(
            title="C-MAPSS Engine 1 Degradation Assessment",
            final_finding="Engine 1 completed a total of 192 operational time cycles prior to maintenance retirement.",
            key_conclusions=[
                "Maximum observed lifespan for unit 1 was 192 time cycles.",
                "Degradation trajectory accelerates in later operational cycles.",
            ],
            actionable_recommendations=[
                "Schedule inspection intervals prior to cycle 180.",
                "Monitor high-pressure compressor discharge temperature closely.",
            ],
            evidence_summary="Empirical queries confirm 192 time cycles for unit 1.",
            data_analysis_summary="SQL query against C-MAPSS dataset verified cycle lifespan.",
        )
    else:
        synth_output = SynthesizerOutput(
            title="C-MAPSS Engine 1 Degradation Assessment",
            final_finding="Engine 1 completed its operational time cycles prior to maintenance retirement.",
            key_conclusions=[
                "Observed lifespan for unit 1 was evaluated.",
                "Degradation trajectory accelerates in later operational cycles.",
            ],
            actionable_recommendations=[
                "Schedule inspection intervals prior to retirement.",
                "Monitor high-pressure compressor discharge temperature closely.",
            ],
            evidence_summary="Empirical queries confirm lifecycle for unit 1.",
            data_analysis_summary="SQL query against C-MAPSS dataset verified cycle lifespan.",
        )

    return FakeLLM(
        responses=[
            plan,
            res_text,
            res_output,
            da_tool_call,
            da_text,
            da_output,
            judge_verdict,
            critic_output,
            synth_output,
        ]
    )


def test_offline_workflow_cmapss():
    """Offline test running LangGraph with FakeLLM and real SQL tool calls against the sample CSV."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "fd001_train_5engines.csv"
    assert csv_path.is_file(), f"C-MAPSS sample file not found at {csv_path}"

    df = pd.read_csv(csv_path)
    csv_text = csv_path.read_text(encoding="utf-8")

    task_context = TaskContext(dataset=df, task_id="offline-cmapss-run")
    try:
        tool_reg = ToolRegistry.from_context(task_context)
        fake_llm = _build_cmapss_fake_llm(include_192_in_report=True)

        orchestrator = AegisOrchestrator(
            llm_provider=fake_llm,
            tool_registry=tool_reg,
            task_context=task_context,
        )

        query = (
            "What is the maximum number of time_cycles observed for unit_number 1 (engine 1) in the dataset, "
            "and what are its operational degradation characteristics?"
        )
        initial_state: AegisState = {
            "task_id": "offline-cmapss-run",
            "query": query,
            "has_dataset": True,
            "dataset_name": "fd001_train_5engines.csv",
            "dataset_content": csv_text,
            "require_human_approval": False,
            "max_retries": 1,
            "retry_count": 0,
        }

        result = orchestrator.run(initial_state)
        if result.get("status") == TaskStatus.WAITING_APPROVAL:
            result = orchestrator.resume("offline-cmapss-run", {"action": "approve"})

        # Validate against the live workflow assertions function
        assert_live_workflow_result(result, df)
        _assert_prompt_contract(fake_llm, csv_text, sql_call_id="call-sql-e1")
    finally:
        task_context.cleanup()


def _prompt_text(call: dict) -> str:
    return json.dumps([m.model_dump() for m in call["messages"]], default=str) + (call.get("system") or "")


def _assert_prompt_contract(fake_llm: FakeLLM, csv_text: str, sql_call_id: str) -> None:
    """What the real model is shown. A scripted FakeLLM cites the right IDs whatever the prompt
    says, so these two live-run failures were invisible until checked here."""
    # 1. The raw CSV never reaches a prompt (it cost ~120K input tokens per call live).
    data_row = csv_text.splitlines()[1]
    for call in fake_llm.calls:
        text = _prompt_text(call)
        assert data_row not in text, f"Raw dataset rows were sent in a {call['method']} prompt"
        assert len(text) < 20_000, f"{call['method']} prompt is {len(text)} chars"

    # 2. The model is shown the exact evidence ID it must cite: in the tool result and in the
    #    final SpecialistOutput instruction. Live, it guessed "sql_query_1" and every claim was dropped.
    analyst_final = [
        c for c in fake_llm.calls
        if c["method"] == "structured" and c["schema"] is SpecialistOutput and sql_call_id in _prompt_text(c)
    ]
    assert analyst_final, "The analyst's final structured call never showed the model its tool call ID"
    text = _prompt_text(analyst_final[0])
    assert f"[evidence_id: {sql_call_id}]" in text
    assert f"tool_call {sql_call_id} (sql_query)" in text


def test_offline_workflow_fails_when_report_omits_192():
    """The assertion function must fail if the final report omits 192."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "fd001_train_5engines.csv"
    assert csv_path.is_file(), f"C-MAPSS sample file not found at {csv_path}"

    df = pd.read_csv(csv_path)
    csv_text = csv_path.read_text(encoding="utf-8")

    task_context = TaskContext(dataset=df, task_id="offline-cmapss-fail-run")
    try:
        tool_reg = ToolRegistry.from_context(task_context)
        fake_llm = _build_cmapss_fake_llm(include_192_in_report=False)

        orchestrator = AegisOrchestrator(
            llm_provider=fake_llm,
            tool_registry=tool_reg,
            task_context=task_context,
        )

        query = "What is the maximum number of time_cycles observed for unit_number 1 (engine 1)?"
        initial_state: AegisState = {
            "task_id": "offline-cmapss-fail-run",
            "query": query,
            "has_dataset": True,
            "dataset_name": "fd001_train_5engines.csv",
            "dataset_content": csv_text,
            "require_human_approval": False,
            "max_retries": 1,
            "retry_count": 0,
        }

        result = orchestrator.run(initial_state)
        if result.get("status") == TaskStatus.WAITING_APPROVAL:
            result = orchestrator.resume("offline-cmapss-fail-run", {"action": "approve"})

        # Now simulate report where 192 is missing in report text
        # Even if verification claim had 192, if report synthesis leaves out 192 and claim is not counted:
        # Let's ensure combined_report_text omits 192:
        report = result.get("final_report")
        assert report is not None

        # When the report executive summary omits 192 and supported claims omit 192:
        report_copy = copy.deepcopy(report)
        report_copy.executive_summary.final_finding = "Engine 1 completed its operational time cycles."
        report_copy.executive_summary.key_conclusions = ["Lifespan evaluated."]
        report_copy.verification_section.claims = []
        result_omitted = dict(result)
        result_omitted["final_report"] = report_copy

        with pytest.raises(AssertionError) as exc_info:
            assert_live_workflow_result(result_omitted, df)

        assert "192" in str(exc_info.value)
    finally:
        task_context.cleanup()

