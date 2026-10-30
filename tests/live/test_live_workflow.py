"""Live integration test against real dataset and live LLM provider."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from aegis.agents.verification import extract_numbers, number_in_evidence
from aegis.config import Settings
from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.models.schemas import AgentRole, TaskStatus
from aegis.orchestration.graph import AegisOrchestrator
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY environment variable is not set",
)
def test_live_workflow_cmapss():
    csv_path = Path(__file__).resolve().parents[2] / "data" / "samples" / "fd001_train_5engines.csv"
    assert csv_path.is_file(), f"C-MAPSS sample file not found at {csv_path}"

    df = pd.read_csv(csv_path)
    csv_text = csv_path.read_text(encoding="utf-8")

    api_key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("AEGIS_MODEL", "claude-sonnet-5-5")
    settings = Settings(provider="anthropic", model=model)
    provider = AnthropicProvider(model=model, api_key=api_key, settings=settings)

    task_context = TaskContext(dataset=df, task_id="live-cmapss-run")
    tool_reg = ToolRegistry.from_context(task_context)

    orchestrator = AegisOrchestrator(
        llm_provider=provider,
        tool_registry=tool_reg,
        task_context=task_context,
    )

    query = "What is the maximum number of time_cycles observed for unit_number 1 (engine 1) in the dataset, and what are its operational degradation characteristics?"
    initial_state = {
        "task_id": "live-cmapss-run",
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
        result = orchestrator.resume("live-cmapss-run", {"action": "approve"})

    # 1. Plan has >= 2 subtasks, including a research task and a data-analyst task
    plan = result.get("plan")
    assert plan is not None, "Execution plan was not generated"
    assert len(plan.subtasks) >= 2, f"Expected >= 2 subtasks, got {len(plan.subtasks)}"
    assigned_roles = [st.assigned_agent for st in plan.subtasks]
    assert AgentRole.RESEARCH in assigned_roles, "Plan missing research subtask"
    assert AgentRole.DATA_ANALYST in assigned_roles, "Plan missing data-analyst subtask"

    # 2. Data analyst runs at least one SQL query or Python script
    specialist_results = result.get("specialist_results", {})
    analyst_results = [r for r in specialist_results.values() if r.agent_role == AgentRole.DATA_ANALYST]
    assert len(analyst_results) > 0, "No data analyst subtask was executed"
    all_analyst_tool_calls = [tc.tool_name for r in analyst_results for tc in r.tool_calls]
    assert any(
        t in ("sql_query", "python_exec") for t in all_analyst_tool_calls
    ), f"Data analyst did not execute sql_query or python_exec: {all_analyst_tool_calls}"

    # 3. Verification audits all claims in the specialist findings
    v_res = result.get("verification_result")
    assert v_res is not None, "Verification result is missing"
    all_findings = [f for r in specialist_results.values() for f in r.findings if f.statement.strip()]
    assert len(v_res.claims) == len(all_findings), (
        f"Verification audited {len(v_res.claims)} claims, expected {len(all_findings)}"
    )

    # 4. Synthesizer produces a report with non-empty recommendations and limitations
    report = result.get("final_report")
    assert report is not None, "Final report is missing"
    assert len(report.executive_summary.actionable_recommendations) > 0, "Report recommendations are empty"
    assert len(report.limitations) > 0, "Report limitations are empty"

    # 5. Domain-specific check: engine 1 max time_cycles is 192 in the dataset
    e1_df = df[df["unit_number"] == 1]
    e1_max_cycles = int(e1_df["time_cycles"].max())
    assert e1_max_cycles == 192

    all_report_texts = (
        [report.title, report.final_finding, report.executive_summary.data_analysis_summary]
        + list(report.executive_summary.key_conclusions)
        + [f.statement for f in report.verified_findings]
    )
    combined_report_text = " ".join(all_report_texts)
    assert "192" in combined_report_text, (
        f"Expected computed answer 192 (max time_cycles for engine 1) in final report text, got: {combined_report_text}"
    )

    # Assert that findings citing 192 cite the tool output that produced it
    findings_with_192 = [f for f in report.verified_findings if "192" in f.statement]
    if not findings_with_192:
        all_spec_findings = [f for r in specialist_results.values() for f in r.findings if "192" in f.statement]
        assert len(all_spec_findings) > 0, "No specialist finding contains 192"
        findings_with_192 = all_spec_findings

    for f in findings_with_192:
        assert len(f.evidence) > 0, f"Finding '{f.statement}' does not cite evidence"
        tool_call_ids = [ref.ref_id for ref in f.evidence if ref.kind == "tool_call"]
        assert len(tool_call_ids) > 0, f"Finding '{f.statement}' did not cite any tool_call"
        all_tool_records = [tc for r in specialist_results.values() for tc in r.tool_calls]
        cited_records = [tc for tc in all_tool_records if tc.call_id in tool_call_ids]
        assert len(cited_records) > 0, f"No tool call records found for cited IDs {tool_call_ids}"
        assert any("192" in str(tc.output) for tc in cited_records), (
            f"Cited tool output does not contain 192: {[tc.output for tc in cited_records]}"
        )

    # 6. Asserts every number in supported claims matches computed dataset statistics
    e1_min_cycles = float(e1_df["time_cycles"].min())
    total_rows = float(len(e1_df))

    dataset_nums = [float(e1_max_cycles), e1_min_cycles, total_rows, 1.0]
    for col in e1_df.columns:
        if col not in ("unit_number", "time_cycles"):
            dataset_nums.append(round(float(e1_df[col].mean()), 2))
            dataset_nums.append(round(float(e1_df[col].max()), 2))
            dataset_nums.append(round(float(e1_df[col].min()), 2))

    for claim in v_res.claims:
        if claim.is_supported:
            claim_nums = extract_numbers(claim.claim)
            for num in claim_nums:
                assert number_in_evidence(num, dataset_nums) or any(
                    abs(num - val) <= max(0.05, 0.05 * abs(val)) for val in dataset_nums
                ), f"Number {num} in supported claim '{claim.claim}' does not match computed dataset statistics"

