"""Live integration test against real dataset and live LLM provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aegis.agents.verification import extract_numbers, number_in_evidence
from aegis.config import Settings
from aegis.llm.anthropic_provider import AnthropicProvider
from aegis.models.schemas import (
    AgentRole,
    ExecutionPlan,
    FinalReport,
    TaskStatus,
    VerificationResult,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry


def _serialize_for_json(obj: Any) -> Any:
    """Helper to recursively serialize objects to JSON-compatible data structures."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {str(k): _serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _write_last_run(path: Path, result: dict[str, Any] | AegisState) -> None:
    """Serializes the full orchestrator execution state to disk before running assertions."""
    data = _serialize_for_json(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def print_token_totals(result: dict[str, Any] | AegisState) -> None:
    """Computes and prints token usage aggregated per agent role from trace events."""
    agent_tokens: dict[str, dict[str, int]] = {}
    for event in result.get("trace_events", []):
        role_str = event.agent_role.value if hasattr(event, "agent_role") and event.agent_role else "system"
        if role_str not in agent_tokens:
            agent_tokens[role_str] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        in_tok = getattr(event, "input_tokens", 0) or 0
        out_tok = getattr(event, "output_tokens", 0) or 0
        tot_tok = getattr(event, "total_tokens", 0) or getattr(event, "token_usage", 0) or (in_tok + out_tok)
        agent_tokens[role_str]["input_tokens"] += in_tok
        agent_tokens[role_str]["output_tokens"] += out_tok
        agent_tokens[role_str]["total_tokens"] += tot_tok

    print("\n================== Token Totals Per Agent ==================")
    total_all = 0
    for role, tok in sorted(agent_tokens.items()):
        total_all += tok["total_tokens"]
        print(
            f"  {role:15s} -> Input: {tok['input_tokens']:6d} | Output: {tok['output_tokens']:6d} | Total: {tok['total_tokens']:6d}"
        )
    print(f"  {'TOTAL':15s} -> {total_all:6d} tokens")
    print("============================================================\n")


def assert_live_workflow_result(result: dict[str, Any] | AegisState, df: pd.DataFrame) -> None:
    """Validates the execution results of the C-MAPSS workflow against expected engineering invariants."""
    # 1. Plan has >= 2 subtasks, including a research task and a data-analyst task
    plan = result.get("plan")
    assert plan is not None, "Execution plan was not generated"
    assert isinstance(plan, ExecutionPlan), f"Expected ExecutionPlan, got {type(plan)}"
    assert len(plan.subtasks) >= 2, f"Expected >= 2 subtasks, got {len(plan.subtasks)}"
    assigned_roles = [st.assigned_agent for st in plan.subtasks]
    assert AgentRole.RESEARCH in assigned_roles, "Plan missing research subtask"
    assert AgentRole.DATA_ANALYST in assigned_roles, "Plan missing data-analyst subtask"

    # 2. Data analyst runs at least one SQL query or Python script
    specialist_results = result.get("specialist_results", {})
    if isinstance(specialist_results, dict):
        spec_list = list(specialist_results.values())
    else:
        spec_list = list(specialist_results or [])

    analyst_results = [r for r in spec_list if r.agent_role == AgentRole.DATA_ANALYST]
    assert len(analyst_results) > 0, "No data analyst subtask was executed"
    all_analyst_tool_calls = [tc.tool_name for r in analyst_results for tc in r.tool_calls]
    assert any(
        t in ("sql_query", "python_exec") for t in all_analyst_tool_calls
    ), f"Data analyst did not execute sql_query or python_exec: {all_analyst_tool_calls}"

    # 3. Verification audits all claims in the specialist findings
    v_res = result.get("verification_result")
    assert v_res is not None, "Verification result is missing"
    assert isinstance(v_res, VerificationResult), f"Expected VerificationResult, got {type(v_res)}"
    all_findings = [f for r in spec_list for f in r.findings if f.statement.strip()]
    assert len(v_res.claims) == len(all_findings), (
        f"Verification audited {len(v_res.claims)} claims, expected {len(all_findings)}"
    )

    # 4. Synthesizer produces a report with non-empty recommendations and limitations
    report = result.get("final_report")
    assert report is not None, "Final report is missing"
    assert isinstance(report, FinalReport), f"Expected FinalReport, got {type(report)}"
    assert len(report.executive_summary.actionable_recommendations) > 0, "Report recommendations are empty"
    assert len(report.limitations) > 0, "Report limitations are empty"

    # 5. Domain-specific check: engine 1 max time_cycles is 192 in the dataset
    e1_df = df[df["unit_number"] == 1]
    e1_max_cycles = int(e1_df["time_cycles"].max())
    assert e1_max_cycles == 192

    supported_claims = [c.claim for c in report.verification_section.claims if c.is_supported]
    all_report_texts = (
        [report.title, report.executive_summary.final_finding]
        + list(report.executive_summary.key_conclusions)
        + supported_claims
    )
    combined_report_text = " ".join(all_report_texts)
    assert "192" in combined_report_text, (
        f"Expected computed answer 192 (max time_cycles for engine 1) in final report text, got: {combined_report_text}"
    )

    # Find the "192" evidence through verification_section.claims or specialist findings
    all_tool_records = [tc for r in spec_list for tc in r.tool_calls]
    claims_with_192 = [c for c in report.verification_section.claims if "192" in c.claim and c.is_supported]
    findings_with_192 = [f for r in spec_list for f in r.findings if "192" in f.statement]
    assert len(claims_with_192) > 0 or len(findings_with_192) > 0, (
        "No supported claim or specialist finding contains 192"
    )

    cited_tool_ids: list[str] = []
    for c in claims_with_192:
        cited_tool_ids.extend(c.evidence_sources)
    for f in findings_with_192:
        cited_tool_ids.extend(ref.ref_id for ref in f.evidence if ref.kind in ("tool_call", "tool_output"))

    assert len(cited_tool_ids) > 0, "No tool call IDs cited for 192 findings/claims"
    cited_records = [tc for tc in all_tool_records if tc.call_id in cited_tool_ids]
    assert len(cited_records) > 0, f"No tool call records found matching cited IDs {cited_tool_ids}"
    assert any("192" in str(tc.output) for tc in cited_records), (
        f"Cited tool output does not contain 192: {[tc.output for tc in cited_records]}"
    )

    # 6. A number in a supported claim must appear in the outputs of the tool calls that claim cites
    for claim in v_res.claims:
        if claim.is_supported:
            claim_nums = extract_numbers(claim.claim)
            if not claim_nums:
                continue

            claim_cited_ids = set(claim.evidence_sources)
            for f in all_findings:
                if f.statement.strip() == claim.claim.strip():
                    for ref in f.evidence:
                        if ref.kind in ("tool_call", "tool_output"):
                            claim_cited_ids.add(ref.ref_id)

            matching_tool_calls = [tc for tc in all_tool_records if tc.call_id in claim_cited_ids]
            assert len(matching_tool_calls) > 0, (
                f"Supported claim '{claim.claim}' has no matching cited tool call records ({claim_cited_ids})"
            )

            tool_outputs_text = " ".join(
                json.dumps(tc.output) if isinstance(tc.output, (dict, list)) else str(tc.output)
                for tc in matching_tool_calls
            )
            evidence_numbers = extract_numbers(tool_outputs_text)

            for num in claim_nums:
                assert number_in_evidence(num, evidence_numbers), (
                    f"Number {num} in supported claim '{claim.claim}' does not appear in cited tool outputs: {tool_outputs_text}"
                )


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
    initial_state: AegisState = {
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

    # Always write the full result to tests/live/_last_run.json before assertions
    last_run_path = Path(__file__).resolve().parent / "_last_run.json"
    _write_last_run(last_run_path, result)

    # Print the token totals per agent at the end
    print_token_totals(result)

    # Validate against expected invariants
    assert_live_workflow_result(result, df)
