"""CLI Demonstration Script for Aegis Phase 1 Autonomous Multi-Agent System.

Runs two live queries against real LLMs:
1. Literature Query: ISO 10816 vibration severity thresholds.
2. Dataset Query: C-MAPSS engine 1 degradation and sensor trends.

Prints full execution traces, plan decomposition, verification audits, critic rubric scores,
human review gates (if triggered), final reports, token usage, cost, and wall-clock time.
Writes unedited outputs to docs/phase1_demo.md as proof of real input-dependent behavior.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from aegis.config import get_settings
from aegis.llm import get_llm_provider
from aegis.models.schemas import TaskStatus
from aegis.orchestration.graph import AegisOrchestrator
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_single_query(
    query: str,
    dataset_path: Path | None = None,
    task_id: str = "demo-task",
) -> Dict[str, Any]:
    print(f"\n{'='*80}\n[Aegis Demo] Executing Query: {query}\n{'='*80}")
    t0 = time.perf_counter()

    settings = get_settings()
    if not settings.provider or not settings.model:
        raise ValueError(
            "Missing AEGIS_PROVIDER or AEGIS_MODEL. Configure via environment variables or .env file."
        )

    if settings.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for anthropic provider.")
    elif settings.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is required for openai provider.")

    llm = get_llm_provider()

    df = None
    csv_text = None
    if dataset_path and dataset_path.is_file():
        df = pd.read_csv(dataset_path)
        csv_text = dataset_path.read_text(encoding="utf-8")
        print(f"Loaded dataset '{dataset_path.name}' with shape: {df.shape}")

    task_context = TaskContext(dataset=df, task_id=task_id)
    tool_reg = ToolRegistry.from_context(task_context)

    traces = []

    def on_trace(ev):
        traces.append(ev)
        print(f"[{ev.agent_role.value if ev.agent_role else 'SYSTEM':^12}] {ev.event_type:<14} | {ev.message}")

    orchestrator = AegisOrchestrator(
        llm_provider=llm,
        tool_registry=tool_reg,
        trace_callback=on_trace,
        task_context=task_context,
    )

    initial_state = {
        "task_id": task_id,
        "query": query,
        "has_dataset": bool(df is not None),
        "dataset_name": dataset_path.name if dataset_path else None,
        "dataset_content": csv_text,
        "require_human_approval": False,
        "max_retries": 1,
        "retry_count": 0,
    }

    state = orchestrator.run(initial_state)

    human_gate_prompt = None
    if state.get("status") == TaskStatus.WAITING_APPROVAL:
        approval_req = state.get("human_approval")
        human_gate_prompt = approval_req.model_dump() if approval_req else {}
        print(f"\n[HUMAN GATE TRIGGERED] Status: WAITING_APPROVAL. Reason: {approval_req.reason}")
        print(f"Proposed Actions: {approval_req.proposed_actions}")
        print("Auto-approving for demo execution...")
        state = orchestrator.resume(task_id, {"action": "approve", "feedback": "Approved by demo script."})

    wall_clock_time = round(time.perf_counter() - t0, 2)
    report = state.get("final_report")

    return {
        "query": query,
        "task_id": task_id,
        "wall_clock_seconds": wall_clock_time,
        "plan": state.get("plan"),
        "specialist_results": state.get("specialist_results"),
        "verification_result": state.get("verification_result"),
        "critic_evaluation": state.get("critic_evaluation"),
        "human_gate_prompt": human_gate_prompt,
        "final_report": report,
        "traces": traces,
        "total_tokens": report.total_tokens if report else 0,
        "total_cost_usd": report.total_cost_usd if report else None,
    }


def format_demo_markdown(results: list[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    buf.write("# Aegis Phase 1 Live Demonstration Execution Log\n\n")
    buf.write(
        "> **Notice:** This document contains live, unedited execution transcripts of the Aegis autonomous multi-agent\n"
        "> system running on real natural-language engineering queries against cloud LLMs.\n\n"
    )

    for i, res in enumerate(results, 1):
        buf.write(f"## Query {i}: \"{res['query']}\"\n\n")
        buf.write(f"- **Task ID:** `{res['task_id']}`\n")
        buf.write(f"- **Wall-Clock Duration:** `{res['wall_clock_seconds']} seconds`\n")
        buf.write(f"- **Total LLM Tokens:** `{res['total_tokens']}`\n")
        buf.write(f"- **Total Cost:** `${res['total_cost_usd'] if res['total_cost_usd'] is not None else 'Unpriced / Free'}`\n\n")

        # Plan
        buf.write("### 1. Generated Plan Decomposition\n\n")
        plan = res.get("plan")
        if plan:
            buf.write(f"**Objective:** {plan.objective}\n\n")
            for st in plan.subtasks:
                buf.write(f"- **Subtask `{st.id}`** ({st.assigned_agent.value}): {st.title} — *Tools:* `{st.required_tools}`\n")
                buf.write(f"  *Description:* {st.description}\n")
        buf.write("\n")

        # Verification
        buf.write("### 2. Empirical Verification Audit\n\n")
        v_res = res.get("verification_result")
        if v_res:
            m = v_res.metrics
            buf.write(f"- **Status:** `{v_res.status}`\n")
            buf.write(f"- **Audit Metrics:** {m.supported}/{len(v_res.claims)} supported ({m.supported_ratio*100:.1f}%), "
                      f"{m.contradicted} contradictions, {m.numeric_mismatches} numeric mismatches.\n\n")
            buf.write("#### Verified Claims:\n")
            for c in v_res.claims:
                buf.write(f"- [`{c.status.upper()}`] \"{c.claim}\" (Evidence: `{c.evidence_sources}`)\n")
        buf.write("\n")

        # Critic
        buf.write("### 3. Critic Peer Review Rubric\n\n")
        critic = res.get("critic_evaluation")
        if critic:
            buf.write(f"- **Quality Pass:** `{critic.quality_pass}` (Average Score: `{critic.score}/10`)\n")
            buf.write(f"- **Rubric Breakdown:** `{json.dumps(critic.rubric_scores)}`\n")
            buf.write(f"- **Critique Summary:** {critic.critique_summary}\n")
        buf.write("\n")

        # Human Gate
        if res.get("human_gate_prompt"):
            buf.write("### 4. Human Review Gate (Triggered)\n\n")
            buf.write("```json\n" + json.dumps(res["human_gate_prompt"], indent=2) + "\n```\n\n")

        # Report
        buf.write("### 5. Final Synthesized Engineering Report\n\n")
        rep = res.get("final_report")
        if rep:
            buf.write(f"#### {rep.title}\n\n")
            buf.write(f"**Final Finding:**\n> {rep.executive_summary.final_finding}\n\n")
            buf.write("**Key Conclusions:**\n")
            for kc in rep.executive_summary.key_conclusions:
                buf.write(f"- {kc}\n")
            buf.write("\n**Actionable Recommendations:**\n")
            for rec in rep.executive_summary.actionable_recommendations:
                buf.write(f"- {rec}\n")
            buf.write("\n**Transparent Limitations:**\n")
            for lim in rep.limitations:
                buf.write(f"- {lim}\n")
        buf.write("\n---\n\n")

    return buf.getvalue()


def main():
    print("Starting Aegis Live Demonstration Suite...")

    q1 = "What are standard predictive maintenance thresholds for ISO 10816 vibration severity?"
    cmapss_csv = REPO_ROOT / "data" / "samples" / "fd001_train_5engines.csv"
    q2 = (
        "Using the C-MAPSS sample, what is the maximum cycle count reached by engine 1, "
        "and what sensor shows the strongest trend before end-of-life?"
    )

    results = []

    try:
        r1 = run_single_query(query=q1, dataset_path=None, task_id="demo-q1-iso10816")
        results.append(r1)

        r2 = run_single_query(query=q2, dataset_path=cmapss_csv, task_id="demo-q2-cmapss")
        results.append(r2)

        docs_dir = REPO_ROOT / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        demo_md_path = docs_dir / "phase1_demo.md"
        content = format_demo_markdown(results)
        demo_md_path.write_text(content, encoding="utf-8")
        print(f"\n[Success] Demo output written to: {demo_md_path}")

    except Exception as e:
        print(f"\n[Demo Halt] Live execution halted: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
