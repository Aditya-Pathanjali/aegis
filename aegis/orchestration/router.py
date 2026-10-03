"""Dynamic routing logic and condition evaluators for Aegis LangGraph workflow."""

from __future__ import annotations

from aegis.models.schemas import AgentRole, TaskStatus
from aegis.orchestration.state import AegisState


def route_next_subtask(state: AegisState) -> str:
    """
    Decides whether there are remaining specialist subtasks to execute
    or if all subtasks are complete and ready for verification.
    """
    plan = state.get("plan")
    if not plan or not plan.subtasks:
        return "verification_node"

    current_idx = state.get("current_subtask_index", 0)
    if current_idx < len(plan.subtasks):
        return "specialist_node"
    return "verification_node"


def evaluate_quality_pass(state: AegisState) -> str:
    """
    Evaluates whether the critic approved the work or if a retry is warranted.
    If critic failed and retry_count < max_retries, loop back to retry.
    Otherwise proceed to synthesizer.
    """
    critic = state.get("critic_evaluation")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if critic and not critic.quality_pass:
        if retry_count < max_retries:
            return "retry_node"

    return "synthesizer_node"


def route_human_gate_or_complete(state: AegisState) -> str:
    """
    Evaluates whether human review is mandated due to:
    1. Explicit user configuration (require_human_approval=True)
    2. Low confidence (< 0.70)
    3. Critical / high risk operational actions
    """
    if state.get("require_human_approval", False):
        return "human_gate_node"

    report = state.get("final_report")
    if report:
        conf = report.executive_summary.overall_confidence
        if conf < 0.75:
            return "human_gate_node"

    return "__end__"
