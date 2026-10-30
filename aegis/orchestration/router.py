"""Dynamic routing logic and condition evaluators for Aegis LangGraph workflow."""

from __future__ import annotations

from aegis.orchestration.state import AegisState


def route_next_subtask(state: AegisState) -> str:
    """
    Decides whether there are remaining specialist subtasks to execute
    or if all subtasks are complete and ready for verification.
    """
    pending = state.get("pending_retry_subtasks")
    if pending is not None:
        if len(pending) > 0:
            return "specialist_node"
        return "verification_node"

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
    If critic failed and retry_count < max_retries:
    - routes to 'planner_node' if replan is requested
    - routes to 'retry_node' for targeted subtask retry
    Otherwise proceeds to synthesizer.
    """
    critic = state.get("critic_evaluation")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if critic and not critic.quality_pass:
        if retry_count < max_retries:
            if critic.replan:
                return "planner_node"
            return "retry_node"

    return "synthesizer_node"


def route_human_gate_or_complete(state: AegisState) -> str:
    """Evaluates whether human review is mandated."""
    if state.get("require_human_approval", False):
        return "human_gate_node"

    critic = state.get("critic_evaluation")
    if critic and not critic.quality_pass:
        return "human_gate_node"

    v_res = state.get("verification_result")
    if v_res and v_res.metrics:
        if v_res.metrics.contradicted > 0 or v_res.metrics.supported_ratio < 0.8:
            return "human_gate_node"

    return "__end__"


def route_after_human_gate(state: AegisState) -> str:
    """Routes back to planner if human requested reanalysis, else completes."""
    if state.get("status") == "retrying":
        return "planner_node"
    return "__end__"
