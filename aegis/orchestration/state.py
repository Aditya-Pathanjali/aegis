"""LangGraph State definition for Aegis Orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from typing_extensions import TypedDict

from aegis.models.schemas import (
    AgentExecutionResult,
    CriticEvaluation,
    ExecutionPlan,
    FinalReport,
    HumanApprovalRequest,
    TaskStatus,
    TraceEvent,
    VerificationResult,
)


class AegisState(TypedDict, total=False):
    task_id: str
    query: str
    has_dataset: bool
    dataset_name: Optional[str]
    dataset_content: Optional[str]
    require_human_approval: bool
    max_retries: int
    retry_count: int
    current_subtask_index: int
    plan: Optional[ExecutionPlan]
    specialist_results: Union[Dict[str, AgentExecutionResult], List[AgentExecutionResult]]
    pending_retry_subtasks: Optional[List[str]]
    critic_guidance: Optional[str]
    verification_result: Optional[VerificationResult]
    critic_evaluation: Optional[CriticEvaluation]
    human_approval: Optional[HumanApprovalRequest]
    final_report: Optional[FinalReport]
    trace_events: List[TraceEvent]
    status: TaskStatus
    errors: List[str]
