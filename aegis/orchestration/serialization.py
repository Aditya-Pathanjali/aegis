"""Serialization configuration and allowed types for LangGraph checkpointers."""

from __future__ import annotations

from typing import Any, List

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aegis.models.llm_outputs import (
    CriticRubricOutput,
    EvidenceRef,
    Finding,
    JudgeVerdict,
    PlannedSubtask,
    PlanOutput,
    SpecialistOutput,
    SynthesizerOutput,
)
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    ApprovalAction,
    ClaimVerification,
    CriticEvaluation,
    ExecutionPlan,
    ExecutiveSummary,
    FinalReport,
    HumanApprovalRequest,
    SubTask,
    TaskResponse,
    TaskStatus,
    TaskSubmission,
    ToolCallRecord,
    TraceEvent,
    VerificationMetrics,
    VerificationResult,
)

AEGIS_MSGPACK_MODELS: List[Any] = [
    # Schemas
    ExecutionPlan,
    SubTask,
    AgentExecutionResult,
    ClaimVerification,
    VerificationMetrics,
    VerificationResult,
    CriticEvaluation,
    ExecutiveSummary,
    FinalReport,
    HumanApprovalRequest,
    TraceEvent,
    TaskSubmission,
    TaskResponse,
    ToolCallRecord,
    TaskStatus,
    AgentRole,
    ApprovalAction,
    # LLM outputs
    PlannedSubtask,
    PlanOutput,
    EvidenceRef,
    Finding,
    SpecialistOutput,
    JudgeVerdict,
    CriticRubricOutput,
    SynthesizerOutput,
]


def get_checkpointer_serializer() -> JsonPlusSerializer:
    """Returns a JsonPlusSerializer with Aegis schemas registered in allowed_msgpack_modules.

    This ensures that serialization and deserialization across checkpoints (e.g., PostgresSaver)
    operate cleanly without triggering 'Deserializing unregistered type' warnings.
    """
    return JsonPlusSerializer(allowed_msgpack_modules=AEGIS_MSGPACK_MODELS)
