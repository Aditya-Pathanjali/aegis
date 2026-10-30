"""Data models and schemas for Aegis Orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from aegis.llm.types import Usage


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCH = "research"
    DATA_ANALYST = "data_analyst"
    CODING = "coding"
    VERIFICATION = "verification"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    CRITIQUING = "critiquing"
    RETRYING = "retrying"
    SYNTHESIZING = "synthesizing"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ApprovalAction(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_REANALYSIS = "request_reanalysis"


class EvidenceRef(BaseModel):
    kind: Literal["tool_call", "chunk", "tool_output"]
    ref_id: str


class Finding(BaseModel):
    statement: str
    evidence: List[EvidenceRef] = Field(default_factory=list)


class SpecialistOutput(BaseModel):
    summary: str
    findings: List[Finding] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    call_id: Optional[str] = None
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    execution_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SubTask(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    description: str
    role: AgentRole = Field(default=AgentRole.RESEARCH, alias="assigned_agent")
    tools: List[str] = Field(default_factory=list, alias="required_tools")
    depends_on: List[str] = Field(default_factory=list, alias="dependencies")
    title: str = ""
    status: str = "pending"  # pending, in_progress, completed, failed, retrying
    output: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    retry_count: int = 0
    error: Optional[str] = None

    @property
    def assigned_agent(self) -> AgentRole:
        return self.role

    @property
    def required_tools(self) -> List[str]:
        return self.tools

    @property
    def dependencies(self) -> List[str]:
        return self.depends_on


Subtask = SubTask


class ExecutionPlan(BaseModel):
    subtasks: List[SubTask]
    task_id: str = ""
    original_query: str = ""
    objective: str = ""
    estimated_complexity: str = "medium"  # low, medium, high
    rationale: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentExecutionResult(BaseModel):
    subtask_id: str
    agent_role: AgentRole
    output_summary: str
    findings: List[Finding] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    detailed_findings: List[str] = Field(default_factory=list)
    code_snippets: List[Dict[str, str]] = Field(default_factory=list)
    data_tables: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    usage: Optional[Usage] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimVerification(BaseModel):
    claim: str
    status: Literal["supported", "contradicted", "unsupported", "no_evidence"]
    is_supported: bool = False
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evidence_sources: List[str] = Field(default_factory=list)
    quote_span: Optional[str] = None
    numeric_mismatch: bool = False
    contradiction_detected: bool = False
    notes: str = ""

    @model_validator(mode="after")
    def derive_is_supported(self) -> ClaimVerification:
        self.is_supported = (self.status == "supported")
        return self


class VerificationMetrics(BaseModel):
    supported: int = 0
    contradicted: int = 0
    unsupported: int = 0
    no_evidence: int = 0
    supported_ratio: Optional[float] = None
    citation_coverage: Optional[float] = None
    numeric_mismatches: int = 0
    quality_pass: bool = False
    evidence_coverage: float = 0.0
    source_quality: float = 0.0
    claim_consistency: float = 0.0
    temporal_validity: float = 0.0
    hallucination_risk: float = 0.0
    overall_reliability: float = 0.0


class VerificationResult(BaseModel):
    status: str = "passed"  # passed, needs_revision, failed
    claims: List[ClaimVerification] = Field(default_factory=list)
    metrics: VerificationMetrics = Field(default_factory=VerificationMetrics)
    unsupported_claims: List[str] = Field(default_factory=list)
    audit_notes: str = ""
    verification_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CriticEvaluation(BaseModel):
    quality_pass: bool = True
    score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    rubric_scores: Dict[str, float] = Field(default_factory=dict)
    retry_subtask_ids: List[str] = Field(default_factory=list)
    replan: bool = False
    guidance: str = ""
    issues_found: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    weak_reasoning: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    critique_summary: str = ""
    retry_recommendations: Optional[str] = None
    target_subtask_for_retry: Optional[str] = None


class SynthesizerOutput(BaseModel):
    title: str
    final_finding: str
    key_conclusions: List[str] = Field(default_factory=list)
    actionable_recommendations: List[str] = Field(default_factory=list)
    evidence_summary: str = ""
    data_analysis_summary: str = ""
    recommended_action: str = ""
    urgency: str = "medium"
    operational_steps: List[str] = Field(default_factory=list)


class ExecutiveSummary(BaseModel):
    final_finding: str
    key_conclusions: List[str] = Field(default_factory=list)
    overall_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    business_impact: str = ""
    actionable_recommendations: List[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str
    agent_role: Optional[AgentRole] = None
    event_type: str  # task_start, agent_start, tool_call, agent_finish, verify, critic, retry, human_gate, complete, error
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    token_usage: int = 0
    estimated_cost_usd: Optional[float] = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: Optional[str] = None


class FinalReport(BaseModel):
    task_id: str
    title: str
    executive_summary: ExecutiveSummary
    evidence_section: Dict[str, Any] = Field(default_factory=dict)
    data_analysis_section: Dict[str, Any] = Field(default_factory=dict)
    technical_recommendation: Dict[str, Any] = Field(default_factory=dict)
    verification_section: VerificationResult
    critic_evaluation: Optional[CriticEvaluation] = None
    agent_execution_trace: List[TraceEvent] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    total_execution_time_seconds: float = 0.0
    total_tokens: int = 0
    total_cost_usd: Optional[float] = 0.0
    retries_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HumanApprovalRequest(BaseModel):
    task_id: str
    required: bool = False
    status: str = "not_required"  # not_required, pending, approved, rejected, revision_requested
    reason: str = ""
    risk_level: str = "low"  # low, medium, high
    proposed_actions: List[str] = Field(default_factory=list)
    feedback: Optional[str] = None
    decided_at: Optional[datetime] = None


class TaskSubmission(BaseModel):
    query: str
    dataset_name: Optional[str] = None
    dataset_content: Optional[str] = None  # CSV or JSON raw string
    require_human_approval: bool = False
    max_retries: int = 2


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    plan: Optional[ExecutionPlan] = None
    report: Optional[FinalReport] = None
    approval_request: Optional[HumanApprovalRequest] = None
    trace_events: List[TraceEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
