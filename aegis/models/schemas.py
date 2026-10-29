"""Data models and schemas for Aegis Orchestrator."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    execution_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class SubTask(BaseModel):
    id: str
    title: str
    description: str
    assigned_agent: AgentRole
    required_tools: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed, retrying
    output: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    retry_count: int = 0
    error: Optional[str] = None


class ExecutionPlan(BaseModel):
    task_id: str
    original_query: str = ""
    objective: str
    subtasks: List[SubTask]
    estimated_complexity: str = "medium"  # low, medium, high
    rationale: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentExecutionResult(BaseModel):
    subtask_id: str
    agent_role: AgentRole
    output_summary: str
    detailed_findings: List[str] = Field(default_factory=list)
    code_snippets: List[Dict[str, str]] = Field(default_factory=list)
    data_tables: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.85)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ClaimVerification(BaseModel):
    claim: str
    is_supported: bool
    confidence: float = Field(ge=0.0, le=1.0, default=0.9)
    evidence_sources: List[str] = Field(default_factory=list)
    contradiction_detected: bool = False
    notes: str = ""


class VerificationMetrics(BaseModel):
    evidence_coverage: float = Field(ge=0.0, le=100.0, default=85.0)
    source_quality: float = Field(ge=0.0, le=100.0, default=90.0)
    claim_consistency: float = Field(ge=0.0, le=100.0, default=92.0)
    temporal_validity: float = Field(ge=0.0, le=100.0, default=95.0)
    hallucination_risk: float = Field(ge=0.0, le=100.0, default=5.0)
    overall_reliability: float = Field(ge=0.0, le=100.0, default=89.0)


class VerificationResult(BaseModel):
    status: str = "passed"  # passed, needs_revision, failed
    claims: List[ClaimVerification] = Field(default_factory=list)
    metrics: VerificationMetrics = Field(default_factory=VerificationMetrics)
    unsupported_claims: List[str] = Field(default_factory=list)
    audit_notes: str = ""
    verification_time_seconds: float = 0.0


class CriticEvaluation(BaseModel):
    quality_pass: bool = True
    score: float = Field(ge=0.0, le=10.0, default=8.5)
    issues_found: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    weak_reasoning: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    critique_summary: str = ""
    retry_recommendations: Optional[str] = None
    target_subtask_for_retry: Optional[str] = None


class ExecutiveSummary(BaseModel):
    final_finding: str
    key_conclusions: List[str] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.88)
    business_impact: str = ""
    actionable_recommendations: List[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    task_id: str
    agent_role: Optional[AgentRole] = None
    event_type: str  # task_start, agent_start, tool_call, agent_finish, verify, critic, retry, human_gate, complete, error
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    token_usage: int = 0
    estimated_cost_usd: float = 0.0


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
    total_cost_usd: float = 0.0
    retries_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
