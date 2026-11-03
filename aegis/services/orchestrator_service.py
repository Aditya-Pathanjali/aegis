"""Orchestrator service: persists task state and enqueues work. It never runs a graph.

All graph execution (initial runs, approvals, re-analysis) happens in arq workers
(aegis/worker.py), so the API process needs no LLM keys or Docker access to execute tasks.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Optional

from aegis.db.repository import PostgresTaskStore, get_task_store
from aegis.memory.trace_logger import TraceLogger, get_trace_logger
from aegis.models.schemas import (
    ApprovalAction,
    HumanApprovalRequest,
    TaskResponse,
    TaskStatus,
    TaskSubmission,
    TraceEvent,
)
from aegis.orchestration.state import AegisState

logger = logging.getLogger(__name__)


class QueueUnavailableError(RuntimeError):
    """The job queue could not accept work; the request must be refused (fail closed)."""


class OrchestratorService:
    """Coordinates task submission and human-approval decisions through the job queue."""

    def __init__(
        self,
        state_store: Optional[PostgresTaskStore] = None,
        trace_logger: Optional[TraceLogger] = None,
    ):
        self.task_store = state_store or get_task_store()
        self.state_store = self.task_store  # Backward-compatibility alias
        self.trace_logger = trace_logger or get_trace_logger()

    def _record_event(self, event: TraceEvent) -> None:
        from aegis.queue import publish_trace_event_sync

        self.trace_logger.log(event)
        self.task_store.save_trace_event(event)
        publish_trace_event_sync(event)

    def submit_task(self, submission: TaskSubmission) -> str:
        """Persists the initial state and enqueues the run. Raises QueueUnavailableError."""
        from aegis.queue import enqueue_task_job, run_sync

        task_id = f"task-{uuid.uuid4().hex[:8]}"
        initial_state: AegisState = {
            "task_id": task_id,
            "query": submission.query,
            "has_dataset": bool(submission.dataset_content or submission.dataset_name),
            "dataset_name": submission.dataset_name,
            "dataset_content": submission.dataset_content,
            "require_human_approval": submission.require_human_approval,
            "max_retries": submission.max_retries,
            "retry_count": 0,
            "current_subtask_index": 0,
            "specialist_results": [],
            "trace_events": [],
            "status": TaskStatus.QUEUED,
            "errors": [],
        }
        self.task_store.save_state(task_id, initial_state)

        try:
            run_sync(enqueue_task_job(task_id, submission))
        except Exception as exc:
            logger.exception("Failed to enqueue task %s: %s", task_id, exc)
            initial_state["status"] = TaskStatus.FAILED
            initial_state["errors"] = [f"Queue service unavailable: {exc}"]
            self.task_store.save_state(task_id, initial_state)
            raise QueueUnavailableError(f"Queue service unavailable: {exc}") from exc
        return task_id

    def resolve_human_approval(
        self,
        task_id: str,
        action: ApprovalAction,
        feedback: Optional[str] = None,
    ) -> TaskResponse:
        """Records the decision and enqueues the resume job; the worker resumes the checkpoint.

        Raises KeyError (unknown task), ValueError (not waiting for approval) or
        QueueUnavailableError (decision not queued; task left waiting so it can be retried).
        """
        from aegis.queue import enqueue_resume_job, run_sync

        state = self.task_store.get_state(task_id)
        if not state:
            raise KeyError(f"Task '{task_id}' not found.")
        if state.get("status") != TaskStatus.WAITING_APPROVAL:
            raise ValueError(f"Task '{task_id}' is not in WAITING_APPROVAL status (current: {state.get('status')}).")

        action_val = action.value if hasattr(action, "value") else str(action)
        try:
            run_sync(enqueue_resume_job(task_id, action_val, feedback))
        except Exception as exc:
            logger.exception("Failed to enqueue resume for task %s: %s", task_id, exc)
            raise QueueUnavailableError(f"Queue service unavailable: {exc}") from exc

        req: HumanApprovalRequest = state.get("human_approval") or HumanApprovalRequest(task_id=task_id)
        req.status = "decision_queued"
        req.feedback = feedback
        req.decided_at = datetime.now(UTC)
        state["human_approval"] = req
        state["status"] = TaskStatus.QUEUED
        self.task_store.save_state(task_id, state)

        self._record_event(
            TraceEvent(
                event_id=uuid.uuid4().hex[:8],
                task_id=task_id,
                event_type="human_gate",
                message=f"Operator decision '{action_val}' queued for the worker.",
                details={"action": action_val, "feedback": feedback},
            )
        )
        response = self.task_store.get_response(task_id)
        if response is None:
            raise KeyError(f"Task '{task_id}' response not found.")
        return response


_global_orchestrator_service: Optional[OrchestratorService] = None


def get_orchestrator_service() -> OrchestratorService:
    global _global_orchestrator_service
    if _global_orchestrator_service is None:
        _global_orchestrator_service = OrchestratorService()
    return _global_orchestrator_service


def reset_orchestrator_service() -> None:
    global _global_orchestrator_service
    _global_orchestrator_service = None
