"""Orchestrator Service managing workflow lifecycle, background execution, and approval resolution."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional

from aegis.llm import LLMProvider, get_llm_provider
from aegis.memory.state_store import TaskStateStore, get_state_store
from aegis.memory.trace_logger import TraceLogger, get_trace_logger
from aegis.models.schemas import (
    ApprovalAction,
    HumanApprovalRequest,
    TaskResponse,
    TaskStatus,
    TaskSubmission,
    TraceEvent,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.tools.registry import get_tool_registry


class OrchestratorService:
    """Coordinates task execution, trace streaming, and human approvals."""

    def __init__(
        self,
        state_store: Optional[TaskStateStore] = None,
        trace_logger: Optional[TraceLogger] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.state_store = state_store or get_state_store()
        self.trace_logger = trace_logger or get_trace_logger()
        self.llm = llm_provider
        self._subscribers: Dict[str, List[Callable[[TraceEvent], None]]] = {}

    def subscribe(self, task_id: str, callback: Callable[[TraceEvent], None]) -> None:
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(callback)

    def _broadcast_event(self, event: TraceEvent) -> None:
        self.trace_logger.log(event)
        subscribers = self._subscribers.get(event.task_id, [])
        for cb in subscribers:
            try:
                cb(event)
            except Exception:
                pass

    def submit_task(self, submission: TaskSubmission) -> str:
        task_id = f"task-{uuid.uuid4().hex[:8]}"

        # Initialize State
        has_data = bool(submission.dataset_content or submission.dataset_name)
        initial_state: AegisState = {
            "task_id": task_id,
            "query": submission.query,
            "has_dataset": has_data,
            "dataset_name": submission.dataset_name,
            "dataset_content": submission.dataset_content,
            "require_human_approval": submission.require_human_approval,
            "max_retries": submission.max_retries,
            "retry_count": 0,
            "current_subtask_index": 0,
            "specialist_results": [],
            "trace_events": [],
            "status": TaskStatus.PLANNING,
            "errors": [],
        }

        self.state_store.save_state(task_id, initial_state)

        # Launch execution in background thread
        thread = threading.Thread(
            target=self._run_workflow_worker,
            args=(task_id, initial_state, submission),
            daemon=True,
        )
        thread.start()

        return task_id

    def _run_workflow_worker(
        self,
        task_id: str,
        initial_state: AegisState,
        submission: TaskSubmission,
    ) -> None:
        try:
            llm = self.llm or get_llm_provider()

            orchestrator = AegisOrchestrator(
                llm_provider=llm,
                tool_registry=get_tool_registry(),
                trace_callback=self._broadcast_event,
            )

            final_state = orchestrator.run(initial_state)
            self.state_store.save_state(task_id, final_state)

        except Exception as e:
            state = self.state_store.get_state(task_id) or initial_state
            state["status"] = TaskStatus.FAILED
            state["errors"].append(str(e))
            self._broadcast_event(
                TraceEvent(
                    event_id=str(uuid.uuid4())[:8],
                    task_id=task_id,
                    event_type="error",
                    message=f"Workflow execution halted with error: {str(e)}",
                    details={"error": str(e)},
                )
            )
            self.state_store.save_state(task_id, state)

    def resolve_human_approval(
        self,
        task_id: str,
        action: ApprovalAction,
        feedback: Optional[str] = None,
    ) -> TaskResponse:
        state = self.state_store.get_state(task_id)
        if not state:
            raise KeyError(f"Task '{task_id}' not found.")

        req: Optional[HumanApprovalRequest] = state.get("human_approval")
        if not req:
            req = HumanApprovalRequest(task_id=task_id)

        req.decided_at = datetime.utcnow()
        req.feedback = feedback

        if action == ApprovalAction.APPROVE:
            req.status = "approved"
            state["status"] = TaskStatus.COMPLETED
            msg = "Operator APPROVED the AI recommendation."
        elif action == ApprovalAction.REJECT:
            req.status = "rejected"
            state["status"] = TaskStatus.REJECTED
            msg = f"Operator REJECTED recommendation. Reason: {feedback or 'No comment'}"
        elif action == ApprovalAction.REQUEST_REANALYSIS:
            req.status = "revision_requested"
            state["status"] = TaskStatus.RETRYING
            msg = f"Operator requested RE-ANALYSIS with feedback: {feedback}"
            # Re-run orchestrator with re-analysis in background
            state["critic_evaluation"] = None
            state["current_subtask_index"] = 0
            state["retry_count"] = state.get("retry_count", 0) + 1
            thread = threading.Thread(
                target=self._run_workflow_worker,
                args=(
                    task_id,
                    state,
                    TaskSubmission(query=state["query"], require_human_approval=False),
                ),
                daemon=True,
            )
            thread.start()

        state["human_approval"] = req
        self._broadcast_event(
            TraceEvent(
                event_id=str(uuid.uuid4())[:8],
                task_id=task_id,
                event_type="human_gate",
                message=msg,
                details={"action": action.value, "feedback": feedback},
            )
        )
        self.state_store.save_state(task_id, state)
        return self.state_store.get_response(task_id)


_global_orchestrator_service: Optional[OrchestratorService] = None


def get_orchestrator_service() -> OrchestratorService:
    global _global_orchestrator_service
    if _global_orchestrator_service is None:
        _global_orchestrator_service = OrchestratorService()
    return _global_orchestrator_service
