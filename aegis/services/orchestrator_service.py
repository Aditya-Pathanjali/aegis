"""Orchestrator Service managing workflow lifecycle, background execution, and approval resolution."""

from __future__ import annotations

import io
import logging
import threading
import uuid
from datetime import UTC, datetime
from typing import Callable, Dict, List, Optional

import pandas as pd

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
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


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
        self._orchestrators: Dict[str, AegisOrchestrator] = {}

    def subscribe(self, task_id: str, callback: Callable[[TraceEvent], None]) -> None:
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(callback)

    def unsubscribe(self, task_id: str, callback: Callable[[TraceEvent], None]) -> None:
        if task_id in self._subscribers:
            if callback in self._subscribers[task_id]:
                self._subscribers[task_id].remove(callback)
            if not self._subscribers[task_id]:
                del self._subscribers[task_id]

    def _broadcast_event(self, event: TraceEvent) -> None:
        self.trace_logger.log(event)
        subscribers = self._subscribers.get(event.task_id, [])
        for cb in subscribers:
            try:
                cb(event)
            except Exception as exc:
                logger.exception("Error broadcasting event to subscriber: %s", exc)

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

        try:
            self.state_store.save_state(task_id, initial_state)
        except Exception as exc:
            logger.exception("Failed to save initial state for task %s: %s", task_id, exc)
            raise

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
        task_context: Optional[TaskContext] = None
        try:
            llm = self.llm or get_llm_provider()
            df = None
            if submission.dataset_content:
                try:
                    df = pd.read_csv(io.StringIO(submission.dataset_content))
                except Exception as e:
                    raise ValueError(f"Failed to parse dataset content as CSV: {e}") from e

            task_context = TaskContext(dataset=df, task_id=task_id)
            tool_reg = ToolRegistry.from_context(task_context)

            orchestrator = AegisOrchestrator(
                llm_provider=llm,
                tool_registry=tool_reg,
                trace_callback=self._broadcast_event,
                task_context=task_context,
            )
            self._orchestrators[task_id] = orchestrator

            def on_node(s):
                try:
                    self.state_store.save_state(task_id, s)
                except Exception as exc:
                    logger.exception("Failed to save state during on_node for task %s: %s", task_id, exc)
                    raise

            final_state = orchestrator.run(initial_state, on_node_finish=on_node)
            try:
                self.state_store.save_state(task_id, final_state)
            except Exception as exc:
                logger.exception("Failed to save final state for task %s: %s", task_id, exc)
                raise

            if final_state.get("status") in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED):
                self._orchestrators.pop(task_id, None)
                if task_context is not None:
                    task_context.cleanup()

        except Exception as e:
            logger.exception("Workflow worker execution failed for task %s: %s", task_id, e)
            try:
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
            except Exception as save_err:
                logger.exception("Failed to record error state for task %s: %s", task_id, save_err)
            self._orchestrators.pop(task_id, None)
            if task_context is not None:
                try:
                    task_context.cleanup()
                except Exception as clean_err:
                    logger.exception("Failed to clean task context for task %s: %s", task_id, clean_err)
        finally:
            st = self.state_store.get_state(task_id)
            if st and st.get("status") != TaskStatus.WAITING_APPROVAL:
                self._orchestrators.pop(task_id, None)
                if task_context is not None:
                    task_context.cleanup()

    def resolve_human_approval(
        self,
        task_id: str,
        action: ApprovalAction,
        feedback: Optional[str] = None,
    ) -> TaskResponse:
        state = self.state_store.get_state(task_id)
        if not state:
            raise KeyError(f"Task '{task_id}' not found.")

        if state.get("status") != TaskStatus.WAITING_APPROVAL:
            raise ValueError(f"Task '{task_id}' is not in WAITING_APPROVAL status.")

        orchestrator = self._orchestrators.get(task_id)
        action_val = action.value if hasattr(action, "value") else str(action)

        if action == ApprovalAction.REQUEST_REANALYSIS:
            req = state.get("human_approval") or HumanApprovalRequest(task_id=task_id)
            req.decided_at = datetime.now(UTC)
            req.feedback = feedback
            req.status = "revision_requested"
            state["human_approval"] = req
            state["status"] = TaskStatus.EXECUTING
            state["critic_evaluation"] = None
            state["current_subtask_index"] = 0
            state["retry_count"] = state.get("retry_count", 0) + 1
            try:
                self.state_store.save_state(task_id, state)
            except Exception as exc:
                logger.exception("Failed to save state during reanalysis request for task %s: %s", task_id, exc)
                raise

            self._broadcast_event(
                TraceEvent(
                    event_id=str(uuid.uuid4())[:8],
                    task_id=task_id,
                    event_type="human_gate",
                    message=f"Operator requested RE-ANALYSIS with feedback: {feedback}",
                    details={"action": action_val, "feedback": feedback},
                )
            )

            if orchestrator is not None:
                def on_node(s):
                    try:
                        self.state_store.save_state(task_id, s)
                    except Exception as exc:
                        logger.exception("Failed to save state during resume on_node for task %s: %s", task_id, exc)
                        raise

                def _resume_worker():
                    try:
                        final_state = orchestrator.resume(
                            task_id=task_id,
                            decision={"action": action_val, "feedback": feedback},
                            on_node_finish=on_node,
                        )
                        try:
                            self.state_store.save_state(task_id, final_state)
                        except Exception as exc:
                            logger.exception("Failed to save final state after resume for task %s: %s", task_id, exc)
                            raise
                    except Exception as e:
                        logger.exception("Resume reanalysis failed for task %s: %s", task_id, e)
                        try:
                            err_st = self.state_store.get_state(task_id) or state
                            err_st["status"] = TaskStatus.FAILED
                            err_st["errors"].append(str(e))
                            self.state_store.save_state(task_id, err_st)
                        except Exception as save_err:
                            logger.exception("Failed to save failed state for task %s: %s", task_id, save_err)
                    finally:
                        st = self.state_store.get_state(task_id)
                        if st and st.get("status") in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED):
                            self._orchestrators.pop(task_id, None)
                            if hasattr(orchestrator, "task_context") and orchestrator.task_context is not None:
                                try:
                                    orchestrator.task_context.cleanup()
                                except Exception as exc:
                                    logger.exception("Failed to cleanup task context: %s", exc)

                thread = threading.Thread(target=_resume_worker, daemon=True)
                thread.start()

            return self.state_store.get_response(task_id)

        # APPROVE or REJECT:
        if orchestrator is not None:
            def on_node(s):
                try:
                    self.state_store.save_state(task_id, s)
                except Exception as exc:
                    logger.exception("Failed to save state during resume on_node for task %s: %s", task_id, exc)
                    raise

            final_state = orchestrator.resume(
                task_id=task_id,
                decision={"action": action_val, "feedback": feedback},
                on_node_finish=on_node,
            )
            try:
                self.state_store.save_state(task_id, final_state)
            except Exception as exc:
                logger.exception("Failed to save final state after resume for task %s: %s", task_id, exc)
                raise

            if final_state.get("status") in (TaskStatus.COMPLETED, TaskStatus.REJECTED, TaskStatus.FAILED):
                self._orchestrators.pop(task_id, None)
                if hasattr(orchestrator, "task_context") and orchestrator.task_context is not None:
                    try:
                        orchestrator.task_context.cleanup()
                    except Exception as exc:
                        logger.exception("Failed to cleanup task context: %s", exc)
        else:
            req = state.get("human_approval") or HumanApprovalRequest(task_id=task_id)
            req.decided_at = datetime.now(UTC)
            req.feedback = feedback

            if action == ApprovalAction.APPROVE:
                req.status = "approved"
                state["status"] = TaskStatus.COMPLETED
                msg = "Operator APPROVED the AI recommendation."
            elif action == ApprovalAction.REJECT:
                req.status = "rejected"
                state["status"] = TaskStatus.REJECTED
                msg = f"Operator REJECTED recommendation. Reason: {feedback or 'No comment'}"
            else:
                msg = f"Operator decision: {action_val}"

            state["human_approval"] = req
            self._broadcast_event(
                TraceEvent(
                    event_id=str(uuid.uuid4())[:8],
                    task_id=task_id,
                    event_type="human_gate",
                    message=msg,
                    details={"action": action_val, "feedback": feedback},
                )
            )
            try:
                self.state_store.save_state(task_id, state)
            except Exception as exc:
                logger.exception("Failed to save state in fallback approval for task %s: %s", task_id, exc)
                raise

        return self.state_store.get_response(task_id)


_global_orchestrator_service: Optional[OrchestratorService] = None


def get_orchestrator_service() -> OrchestratorService:
    global _global_orchestrator_service
    if _global_orchestrator_service is None:
        _global_orchestrator_service = OrchestratorService()
    return _global_orchestrator_service
