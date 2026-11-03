"""Aegis Worker Process executing LangGraph orchestrator jobs with arq and Redis."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from arq.worker import run_worker

from aegis.config import get_settings
from aegis.db import get_task_store
from aegis.llm import LLMProvider, get_llm_provider
from aegis.models.schemas import (
    ApprovalAction,
    HumanApprovalRequest,
    TaskStatus,
    TaskSubmission,
    TraceEvent,
)
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from aegis.queue import (
    get_redis_settings,
    is_task_cancelled,
    publish_trace_event_sync,
)
from aegis.tools.context import TaskContext
from aegis.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class TaskCancelledException(Exception):
    """Raised when a task is aborted due to user cancellation."""
    pass


class TaskTimedOutException(Exception):
    """Raised inside the task thread once its wall-clock timeout has passed."""
    pass


class _StopGuardLLM:
    """Wraps a task's LLM so a cancel or timeout stops the very next LLM call.

    The graph runs in a worker thread, and asyncio.wait_for cannot kill a thread: without this
    guard a timed-out or cancelled task kept calling (and paying for) the LLM until it finished.
    """

    def __init__(self, inner: LLMProvider, task_id: str, timed_out: threading.Event):
        self._inner = inner
        self._task_id = task_id
        self._timed_out = timed_out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _check(self) -> None:
        if self._timed_out.is_set():
            raise TaskTimedOutException(f"Task {self._task_id} exceeded its timeout.")
        if is_task_cancelled(self._task_id):
            raise TaskCancelledException(f"Task {self._task_id} was cancelled.")

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return self._inner.chat(*args, **kwargs)

    def structured(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        return self._inner.structured(*args, **kwargs)


def _make_on_node(store: Any, task_id: str, timed_out: threading.Event) -> Any:
    """Per-node callback: persist progress unless the task was cancelled or timed out."""

    def on_node(current_state: AegisState) -> None:
        if timed_out.is_set():
            # The job already recorded FAILED; never overwrite it from the lingering thread.
            raise TaskTimedOutException(f"Task {task_id} exceeded its timeout.")
        if is_task_cancelled(task_id):
            current_state["status"] = TaskStatus.CANCELLED
            store.save_state(task_id, current_state)
            raise TaskCancelledException(f"Task {task_id} was cancelled.")
        store.save_state(task_id, current_state)

    return on_node


def _publish_event_both(event: TraceEvent) -> None:
    """Persists trace event to Postgres and broadcasts to Redis pub/sub."""
    store = get_task_store()
    try:
        store.save_trace_event(event)
    except Exception as exc:
        logger.debug("Failed to save trace event to DB: %s", exc)

    publish_trace_event_sync(event)


async def run_task_job(
    ctx: Dict[str, Any],
    task_id: str,
    submission_dict: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Worker job executing a task end-to-end via LangGraph."""
    logger.info("Worker starting task job %s", task_id)
    store = get_task_store()
    cfg = get_settings()
    timeout_s = int(ctx.get("job_timeout") or cfg.task_timeout_s)

    # 1. Check for immediate cancellation while waiting in queue
    if is_task_cancelled(task_id):
        logger.info("Task %s was cancelled before worker pickup.", task_id)
        st = store.get_state(task_id) or {}
        st["status"] = TaskStatus.CANCELLED
        store.save_state(task_id, st)
        return {"status": "cancelled", "task_id": task_id}

    submission = TaskSubmission.model_validate(submission_dict)
    initial_state = store.get_state(task_id)
    if not initial_state:
        has_data = bool(submission.dataset_content or submission.dataset_name)
        initial_state = {
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
        store.save_state(task_id, initial_state)

    task_context: Optional[TaskContext] = None
    try:
        df = None
        if submission.dataset_content:
            try:
                df = pd.read_csv(io.StringIO(submission.dataset_content))
            except Exception as e:
                raise ValueError(f"Failed to parse dataset content as CSV: {e}") from e

        task_context = TaskContext(dataset=df, task_id=task_id)
        tool_reg = ToolRegistry.from_context(task_context)

        # Allow test injection of custom LLM (e.g. FakeLLM) via ctx
        llm: LLMProvider = ctx.get("llm") or get_llm_provider()
        timed_out = threading.Event()

        orchestrator = AegisOrchestrator(
            llm_provider=_StopGuardLLM(llm, task_id, timed_out),  # type: ignore[arg-type]
            tool_registry=tool_reg,
            trace_callback=_publish_event_both,
            task_context=task_context,
        )
        on_node = _make_on_node(store, task_id, timed_out)

        # Execute orchestrator run inside a thread to keep the asyncio event loop responsive
        try:
            final_state = await asyncio.wait_for(
                asyncio.to_thread(orchestrator.run, initial_state, on_node),
                timeout=float(timeout_s),
            )
        except (TimeoutError, asyncio.TimeoutError):
            timed_out.set()  # the thread stops at its next LLM call or node boundary
            raise

        # Check if paused at human gate
        if final_state.get("status") == TaskStatus.WAITING_APPROVAL:
            logger.info("Task %s paused at human review gate. Slot released.", task_id)
            store.save_state(task_id, final_state)
            return {"status": "waiting_approval", "task_id": task_id}

        store.save_state(task_id, final_state)
        return {"status": str(final_state.get("status")), "task_id": task_id}

    except TaskCancelledException:
        logger.info("Task %s aborted cleanly due to cancellation.", task_id)
        st = store.get_state(task_id) or initial_state
        st["status"] = TaskStatus.CANCELLED
        store.save_state(task_id, st)
        _publish_event_both(
            TraceEvent(
                event_id=uuid.uuid4().hex[:8],
                task_id=task_id,
                event_type="task_cancelled",
                message=f"Task {task_id} was cancelled by operator.",
                details={"status": "cancelled"},
            )
        )
        return {"status": "cancelled", "task_id": task_id}

    except (TimeoutError, asyncio.TimeoutError) as exc:
        err_msg = f"Task execution timed out after {timeout_s} seconds limit."
        logger.error("Task %s timed out: %s", task_id, exc)
        st = store.get_state(task_id) or initial_state
        st["status"] = TaskStatus.FAILED
        st["errors"] = st.get("errors", []) + [err_msg]
        store.save_state(task_id, st)
        _publish_event_both(
            TraceEvent(
                event_id=uuid.uuid4().hex[:8],
                task_id=task_id,
                event_type="error",
                message=err_msg,
                details={"error": err_msg},
            )
        )
        return {"status": "failed", "error": err_msg, "task_id": task_id}

    except Exception as exc:
        logger.exception("Task %s failed in worker: %s", task_id, exc)
        st = store.get_state(task_id) or initial_state
        st["status"] = TaskStatus.FAILED
        st["errors"] = st.get("errors", []) + [str(exc)]
        store.save_state(task_id, st)
        _publish_event_both(
            TraceEvent(
                event_id=uuid.uuid4().hex[:8],
                task_id=task_id,
                event_type="error",
                message=f"Workflow failed: {exc}",
                details={"error": str(exc)},
            )
        )
        return {"status": "failed", "error": str(exc), "task_id": task_id}

    finally:
        if task_context is not None:
            try:
                task_context.cleanup()
            except Exception as e:
                logger.debug("Error cleaning task context: %s", e)


async def resume_task_job(
    ctx: Dict[str, Any],
    task_id: str,
    action: str,
    feedback: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Worker job resuming a task after human approval or reanalysis."""
    logger.info("Worker resuming task %s with action %s", task_id, action)
    store = get_task_store()
    state = store.get_state(task_id)
    if not state:
        logger.error("Task %s not found for resume", task_id)
        return {"status": "not_found", "task_id": task_id}

    ApprovalAction(action)
    req: Optional[HumanApprovalRequest] = state.get("human_approval")
    if not req:
        req = HumanApprovalRequest(task_id=task_id)

    task_context: Optional[TaskContext] = None
    timed_out = threading.Event()
    try:
        df = None
        if state.get("dataset_content"):
            try:
                df = pd.read_csv(io.StringIO(state["dataset_content"]))
            except Exception as e:
                # Fail closed: never resume an analysis without the task's data.
                raise ValueError(f"Failed to parse stored dataset content as CSV: {e}") from e
        task_context = TaskContext(dataset=df, task_id=task_id)
        tool_reg = ToolRegistry.from_context(task_context)

        llm: LLMProvider = ctx.get("llm") or get_llm_provider()
        orchestrator = AegisOrchestrator(
            llm_provider=_StopGuardLLM(llm, task_id, timed_out),  # type: ignore[arg-type]
            tool_registry=tool_reg,
            task_context=task_context,
            trace_callback=_publish_event_both,
        )
        on_node = _make_on_node(store, task_id, timed_out)

        cfg = get_settings()
        timeout_s = int(ctx.get("job_timeout") or cfg.task_timeout_s)
        try:
            final_state = await asyncio.wait_for(
                asyncio.to_thread(
                    orchestrator.resume,
                    task_id,
                    {"action": action, "feedback": feedback},
                    on_node,
                ),
                timeout=float(timeout_s),
            )
        except (TimeoutError, asyncio.TimeoutError):
            timed_out.set()
            raise

        store.save_state(task_id, final_state)
        return {"status": str(final_state.get("status")), "task_id": task_id}

    except TaskCancelledException:
        st = store.get_state(task_id) or state
        st["status"] = TaskStatus.CANCELLED
        store.save_state(task_id, st)
        return {"status": "cancelled", "task_id": task_id}
    except Exception as exc:
        logger.exception("Resume failed for task %s: %s", task_id, exc)
        st = store.get_state(task_id) or state
        st["status"] = TaskStatus.FAILED
        st["errors"] = st.get("errors", []) + [str(exc)]
        store.save_state(task_id, st)
        return {"status": "failed", "error": str(exc), "task_id": task_id}
    finally:
        if task_context is not None:
            try:
                task_context.cleanup()
            except Exception:
                pass


async def _sandbox_heartbeat_loop() -> None:
    """Reports sandbox runtime health to Redis for the API's /health (the API has no Docker access)."""
    from aegis.queue import SANDBOX_HEARTBEAT_TTL_S, write_sandbox_heartbeat
    from aegis.tools.python_sandbox import sandbox_runtime_ok

    while True:
        ok = await asyncio.to_thread(sandbox_runtime_ok)
        try:
            write_sandbox_heartbeat(ok)
        except Exception as exc:
            logger.warning("Could not write sandbox heartbeat: %s", exc)
        await asyncio.sleep(SANDBOX_HEARTBEAT_TTL_S / 3)


async def startup(ctx: Dict[str, Any]) -> None:
    ctx["sandbox_heartbeat"] = asyncio.create_task(_sandbox_heartbeat_loop())
    logger.info("Aegis worker process started up successfully.")


async def shutdown(ctx: Dict[str, Any]) -> None:
    task = ctx.get("sandbox_heartbeat")
    if task is not None:
        task.cancel()
    logger.info("Aegis worker process shutting down.")


class WorkerSettings:
    """arq worker configuration settings."""
    functions = [run_task_job, resume_task_job]
    redis_settings = get_redis_settings()
    max_jobs = get_settings().max_concurrent_tasks
    job_timeout = get_settings().task_timeout_s
    allow_abort_jobs = True
    on_startup = startup
    on_shutdown = shutdown


def get_worker_settings():
    """Builds a fresh WorkerSettings class using current settings."""
    class DynamicWorkerSettings:
        functions = [run_task_job, resume_task_job]
        redis_settings = get_redis_settings()
        max_jobs = get_settings().max_concurrent_tasks
        job_timeout = get_settings().task_timeout_s
        allow_abort_jobs = True
        on_startup = startup
        on_shutdown = shutdown
    return DynamicWorkerSettings


def main() -> None:
    """CLI entrypoint for running the Aegis worker."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Starting Aegis worker via python -m aegis.worker ...")
    try:
        run_worker(get_worker_settings())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker terminated.")


if __name__ == "__main__":
    main()

