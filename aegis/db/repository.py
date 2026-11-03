"""Postgres repository for tasks, states, trace events, and human approvals."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from aegis.db.models import HumanApprovalModel, TaskModel, TraceEventModel
from aegis.db.session import get_db_session, get_session_factory
from aegis.models.schemas import (
    AgentExecutionResult,
    CriticEvaluation,
    ExecutionPlan,
    FinalReport,
    HumanApprovalRequest,
    TaskResponse,
    TaskStatus,
    TraceEvent,
    VerificationResult,
)
from aegis.orchestration.state import AegisState

logger = logging.getLogger(__name__)


def _to_json_serializable(val: Any) -> Any:
    """Recursively converts Pydantic models, datetimes, and enums to json-safe dicts/primitives."""
    if hasattr(val, "model_dump"):
        return val.model_dump(mode="json")
    if isinstance(val, dict):
        return {k: _to_json_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [_to_json_serializable(item) for item in val]
    if isinstance(val, datetime):
        return val.isoformat()
    if hasattr(val, "value"):
        return val.value
    return val


def _reconstruct_state(raw: Dict[str, Any]) -> AegisState:
    """Reconstructs typed Pydantic models from persisted json dictionary."""
    state = dict(raw)

    if state.get("plan") and isinstance(state["plan"], dict):
        try:
            state["plan"] = ExecutionPlan.model_validate(state["plan"])
        except Exception as e:
            logger.debug("Could not validate plan model: %s", e)

    if state.get("final_report") and isinstance(state["final_report"], dict):
        try:
            state["final_report"] = FinalReport.model_validate(state["final_report"])
        except Exception as e:
            logger.debug("Could not validate final_report model: %s", e)

    if state.get("human_approval") and isinstance(state["human_approval"], dict):
        try:
            state["human_approval"] = HumanApprovalRequest.model_validate(state["human_approval"])
        except Exception as e:
            logger.debug("Could not validate human_approval model: %s", e)

    if state.get("critic_evaluation") and isinstance(state["critic_evaluation"], dict):
        try:
            state["critic_evaluation"] = CriticEvaluation.model_validate(state["critic_evaluation"])
        except Exception as e:
            logger.debug("Could not validate critic_evaluation model: %s", e)

    if state.get("verification_result") and isinstance(state["verification_result"], dict):
        try:
            state["verification_result"] = VerificationResult.model_validate(state["verification_result"])
        except Exception as e:
            logger.debug("Could not validate verification_result model: %s", e)

    if "specialist_results" in state and isinstance(state["specialist_results"], dict):
        reconstructed_spec: Dict[str, Any] = {}
        for k, v in state["specialist_results"].items():
            if isinstance(v, dict):
                try:
                    reconstructed_spec[k] = AgentExecutionResult.model_validate(v)
                except Exception:
                    reconstructed_spec[k] = v
            else:
                reconstructed_spec[k] = v
        state["specialist_results"] = reconstructed_spec

    if "trace_events" in state and isinstance(state["trace_events"], list):
        evs = []
        for ev in state["trace_events"]:
            if isinstance(ev, dict):
                try:
                    evs.append(TraceEvent.model_validate(ev))
                except Exception:
                    pass
            elif isinstance(ev, TraceEvent):
                evs.append(ev)
        state["trace_events"] = evs

    if "status" in state:
        st = state["status"]
        if isinstance(st, str):
            try:
                state["status"] = TaskStatus(st)
            except ValueError:
                pass

    return state  # type: ignore[return-value]


class PostgresTaskStore:
    """PostgreSQL-backed store for Aegis task states, trace events, and human approvals."""

    def __init__(self, session_factory: Optional[sessionmaker[Session]] = None):
        self._session_factory = session_factory

    def _get_factory(self) -> sessionmaker[Session]:
        return self._session_factory or get_session_factory()

    def save_state(self, task_id: str, state: AegisState) -> None:
        raw_status = state.get("status", TaskStatus.PENDING)
        status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        serialized_state = _to_json_serializable(dict(state))

        with get_db_session(self._get_factory()) as session:
            task = session.get(TaskModel, task_id)
            if not task:
                task = TaskModel(
                    task_id=task_id,
                    query=state.get("query", ""),
                    dataset_name=state.get("dataset_name"),
                    dataset_content=state.get("dataset_content"),
                    status=status_str,
                    state_data=serialized_state,
                )
                session.add(task)
            else:
                task.status = status_str
                task.state_data = serialized_state
                task.updated_at = datetime.now(UTC)
                if state.get("dataset_name"):
                    task.dataset_name = state["dataset_name"]
                if state.get("dataset_content"):
                    task.dataset_content = state["dataset_content"]

            # Synchronize Human Approval Request
            approval = state.get("human_approval")
            if approval:
                appr_dict: Dict[str, Any] = (
                    approval.model_dump(mode="json")
                    if hasattr(approval, "model_dump")
                    else dict(approval)  # type: ignore[arg-type]
                )
                appr_model = session.get(HumanApprovalModel, task_id)
                if not appr_model:
                    appr_model = HumanApprovalModel(
                        task_id=task_id,
                        status=appr_dict.get("status", "pending"),
                        required=appr_dict.get("required", True),
                        proposed_actions=appr_dict.get("proposed_actions", []),
                        feedback=appr_dict.get("feedback"),
                    )
                    session.add(appr_model)
                else:
                    appr_model.status = appr_dict.get("status", appr_model.status)
                    appr_model.feedback = appr_dict.get("feedback", appr_model.feedback)
                    appr_model.proposed_actions = appr_dict.get("proposed_actions", appr_model.proposed_actions)
                    if appr_dict.get("decided_at"):
                        dec_at = appr_dict["decided_at"]
                        if isinstance(dec_at, str):
                            dec_at = datetime.fromisoformat(dec_at)
                        appr_model.decided_at = dec_at

            # Synchronize Trace Events
            trace_events = state.get("trace_events", [])
            if trace_events:
                existing_event_ids = set(
                    session.scalars(
                        select(TraceEventModel.event_id).where(TraceEventModel.task_id == task_id)
                    ).all()
                )
                for ev in trace_events:
                    ev_dict: Dict[str, Any] = (
                        ev.model_dump(mode="json")
                        if hasattr(ev, "model_dump")
                        else dict(ev)  # type: ignore[arg-type]
                    )
                    ev_id = ev_dict.get("event_id")
                    if ev_id and ev_id not in existing_event_ids:
                        t_stamp = ev_dict.get("timestamp")
                        if isinstance(t_stamp, str):
                            t_stamp = datetime.fromisoformat(t_stamp)
                        elif not isinstance(t_stamp, datetime):
                            t_stamp = datetime.now(UTC)
                        new_ev = TraceEventModel(
                            event_id=ev_id,
                            task_id=task_id,
                            event_type=ev_dict.get("event_type", "unknown"),
                            agent_role=ev_dict.get("agent_role"),
                            message=ev_dict.get("message", ""),
                            details=ev_dict.get("details", {}),
                            timestamp=t_stamp,
                        )
                        session.add(new_ev)
                        existing_event_ids.add(ev_id)

    def get_state(self, task_id: str) -> Optional[AegisState]:
        with get_db_session(self._get_factory()) as session:
            task = session.get(TaskModel, task_id)
            if not task or not task.state_data:
                return None
            return _reconstruct_state(task.state_data)

    def get_response(self, task_id: str) -> Optional[TaskResponse]:
        with get_db_session(self._get_factory()) as session:
            task = session.get(TaskModel, task_id)
            if not task:
                return None

            state = _reconstruct_state(task.state_data) if task.state_data else {}
            status = state.get("status", TaskStatus.PENDING)
            if isinstance(status, str):
                try:
                    status = TaskStatus(status)
                except ValueError:
                    status = TaskStatus.PENDING

            # Load trace events from relation/query
            ev_models = session.scalars(
                select(TraceEventModel)
                .where(TraceEventModel.task_id == task_id)
                .order_by(TraceEventModel.timestamp)
            ).all()

            trace_events: List[TraceEvent] = []
            for em in ev_models:
                trace_events.append(
                    TraceEvent(
                        event_id=em.event_id,
                        task_id=em.task_id,
                        event_type=em.event_type,
                        agent_role=em.agent_role,  # type: ignore[arg-type]
                        message=em.message,
                        details=em.details,
                        timestamp=em.timestamp,
                    )
                )

            # Fall back to state trace events if empty
            if not trace_events and state.get("trace_events"):
                trace_events = state["trace_events"]

            return TaskResponse(
                task_id=task.task_id,
                status=status,
                plan=state.get("plan"),
                report=state.get("final_report"),
                approval_request=state.get("human_approval"),
                trace_events=trace_events,
                created_at=task.created_at,
                updated_at=task.updated_at,
            )

    def list_tasks(self) -> List[TaskResponse]:
        with get_db_session(self._get_factory()) as session:
            tasks = session.scalars(
                select(TaskModel).order_by(TaskModel.created_at.desc())
            ).all()
            results: List[TaskResponse] = []
            for t in tasks:
                resp = self.get_response(t.task_id)
                if resp:
                    results.append(resp)
            return results

    def save_trace_event(self, event: TraceEvent) -> None:
        with get_db_session(self._get_factory()) as session:
            ev_id = event.event_id
            existing = session.scalar(
                select(TraceEventModel).where(
                    TraceEventModel.task_id == event.task_id,
                    TraceEventModel.event_id == ev_id,
                )
            )
            if not existing:
                new_ev = TraceEventModel(
                    event_id=ev_id,
                    task_id=event.task_id,
                    event_type=event.event_type,
                    agent_role=event.agent_role.value if (event.agent_role and hasattr(event.agent_role, "value")) else (event.agent_role or None),
                    message=event.message,
                    details=event.details,
                    timestamp=event.timestamp or datetime.now(UTC),
                )
                session.add(new_ev)

    def get_trace_events(self, task_id: str) -> List[TraceEvent]:
        with get_db_session(self._get_factory()) as session:
            ev_models = session.scalars(
                select(TraceEventModel)
                .where(TraceEventModel.task_id == task_id)
                .order_by(TraceEventModel.timestamp)
            ).all()
            return [
                TraceEvent(
                    event_id=em.event_id,
                    task_id=em.task_id,
                    event_type=em.event_type,
                    agent_role=em.agent_role,  # type: ignore[arg-type]
                    message=em.message,
                    details=em.details,
                    timestamp=em.timestamp,
                )
                for em in ev_models
            ]


_global_task_store: Optional[PostgresTaskStore] = None


def get_task_store() -> PostgresTaskStore:
    global _global_task_store
    if _global_task_store is None:
        _global_task_store = PostgresTaskStore()
    return _global_task_store

