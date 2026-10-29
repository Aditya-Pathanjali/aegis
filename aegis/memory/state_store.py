"""State storage and persistence repository for tasks and human reviews."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from aegis.models.schemas import TaskResponse, TaskStatus
from aegis.orchestration.state import AegisState


class TaskStateStore:
    """Manages active and historical task state records."""

    def __init__(self):
        self._states: Dict[str, AegisState] = {}
        self._created_timestamps: Dict[str, datetime] = {}
        self._updated_timestamps: Dict[str, datetime] = {}

    def save_state(self, task_id: str, state: AegisState) -> None:
        now = datetime.utcnow()
        if task_id not in self._created_timestamps:
            self._created_timestamps[task_id] = now
        self._updated_timestamps[task_id] = now
        self._states[task_id] = state

    def get_state(self, task_id: str) -> Optional[AegisState]:
        return self._states.get(task_id)

    def get_response(self, task_id: str) -> Optional[TaskResponse]:
        state = self._states.get(task_id)
        if not state:
            return None

        status = state.get("status", TaskStatus.PENDING)
        if isinstance(status, str):
            status = TaskStatus(status)

        return TaskResponse(
            task_id=task_id,
            status=status,
            plan=state.get("plan"),
            report=state.get("final_report"),
            approval_request=state.get("human_approval"),
            trace_events=state.get("trace_events", []),
            created_at=self._created_timestamps.get(task_id, datetime.utcnow()),
            updated_at=self._updated_timestamps.get(task_id, datetime.utcnow()),
        )

    def list_tasks(self) -> List[TaskResponse]:
        responses = []
        for task_id in list(self._states.keys()):
            resp = self.get_response(task_id)
            if resp:
                responses.append(resp)
        responses.sort(key=lambda x: x.created_at, reverse=True)
        return responses


_global_state_store: Optional[TaskStateStore] = None


def get_state_store() -> TaskStateStore:
    global _global_state_store
    if _global_state_store is None:
        _global_state_store = TaskStateStore()
    return _global_state_store
