"""Trace logging and audit repository for Aegis execution observability."""

from __future__ import annotations

from typing import Dict, List, Optional

from aegis.models.schemas import TraceEvent


class TraceLogger:
    """Maintains an ordered, indexed repository of workflow trace events."""

    def __init__(self):
        self._traces_by_task: Dict[str, List[TraceEvent]] = {}

    def log(self, event: TraceEvent) -> None:
        if event.task_id not in self._traces_by_task:
            self._traces_by_task[event.task_id] = []
        self._traces_by_task[event.task_id].append(event)

    def get_events(self, task_id: str) -> List[TraceEvent]:
        return self._traces_by_task.get(task_id, [])

    def clear(self, task_id: Optional[str] = None) -> None:
        if task_id:
            self._traces_by_task.pop(task_id, None)
        else:
            self._traces_by_task.clear()


_global_trace_logger: Optional[TraceLogger] = None


def get_trace_logger() -> TraceLogger:
    global _global_trace_logger
    if _global_trace_logger is None:
        _global_trace_logger = TraceLogger()
    return _global_trace_logger
