"""Base tool interface and execution contracts."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel, Field

from aegis.models.schemas import ToolCallRecord


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_record(self, arguments: Dict[str, Any]) -> ToolCallRecord:
        return ToolCallRecord(
            tool_name=self.tool_name,
            arguments=arguments,
            output=self.data if self.success else None,
            execution_time_ms=self.execution_time_ms,
            success=self.success,
            error=self.error,
        )


class BaseTool(ABC):
    """Abstract Base Class for Aegis Tools."""

    name: str = "base_tool"
    description: str = "Base tool description"
    parameters: Dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolResult:
        start_time = time.perf_counter()
        try:
            data = self._run(**kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                execution_time_ms=round(duration_ms, 2),
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=round(duration_ms, 2),
            )

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        pass
