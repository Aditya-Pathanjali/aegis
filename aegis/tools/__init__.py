"""Aegis Tools package."""

from aegis.tools.base import BaseTool, ToolResult
from aegis.tools.context import TaskContext
from aegis.tools.describe_tool import DescribeDatasetTool
from aegis.tools.python_sandbox import PythonExecTool, SecurityViolationError
from aegis.tools.rag_tool import RAGSearchTool
from aegis.tools.registry import ToolRegistry, get_tool_registry
from aegis.tools.sql_tool import SQLQueryTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "TaskContext",
    "DescribeDatasetTool",
    "RAGSearchTool",
    "PythonExecTool",
    "SecurityViolationError",
    "SQLQueryTool",
    "ToolRegistry",
    "get_tool_registry",
]
