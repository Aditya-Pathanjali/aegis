"""Aegis Tools package."""

from aegis.tools.base import BaseTool, ToolResult
from aegis.tools.file_processor import DocumentProcessorTool
from aegis.tools.python_sandbox import PythonSandboxTool, SecurityViolationError
from aegis.tools.rag_tool import RAGRetrieverTool
from aegis.tools.registry import ToolRegistry, get_tool_registry
from aegis.tools.rest_api import GitHubTool, RestApiTool
from aegis.tools.sql_tool import SQLQueryTool
from aegis.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "WebSearchTool",
    "RAGRetrieverTool",
    "PythonSandboxTool",
    "SecurityViolationError",
    "SQLQueryTool",
    "DocumentProcessorTool",
    "RestApiTool",
    "GitHubTool",
    "ToolRegistry",
    "get_tool_registry",
]
