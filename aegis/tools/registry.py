"""Tool Registry with role-based access control and execution governance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aegis.models.schemas import AgentRole, ToolCallRecord
from aegis.tools.base import BaseTool, ToolResult
from aegis.tools.python_sandbox import PythonSandboxTool
from aegis.tools.rag_tool import RAGRetrieverTool
from aegis.tools.sql_tool import SQLQueryTool


class UnauthorizedToolAccessError(PermissionError):
    """Raised when an agent attempts to invoke a tool outside its security boundary."""
    pass


class ToolRegistry:
    """
    Central repository for all Aegis tools with role-based execution governance.
    Restricts tool execution strictly to designated agent roles.
    """

    # Explicit role-based permissions matrix
    ROLE_PERMISSIONS: Dict[AgentRole, List[str]] = {
        AgentRole.PLANNER: [],  # Planner only reasons & plans; no tool execution
        AgentRole.RESEARCH: ["rag_retriever"],
        AgentRole.DATA_ANALYST: ["python_sandbox", "sql_query"],
        AgentRole.VERIFICATION: ["rag_retriever", "python_sandbox"],
        AgentRole.CRITIC: ["rag_retriever"],
        AgentRole.SYNTHESIZER: [],  # Synthesizer only compiles validated outputs
    }

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register(RAGRetrieverTool())
        self.register(PythonSandboxTool())
        self.register(SQLQueryTool())

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)

    def list_available_tools(self, role: Optional[AgentRole] = None) -> List[str]:
        if role is None:
            return list(self._tools.keys())
        allowed = self.ROLE_PERMISSIONS.get(role, [])
        return [name for name in allowed if name in self._tools]

    def can_agent_use(self, role: AgentRole, tool_name: str) -> bool:
        allowed = self.ROLE_PERMISSIONS.get(role, [])
        return tool_name in allowed

    def execute_tool(
        self,
        role: AgentRole,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolCallRecord:
        """Executes a tool under role-based authorization check."""
        if not self.can_agent_use(role, tool_name):
            error_msg = f"Security Policy Violation: Agent role '{role.value}' is NOT authorized to execute tool '{tool_name}'."
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                output=None,
                execution_time_ms=0.0,
                success=False,
                error=error_msg,
            )

        tool = self._tools.get(tool_name)
        if not tool:
            return ToolCallRecord(
                tool_name=tool_name,
                arguments=arguments,
                output=None,
                execution_time_ms=0.0,
                success=False,
                error=f"Tool '{tool_name}' is not registered in Aegis ToolRegistry.",
            )

        result: ToolResult = tool.execute(**arguments)
        return result.to_record(arguments=arguments)


# Global Singleton Registry
_global_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _global_tool_registry
    if _global_tool_registry is None:
        _global_tool_registry = ToolRegistry()
    return _global_tool_registry
