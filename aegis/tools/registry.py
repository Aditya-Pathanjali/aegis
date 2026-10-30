"""Tool Registry with role-based access control and execution governance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from aegis.models.schemas import AgentRole, ToolCallRecord
from aegis.tools.base import BaseTool, ToolResult
from aegis.tools.describe_tool import DescribeDatasetTool
from aegis.tools.python_sandbox import PythonExecTool
from aegis.tools.rag_tool import RAGSearchTool
from aegis.tools.sql_tool import SQLQueryTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext


class UnauthorizedToolAccessError(PermissionError):
    """Raised when an agent attempts to invoke a tool outside its security boundary."""
    pass


class ToolRegistry:
    """
    Central repository for all Aegis tools with role-based execution governance.
    Restricts tool execution strictly to designated agent roles.
    """

    ROLE_PERMISSIONS: Dict[AgentRole, List[str]] = {
        AgentRole.PLANNER: [],
        AgentRole.RESEARCH: ["rag_search"],
        AgentRole.DATA_ANALYST: ["describe_dataset", "sql_query", "python_exec"],
        AgentRole.VERIFICATION: ["rag_search", "python_exec"],
        AgentRole.CRITIC: ["rag_search"],
        AgentRole.SYNTHESIZER: [],
    }

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        self.register(RAGSearchTool())
        self.register(DescribeDatasetTool())
        self.register(SQLQueryTool())
        self.register(PythonExecTool())

    @classmethod
    def from_context(cls, context: TaskContext) -> ToolRegistry:
        """Constructs a tool registry bound to a specific task context."""
        registry = cls()
        for tool in context.tools.values():
            registry.register(tool)
        return registry

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


_global_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _global_tool_registry
    if _global_tool_registry is None:
        _global_tool_registry = ToolRegistry()
    return _global_tool_registry
