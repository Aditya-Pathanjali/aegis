"""Base Agent architecture for Aegis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from aegis.models.llm_provider import BaseLLMProvider, get_llm_provider
from aegis.models.schemas import AgentRole, ToolCallRecord
from aegis.tools.registry import ToolRegistry, get_tool_registry


class BaseAgent(ABC):
    """Abstract base class for all Aegis specialized agents."""

    def __init__(
        self,
        role: AgentRole,
        llm_provider: Optional[BaseLLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.role = role
        self.llm = llm_provider or get_llm_provider()
        self.tool_registry = tool_registry or get_tool_registry()

    def get_allowed_tools(self) -> List[str]:
        return self.tool_registry.list_available_tools(self.role)

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallRecord:
        """Invokes a tool with permission enforcement and telemetry recording."""
        return self.tool_registry.execute_tool(
            role=self.role,
            tool_name=tool_name,
            arguments=arguments,
        )

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Executes the agent's core responsibility."""
        pass
