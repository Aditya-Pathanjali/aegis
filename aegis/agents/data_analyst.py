"""Data Analyst Agent for telemetry analysis, SQL queries, and Python modeling."""

from __future__ import annotations

from typing import Any, Dict, Optional

from aegis.agents.base import BaseAgent
from aegis.agents.tool_agent import run_tool_agent
from aegis.models.schemas import AgentExecutionResult, AgentRole, SubTask


class DataAnalystAgent(BaseAgent):
    """Performs empirical statistical analysis and modeling on uploaded task datasets."""

    SYSTEM_PROMPT = (
        "You are the Aegis Data Analyst Specialist Agent.\n"
        "Rules:\n"
        "1. Run code or SQL: inspect columns and stats with describe_dataset, execute SQL queries on the dataset table with sql_query, or run analytical Python scripts using python_exec.\n"
        "2. Quote exact values: Formulate clear, atomic claims quoting the exact numeric and statistical values obtained from tool execution.\n"
        "3. Document dataset limitations: Explicitly document any data quality issues, missing values, or questions that could not be answered from the dataset in limitations.\n"
        "4. Every finding must cite valid tool_call evidence references produced directly during this execution run."
    )

    def __init__(self, llm_provider=None, tool_registry=None):
        super().__init__(role=AgentRole.DATA_ANALYST, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        user_prompt = (
            f"Data Analysis Subtask: {subtask.description or subtask.title}\n"
        )
        if context and "upstream_context" in context and context["upstream_context"]:
            user_prompt += f"\nUpstream Results:\n{context['upstream_context']}\n"
        if context and "content" in context and context["content"]:
            user_prompt += f"\nDataset:\n{context['content']}\n"
        if context and context.get("critic_guidance"):
            user_prompt += f"\nReviewer guidance for this retry:\n{context['critic_guidance']}\n"

        # Permissions: only tools in subtask.tools, restricted to allowed analyst tools
        allowed_tools = ["describe_dataset", "sql_query", "python_exec", "python_sandbox"]
        subtask_tools = subtask.tools if subtask.tools else allowed_tools
        permitted_tools = [t for t in subtask_tools if t in allowed_tools]

        return run_tool_agent(
            llm_provider=self.llm,
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            permitted_tool_names=permitted_tools,
            tool_registry=self.tool_registry,
            role=self.role,
            subtask_id=subtask.id,
        )
