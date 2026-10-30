"""Research Agent for literature, RAG retrieval, and empirical evidence gathering."""

from __future__ import annotations

from typing import Any, Dict, Optional

from aegis.agents.base import BaseAgent
from aegis.agents.tool_agent import run_tool_agent
from aegis.models.schemas import AgentExecutionResult, AgentRole, SubTask


class ResearchAgent(BaseAgent):
    """Retrieves and synthesizes empirical technical evidence from RAG knowledge bases."""

    SYSTEM_PROMPT = (
        "You are the Aegis Research Specialist Agent.\n"
        "Rules:\n"
        "1. Cite retrieved chunks: Query the RAG knowledge base using rag_search to find relevant standards, manuals, and papers.\n"
        "2. Formulate atomic, verifiable claims: Exactly one clear, atomic, verifiable technical claim per finding.\n"
        "3. Acknowledge when no info found: If the retrieved documents do not contain sufficient evidence, state this clearly in limitations.\n"
        "4. Every finding must cite valid evidence references (chunk IDs or tool_call IDs) produced directly during this execution run."
    )

    def __init__(self, llm_provider=None, tool_registry=None):
        super().__init__(role=AgentRole.RESEARCH, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(self, subtask: SubTask, context: Optional[Dict[str, Any]] = None) -> AgentExecutionResult:
        user_prompt = (
            f"Research Subtask: {subtask.description or subtask.title}\n"
        )
        if context and "upstream_context" in context and context["upstream_context"]:
            user_prompt += f"\nUpstream Results:\n{context['upstream_context']}\n"
        if context and "content" in context and context["content"]:
            user_prompt += f"\nContext:\n{context['content']}\n"

        # Permissions: only tools in subtask.tools, restricted to rag_search
        allowed_tools = ["rag_search"]
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
