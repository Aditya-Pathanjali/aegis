"""Planner Agent for task decomposition and dynamic specialist routing."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from aegis.agents.base import BaseAgent
from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask


class PlannerAgent(BaseAgent):
    """
    Decomposes complex natural-language engineering requests into logically ordered,
    specialized subtasks with explicit tool requirements and dependency chains.
    """

    def __init__(self, llm_provider=None, tool_registry=None):
        super().__init__(role=AgentRole.PLANNER, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(
        self,
        query: str,
        has_dataset: bool = False,
        dataset_info: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[str] = None,
    ) -> ExecutionPlan:
        """
        Decomposes request into a typed ExecutionPlan.
        """
        prompt = (
            f"User Engineering Request:\n\"{query}\"\n\n"
            f"Dataset Available: {'Yes' if has_dataset else 'No'}\n"
            f"Dataset Context: {dataset_info or 'None'}\n"
        )
        if critic_feedback:
            prompt += f"\nCRITIC RE-PLANNING FEEDBACK:\n\"{critic_feedback}\"\nRefine the plan to address these specific gaps."

        system_prompt = (
            "You are the Aegis Lead Architect & Planner Agent. "
            "Decompose the user's complex request into a sequence of atomic, specialized subtasks. "
            "Available specialist roles: 'research', 'data_analyst', 'coding'. "
            "Do NOT include verification or synthesizer in the subtasks as they are dedicated pipeline stages. "
            "Assign appropriate tools from: web_search, rag_retriever, python_sandbox, sql_query, document_processor."
        )

        plan = self.llm.generate_structured(
            prompt=prompt,
            schema=ExecutionPlan,
            system_prompt=system_prompt,
        )

        # Enforce valid IDs and consistent plan metadata
        plan.task_id = plan.task_id or str(uuid.uuid4())[:8]
        plan.original_query = query
        for i, st in enumerate(plan.subtasks):
            if not st.id:
                st.id = f"subtask-{i+1}"
        return plan
