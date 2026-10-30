"""Planner Agent for task decomposition, acyclic validation, repair, and dependency ordering."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from aegis.agents.base import BaseAgent
from aegis.llm.types import Message, StructuredResult, TextBlock, Usage
from aegis.models.llm_outputs import PlanOutput
from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask
from aegis.tools.registry import ToolRegistry, get_tool_registry


def validate_plan(plan: ExecutionPlan, has_dataset: bool, tool_registry: ToolRegistry) -> List[str]:
    """Validates plan against structural, role, dataset, permission, and acyclicity constraints."""
    errors: List[str] = []
    if not plan.subtasks:
        errors.append("Plan must contain at least 1 subtask.")
        return errors

    if len(plan.subtasks) > 6:
        errors.append(f"Plan exceeds maximum allowed 6 subtasks (got {len(plan.subtasks)}).")

    allowed_roles = {AgentRole.RESEARCH, AgentRole.DATA_ANALYST}
    subtask_ids: set[str] = set()

    for idx, st in enumerate(plan.subtasks):
        if not st.id:
            st.id = f"subtask-{idx+1}"

        if st.id in subtask_ids:
            errors.append(f"Duplicate subtask ID '{st.id}'.")
        else:
            subtask_ids.add(st.id)

        if st.assigned_agent not in allowed_roles:
            errors.append(
                f"Subtask '{st.id}' assigned to unauthorized role '{st.assigned_agent}'. "
                f"Roles are strictly limited to 'research' and 'data_analyst'."
            )

        if st.assigned_agent == AgentRole.DATA_ANALYST and not has_dataset:
            errors.append(
                f"Subtask '{st.id}' assigned to data_analyst, but no dataset is available for this task."
            )

        # Validate tool permissions
        allowed_tools = set(tool_registry.list_available_tools(st.assigned_agent))
        for tool_name in st.required_tools:
            if tool_name not in allowed_tools:
                errors.append(
                    f"Tool '{tool_name}' in subtask '{st.id}' is not permitted for role '{st.assigned_agent.value}'. "
                    f"Permitted tools: {sorted(list(allowed_tools))}."
                )

    # Validate dependencies exist and are not self-referential
    for st in plan.subtasks:
        for dep in st.dependencies:
            if dep == st.id:
                errors.append(f"Subtask '{st.id}' cannot depend on itself.")
            if dep not in subtask_ids:
                errors.append(f"Subtask '{st.id}' has unknown dependency '{dep}'.")

    # Acyclicity check via Kahn's algorithm
    if not any("dependency" in e or "Duplicate" in e for e in errors):
        adj: Dict[str, List[str]] = {st.id: [] for st in plan.subtasks}
        in_degree: Dict[str, int] = {st.id: 0 for st in plan.subtasks}
        for st in plan.subtasks:
            for dep in st.dependencies:
                adj[dep].append(st.id)
                in_degree[st.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        visited_count = 0
        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if visited_count < len(plan.subtasks):
            errors.append("Cyclic dependency detected among plan subtasks.")

    return errors


def topological_sort_subtasks(subtasks: List[SubTask]) -> List[SubTask]:
    """Orders subtasks such that all dependencies precede their dependents."""
    by_id = {st.id: st for st in subtasks}
    adj: Dict[str, List[str]] = {st.id: [] for st in subtasks}
    in_degree: Dict[str, int] = {st.id: 0 for st in subtasks}

    for st in subtasks:
        for dep in st.dependencies:
            if dep in adj:
                adj[dep].append(st.id)
                in_degree[st.id] += 1

    queue = [sid for sid, deg in in_degree.items() if deg == 0]
    sorted_subtasks: List[SubTask] = []
    while queue:
        curr_id = queue.pop(0)
        sorted_subtasks.append(by_id[curr_id])
        for nxt_id in adj[curr_id]:
            in_degree[nxt_id] -= 1
            if in_degree[nxt_id] == 0:
                queue.append(nxt_id)

    if len(sorted_subtasks) < len(subtasks):
        remaining = [st for st in subtasks if st not in sorted_subtasks]
        sorted_subtasks.extend(remaining)

    return sorted_subtasks


class PlannerAgent(BaseAgent):
    """
    Decomposes complex natural-language engineering requests into logically ordered,
    specialized subtasks with validation, repair, and dependency execution order.
    """

    SYSTEM_PROMPT = (
        "You are the Aegis Lead Architect & Planner Agent.\n"
        "Rules:\n"
        "1. Decompose the request into maximum 6 atomic subtasks.\n"
        "2. Roles are strictly limited to 'research' and 'data_analyst'.\n"
        "3. Do not assign 'data_analyst' if no dataset is provided.\n"
        "4. Assign only permitted tools: research -> ['rag_search']; "
        "data_analyst -> ['describe_dataset', 'sql_query', 'python_exec'].\n"
        "5. Ensure dependencies are strictly acyclic and refer only to valid subtask IDs."
    )

    def __init__(self, llm_provider=None, tool_registry=None):
        super().__init__(role=AgentRole.PLANNER, llm_provider=llm_provider, tool_registry=tool_registry)
        self.last_structured_result: Optional[StructuredResult] = None

    def _build_execution_plan(self, output_val: Any, query: str) -> ExecutionPlan:
        """Converts an LLM-facing PlanOutput into a runtime ExecutionPlan, or passes through ExecutionPlan."""
        if isinstance(output_val, PlanOutput):
            subtasks = [
                SubTask(
                    id=ps.id,
                    description=ps.description,
                    role=ps.role,
                    tools=list(ps.tools),
                    depends_on=list(ps.depends_on),
                    title=ps.description,
                    status="pending",
                )
                for ps in output_val.subtasks
            ]
            return ExecutionPlan(
                subtasks=subtasks,
                task_id=str(uuid.uuid4())[:8],
                original_query=query,
                objective=output_val.objective,
                rationale=output_val.rationale,
            )
        elif isinstance(output_val, ExecutionPlan):
            output_val.task_id = output_val.task_id or str(uuid.uuid4())[:8]
            output_val.original_query = query
            return output_val
        raise TypeError(f"Expected PlanOutput or ExecutionPlan, got {type(output_val)}")

    def execute(
        self,
        query: str,
        has_dataset: bool = False,
        dataset_info: Optional[Dict[str, Any]] = None,
        critic_feedback: Optional[str] = None,
    ) -> ExecutionPlan:
        """Decomposes request into a validated, dependency-ordered ExecutionPlan with repair."""
        prompt = (
            f"User Engineering Request:\n\"{query}\"\n\n"
            f"Dataset Available: {'Yes' if has_dataset else 'No'}\n"
            f"Dataset Context: {dataset_info or 'None'}\n"
        )
        if critic_feedback:
            prompt += f"\nCRITIC RE-PLANNING FEEDBACK:\n\"{critic_feedback}\"\nRefine the plan to address these specific gaps."

        msg = Message(role="user", content=[TextBlock(text=prompt)])

        # Initial generation attempt
        result: StructuredResult = self.llm.structured(
            messages=[msg],
            schema=PlanOutput,
            system=self.SYSTEM_PROMPT,
        )
        self.last_structured_result = result
        plan: ExecutionPlan = self._build_execution_plan(result.value, query)

        reg = self.tool_registry or get_tool_registry()
        errors = validate_plan(plan, has_dataset=has_dataset, tool_registry=reg)

        if errors:
            # One repair attempt
            repair_text = (
                "The previously generated plan failed validation with the following errors:\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\nPlease correct these errors and regenerate a valid PlanOutput conforming strictly to all rules."
            )
            repair_messages = [
                msg,
                Message(role="assistant", content=[TextBlock(text=f"Generated plan: {plan.objective}")]),
                Message(role="user", content=[TextBlock(text=repair_text)]),
            ]

            repair_result: StructuredResult = self.llm.structured(
                messages=repair_messages,
                schema=PlanOutput,
                system=self.SYSTEM_PROMPT,
            )

            # Accumulate usage tokens across initial and repair attempts
            if self.last_structured_result and repair_result:
                total_in = self.last_structured_result.usage.input_tokens + repair_result.usage.input_tokens
                total_out = self.last_structured_result.usage.output_tokens + repair_result.usage.output_tokens
                repair_result.usage = Usage(input_tokens=total_in, output_tokens=total_out)
            self.last_structured_result = repair_result

            plan = self._build_execution_plan(repair_result.value, query)

            second_errors = validate_plan(plan, has_dataset=has_dataset, tool_registry=reg)
            if second_errors:
                raise ValueError(
                    f"ExecutionPlan validation failed after repair attempt: {'; '.join(second_errors)}"
                )

        plan.subtasks = topological_sort_subtasks(plan.subtasks)
        return plan
