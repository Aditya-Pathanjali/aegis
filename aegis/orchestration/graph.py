"""Aegis Multi-Agent StateGraph Orchestration Engine using LangGraph."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, START, StateGraph

from aegis.agents.critic import CriticAgent
from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.agents.research import ResearchAgent
from aegis.agents.synthesizer import SynthesizerAgent
from aegis.agents.verification import VerificationAgent
from aegis.llm import LLMProvider, get_llm_provider
from aegis.models.schemas import (
    AgentRole,
    HumanApprovalRequest,
    TaskStatus,
    TraceEvent,
)
from aegis.orchestration.router import (
    evaluate_quality_pass,
    route_human_gate_or_complete,
    route_next_subtask,
)
from aegis.orchestration.state import AegisState
from aegis.tools.registry import ToolRegistry, get_tool_registry


class AegisOrchestrator:
    """
    Compiles and executes the LangGraph state machine governing Aegis agents.
    Provides complete lifecycle state management, real-time trace events,
    tool execution sandboxing, verification audits, critic loops, and human review gates.
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        trace_callback: Optional[Callable[[TraceEvent], None]] = None,
        task_context: Optional[Any] = None,
    ):
        self.llm = llm_provider or get_llm_provider()
        self.task_context = task_context
        if tool_registry is not None:
            self.tool_registry = tool_registry
        elif task_context is not None and hasattr(ToolRegistry, "from_context"):
            self.tool_registry = ToolRegistry.from_context(task_context)
        else:
            self.tool_registry = get_tool_registry()
        self.trace_callback = trace_callback

        # Initialize Agents
        self.planner_agent = PlannerAgent(llm_provider=self.llm, tool_registry=self.tool_registry)
        self.research_agent = ResearchAgent(llm_provider=self.llm, tool_registry=self.tool_registry)
        self.data_analyst_agent = DataAnalystAgent(llm_provider=self.llm, tool_registry=self.tool_registry)
        self.verification_agent = VerificationAgent(llm_provider=self.llm, tool_registry=self.tool_registry)
        self.critic_agent = CriticAgent(llm_provider=self.llm, tool_registry=self.tool_registry)
        self.synthesizer_agent = SynthesizerAgent(llm_provider=self.llm, tool_registry=self.tool_registry)

        # Build Graph
        self.graph = self._build_graph()

    def _log_trace(
        self,
        state: AegisState,
        agent_role: Optional[AgentRole],
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
        token_usage: int = 0,
        cost_usd: Optional[float] = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=str(uuid.uuid4())[:8],
            task_id=state.get("task_id", "unknown"),
            agent_role=agent_role,
            event_type=event_type,
            message=message,
            details=details or {},
            duration_ms=round(duration_ms, 2),
            token_usage=token_usage,
            estimated_cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
        if "trace_events" not in state or state["trace_events"] is None:
            state["trace_events"] = []
        state["trace_events"].append(event)

        if self.trace_callback:
            try:
                self.trace_callback(event)
            except Exception:
                pass
        return event

    def _planner_node(self, state: AegisState) -> dict:
        t0 = time.perf_counter()
        query = state["query"]
        has_dataset = state.get("has_dataset", False)
        critic = state.get("critic_evaluation")
        critic_feedback = critic.guidance if (critic and critic.guidance) else None

        self._log_trace(
            state=state,
            agent_role=AgentRole.PLANNER,
            event_type="agent_start",
            message=f"Planner Agent decomposing engineering request: '{query[:80]}...'",
        )

        plan = self.planner_agent.execute(
            query=query,
            has_dataset=has_dataset,
            dataset_info={"name": state.get("dataset_name")} if has_dataset else None,
            critic_feedback=critic_feedback,
        )

        struct_res = getattr(self.planner_agent, "last_structured_result", None)
        in_tokens = struct_res.usage.input_tokens if struct_res else 0
        out_tokens = struct_res.usage.output_tokens if struct_res else 0
        tokens_total = in_tokens + out_tokens
        model_name = struct_res.model if struct_res else None
        latency_ms = struct_res.latency_ms if struct_res else round((time.perf_counter() - t0) * 1000.0, 2)
        cost_usd = struct_res.cost_usd if struct_res else None

        self._log_trace(
            state=state,
            agent_role=AgentRole.PLANNER,
            event_type="agent_finish",
            message=f"Planner generated structured plan with {len(plan.subtasks)} subtasks.",
            details={"subtasks": [st.title for st in plan.subtasks]},
            duration_ms=latency_ms,
            token_usage=tokens_total,
            cost_usd=cost_usd,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model=model_name,
        )

        # Replan clears specialist results; normal run initializes empty dict
        is_replan = critic is not None and critic.replan
        existing_results = state.get("specialist_results")
        if is_replan or existing_results is None:
            spec_dict = {}
        elif isinstance(existing_results, dict):
            spec_dict = dict(existing_results)
        else:
            spec_dict = {r.subtask_id: r for r in existing_results}

        return {
            "plan": plan,
            "current_subtask_index": 0,
            "pending_retry_subtasks": None,
            "specialist_results": spec_dict,
            "status": TaskStatus.EXECUTING,
            "trace_events": state.get("trace_events", []),
        }

    def _specialist_node(self, state: AegisState) -> dict:
        t0 = time.perf_counter()
        plan = state["plan"]
        idx = state.get("current_subtask_index", 0)

        # Normalize specialist_results to dict
        existing_results = state.get("specialist_results")
        if isinstance(existing_results, dict):
            specialist_dict = dict(existing_results)
        elif isinstance(existing_results, list):
            specialist_dict = {r.subtask_id: r for r in existing_results}
        else:
            specialist_dict = {}

        pending_retries = state.get("pending_retry_subtasks")
        is_retry = pending_retries is not None and len(pending_retries) > 0

        if is_retry:
            target_id = pending_retries[0]
            subtask = next((st for st in plan.subtasks if st.id == target_id), None)
            if subtask is None:
                return {"pending_retry_subtasks": pending_retries[1:]}
        else:
            if not plan or idx >= len(plan.subtasks):
                return {}
            subtask = plan.subtasks[idx]

        agent_role = subtask.assigned_agent

        self._log_trace(
            state=state,
            agent_role=agent_role,
            event_type="agent_start",
            message=f"Dispatched {agent_role.value.capitalize()} Agent to execute: '{subtask.title or subtask.id}'",
            details={"required_tools": subtask.required_tools},
        )

        # Prepare context including upstream outputs and retry guidance
        upstream_parts: list[str] = []
        for dep_id in subtask.depends_on:
            past_res = specialist_dict.get(dep_id)
            if past_res:
                finding_lines = "\n".join(f"- {f.statement}" for f in past_res.findings)
                upstream_parts.append(
                    f"Subtask '{dep_id}' output:\nSummary: {past_res.output_summary}\nFindings:\n{finding_lines}"
                )

        critic_guidance = state.get("critic_guidance", "") if is_retry else ""
        context = {
            "content": state.get("dataset_content"),
            "upstream_context": "\n\n".join(upstream_parts),
            "critic_guidance": critic_guidance,
        }

        # Dynamic routing to specialist agent
        if agent_role == AgentRole.RESEARCH:
            result = self.research_agent.execute(subtask=subtask, context=context)
        elif agent_role == AgentRole.DATA_ANALYST:
            result = self.data_analyst_agent.execute(subtask=subtask, context=context)
        else:
            result = self.research_agent.execute(subtask=subtask, context=context)

        # Log tool calls
        for tc in result.tool_calls:
            self._log_trace(
                state=state,
                agent_role=agent_role,
                event_type="tool_call",
                message=f"{agent_role.value.capitalize()} invoked tool '{tc.tool_name}'",
                details={"arguments": tc.arguments, "success": tc.success, "error": tc.error},
                duration_ms=tc.execution_time_ms,
            )

        subtask.status = "completed"
        subtask.output = result.output_summary
        subtask.tool_calls = result.tool_calls
        specialist_dict[subtask.id] = result

        duration = (time.perf_counter() - t0) * 1000.0
        in_tok = result.usage.input_tokens if result.usage else 0
        out_tok = result.usage.output_tokens if result.usage else 0
        tokens_total = in_tok + out_tok
        cost = result.metadata.get("cost_usd") if result.metadata else None
        model_name = result.metadata.get("model") if result.metadata else None

        self._log_trace(
            state=state,
            agent_role=agent_role,
            event_type="agent_finish",
            message=f"{agent_role.value.capitalize()} Agent completed: {result.output_summary[:100]}...",
            details={"findings_count": len(result.findings)},
            duration_ms=duration,
            token_usage=tokens_total,
            cost_usd=cost,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model_name,
        )

        if is_retry:
            return {
                "specialist_results": specialist_dict,
                "pending_retry_subtasks": pending_retries[1:],
                "trace_events": state.get("trace_events", []),
            }
        else:
            return {
                "specialist_results": specialist_dict,
                "current_subtask_index": idx + 1,
                "trace_events": state.get("trace_events", []),
            }

    def _verification_node(self, state: AegisState) -> dict:
        t0 = time.perf_counter()
        self._log_trace(
            state=state,
            agent_role=AgentRole.VERIFICATION,
            event_type="verify",
            message="Verification Agent auditing technical claims and cross-referencing RAG standards...",
        )

        results = state.get("specialist_results", {})
        v_res = self.verification_agent.execute(specialist_results=results, original_query=state.get("query", ""))

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=AgentRole.VERIFICATION,
            event_type="verify",
            message=(
                f"Verification completed. Supported: {v_res.metrics.supported}/{len(v_res.claims)}, "
                f"Contradictions: {v_res.metrics.contradicted}, Discrepancies: {v_res.metrics.numeric_mismatches}"
            ),
            details=v_res.metrics.model_dump(),
            duration_ms=duration,
            token_usage=0,
        )

        return {
            "verification_result": v_res,
            "status": TaskStatus.VERIFYING,
            "trace_events": state.get("trace_events", []),
        }

    def _critic_node(self, state: AegisState) -> dict:
        t0 = time.perf_counter()
        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="critic",
            message="Critic Agent reviewing findings for contradictions, evidence gaps, and reasoning rigor...",
        )

        results = state.get("specialist_results", {})
        v_res = state.get("verification_result")
        retry_count = state.get("retry_count", 0)
        from aegis.config import get_settings

        cfg = get_settings()
        max_retries = state.get("max_retries", getattr(cfg, "max_agent_retries", 2))

        critic_eval = self.critic_agent.execute(
            specialist_results=results,
            verification_result=v_res,
            current_retry_count=retry_count,
            max_retries=max_retries,
        )

        duration = (time.perf_counter() - t0) * 1000.0
        verdict = "PASSED" if critic_eval.quality_pass else "NEEDS REVISION"
        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="critic",
            message=f"Critic peer review result: {verdict} (Score: {critic_eval.score}/10). {critic_eval.critique_summary[:90]}",
            details=critic_eval.model_dump(),
            duration_ms=duration,
            token_usage=0,
        )

        return {
            "critic_evaluation": critic_eval,
            "status": TaskStatus.CRITIQUING,
            "trace_events": state.get("trace_events", []),
        }

    def _retry_node(self, state: AegisState) -> dict:
        retry_count = state.get("retry_count", 0) + 1
        critic = state.get("critic_evaluation")
        targeted_subtasks = list(critic.retry_subtask_ids) if critic and critic.retry_subtask_ids else []
        guidance = critic.guidance if critic else ""

        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="retry",
            message=f"Initiating Targeted Retry #{retry_count} for subtasks: {targeted_subtasks}. Guidance: {guidance}",
            details={"retry_count": retry_count, "targeted_subtasks": targeted_subtasks},
            duration_ms=10.0,
        )

        return {
            "status": TaskStatus.RETRYING,
            "retry_count": retry_count,
            "pending_retry_subtasks": targeted_subtasks,
            "critic_guidance": guidance,
            "trace_events": state.get("trace_events", []),
        }

    def _synthesizer_node(self, state: AegisState) -> dict:
        t0 = time.perf_counter()
        self._log_trace(
            state=state,
            agent_role=AgentRole.SYNTHESIZER,
            event_type="agent_start",
            message="Synthesizer Agent assembling verified telemetry, evidence, and recommendations into final report...",
        )

        from aegis.config import get_settings

        cfg = get_settings()
        max_retries = state.get("max_retries", getattr(cfg, "max_agent_retries", 2))

        report = self.synthesizer_agent.execute(
            task_id=state["task_id"],
            query=state["query"],
            specialist_results=state.get("specialist_results", {}),
            verification_result=state.get("verification_result"),
            critic_evaluation=state.get("critic_evaluation"),
            trace_events=state.get("trace_events", []),
            retries_count=state.get("retry_count", 0),
            max_retries=max_retries,
        )

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=AgentRole.SYNTHESIZER,
            event_type="agent_finish",
            message=f"Final Report compiled successfully: '{report.title}'.",
            details={"title": report.title},
            duration_ms=duration,
            token_usage=0,
        )

        return {
            "final_report": report,
            "status": TaskStatus.COMPLETED,
            "trace_events": state.get("trace_events", []),
        }

    def _human_gate_node(self, state: AegisState) -> dict:
        report = state.get("final_report")
        conf = report.executive_summary.overall_confidence if report else None
        actions = list(report.executive_summary.actionable_recommendations) if report else []

        approval_req = HumanApprovalRequest(
            task_id=state["task_id"],
            required=True,
            status="pending",
            reason="Human approval required before finalizing actions.",
            risk_level="high" if (conf is not None and conf < 0.75) else "medium",
            proposed_actions=actions,
        )

        self._log_trace(
            state=state,
            agent_role=None,
            event_type="human_gate",
            message="Workflow paused at Human Review Gate awaiting operator sign-off.",
            details=approval_req.model_dump(),
        )
        return {
            "human_approval": approval_req,
            "status": TaskStatus.WAITING_APPROVAL,
            "trace_events": state.get("trace_events", []),
        }

    def _build_graph(self) -> Any:
        workflow = StateGraph(AegisState)

        # Add Nodes
        workflow.add_node("planner_node", self._planner_node)
        workflow.add_node("specialist_node", self._specialist_node)
        workflow.add_node("verification_node", self._verification_node)
        workflow.add_node("critic_node", self._critic_node)
        workflow.add_node("retry_node", self._retry_node)
        workflow.add_node("synthesizer_node", self._synthesizer_node)
        workflow.add_node("human_gate_node", self._human_gate_node)

        # Add Edges & Conditional Edges
        workflow.add_edge(START, "planner_node")
        workflow.add_edge("planner_node", "specialist_node")

        # Specialist dispatch loop
        workflow.add_conditional_edges(
            "specialist_node",
            route_next_subtask,
            {
                "specialist_node": "specialist_node",
                "verification_node": "verification_node",
            },
        )

        workflow.add_edge("verification_node", "critic_node")

        # Critic quality pass or retry loop
        workflow.add_conditional_edges(
            "critic_node",
            evaluate_quality_pass,
            {
                "synthesizer_node": "synthesizer_node",
                "retry_node": "retry_node",
                "planner_node": "planner_node",
            },
        )

        workflow.add_edge("retry_node", "specialist_node")

        # Synthesizer to Human Review Gate or Complete
        workflow.add_conditional_edges(
            "synthesizer_node",
            route_human_gate_or_complete,
            {
                "human_gate_node": "human_gate_node",
                "__end__": END,
            },
        )

        workflow.add_edge("human_gate_node", END)

        return workflow.compile()

    def run(self, initial_state: AegisState) -> AegisState:
        """Executes the full compiled LangGraph workflow."""
        try:
            result = self.graph.invoke(initial_state)
            return result
        finally:
            if hasattr(self, "task_context") and self.task_context is not None:
                try:
                    self.task_context.cleanup()
                except Exception:
                    pass
