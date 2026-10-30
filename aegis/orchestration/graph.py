"""Aegis Multi-Agent StateGraph Orchestration Engine using LangGraph."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Dict, Optional, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from aegis.agents.critic import CriticAgent
from aegis.agents.data_analyst import DataAnalystAgent
from aegis.agents.planner import PlannerAgent
from aegis.agents.research import ResearchAgent
from aegis.agents.synthesizer import SynthesizerAgent
from aegis.agents.verification import VerificationAgent
from aegis.llm import LLMProvider, get_llm_provider
from aegis.llm.pricing import calculate_cost
from aegis.models.schemas import (
    AgentRole,
    HumanApprovalRequest,
    TaskStatus,
    TraceEvent,
)
from aegis.orchestration.router import (
    evaluate_quality_pass,
    route_after_human_gate,
    route_next_subtask,
)
from aegis.orchestration.state import AegisState
from aegis.tools.registry import ToolRegistry, get_tool_registry

logger = logging.getLogger(__name__)


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
        checkpointer: Optional[Any] = None,
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
        self.checkpointer = checkpointer or MemorySaver()

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
        cost_usd: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        call_count: int = 0,
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
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            call_count=call_count,
            model=model,
        )
        if "trace_events" not in state or state["trace_events"] is None:
            state["trace_events"] = []
        state["trace_events"].append(event)

        if self.trace_callback:
            try:
                self.trace_callback(event)
            except Exception as exc:
                logger.exception("Error in trace_callback: %s", exc)
                raise
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
        cost_usd = (
            struct_res.cost_usd
            if (struct_res and struct_res.cost_usd is not None)
            else (calculate_cost(model_name, in_tokens, out_tokens) if model_name else None)
        )
        call_count = 1 if struct_res else 0

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
            call_count=call_count,
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
        call_count = result.metadata.get("call_count", 1) if result.metadata else 1

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
            call_count=call_count,
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
        v_in = v_res.usage.input_tokens if v_res.usage else 0
        v_out = v_res.usage.output_tokens if v_res.usage else 0
        v_calls = v_res.metadata.get("call_count", 0) if v_res.metadata else 0
        v_cost = v_res.metadata.get("cost_usd") if v_res.metadata else None
        v_model = v_res.metadata.get("model") if v_res.metadata else None

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
            token_usage=v_in + v_out,
            cost_usd=v_cost,
            input_tokens=v_in,
            output_tokens=v_out,
            call_count=v_calls,
            model=v_model,
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
        c_in = critic_eval.usage.input_tokens if critic_eval.usage else 0
        c_out = critic_eval.usage.output_tokens if critic_eval.usage else 0
        c_calls = critic_eval.metadata.get("call_count", 1) if critic_eval.metadata else 1
        c_cost = critic_eval.metadata.get("cost_usd") if critic_eval.metadata else None
        c_model = critic_eval.metadata.get("model") if critic_eval.metadata else None

        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="critic",
            message=f"Critic peer review result: {verdict} (Score: {critic_eval.score}/10). {critic_eval.critique_summary[:90]}",
            details=critic_eval.model_dump(),
            duration_ms=duration,
            token_usage=c_in + c_out,
            cost_usd=c_cost,
            input_tokens=c_in,
            output_tokens=c_out,
            call_count=c_calls,
            model=c_model,
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
        s_res = getattr(self.synthesizer_agent, "last_structured_result", None)
        s_in = s_res.usage.input_tokens if s_res else 0
        s_out = s_res.usage.output_tokens if s_res else 0
        s_calls = 1 if s_res else 0
        s_model = s_res.model if s_res else None
        s_cost = calculate_cost(s_model, s_in, s_out) if s_model else None

        synth_event = self._log_trace(
            state=state,
            agent_role=AgentRole.SYNTHESIZER,
            event_type="agent_finish",
            message=f"Final Report compiled successfully: '{report.title}'.",
            details={"title": report.title},
            duration_ms=duration,
            token_usage=s_in + s_out,
            cost_usd=s_cost,
            input_tokens=s_in,
            output_tokens=s_out,
            call_count=s_calls,
            model=s_model,
        )
        if synth_event not in report.agent_execution_trace:
            report.agent_execution_trace.append(synth_event)

        return {
            "final_report": report,
            "status": TaskStatus.EXECUTING,
            "trace_events": state.get("trace_events", []),
        }

    def _human_gate_node(self, state: AegisState) -> dict:
        req_approval = state.get("require_human_approval", False)
        critic = state.get("critic_evaluation")
        critic_failed = critic is not None and not critic.quality_pass
        v_res = state.get("verification_result")
        contradictions = v_res.metrics.contradicted if (v_res and v_res.metrics) else 0
        supp_ratio = v_res.metrics.supported_ratio if (v_res and v_res.metrics) else None
        v_res_failed = v_res is not None and not v_res.metrics.quality_pass

        is_triggered = (
            req_approval
            or critic_failed
            or v_res_failed
            or (contradictions > 0)
            or (supp_ratio is not None and supp_ratio < 0.8)
        )

        if not is_triggered:
            return {
                "status": TaskStatus.COMPLETED,
                "trace_events": state.get("trace_events", []),
            }

        reasons = []
        if req_approval:
            reasons.append("Task configuration requires human approval")
        if critic_failed:
            reasons.append("Critic quality review did not pass")
        if v_res_failed:
            reasons.append("Quality failure detected (no verified findings): manual review required")
        if contradictions > 0:
            reasons.append(f"Verification detected {contradictions} contradiction(s)")
        if supp_ratio is not None and supp_ratio < 0.8:
            reasons.append(f"Verification supported ratio is {supp_ratio:.2f} (< 0.80)")

        reason_str = "; ".join(reasons)

        if contradictions > 0 or critic_failed or v_res_failed or (supp_ratio is not None and supp_ratio < 0.5):
            risk_level = "high"
        elif supp_ratio is not None and supp_ratio < 0.8:
            risk_level = "medium"
        else:
            risk_level = "low"

        report = state.get("final_report")
        summary_text = (
            report.title
            if report
            else f"Human approval required for task {state.get('task_id')}"
        )
        proposed_actions = (
            list(report.executive_summary.actionable_recommendations)
            if (report and report.executive_summary)
            else []
        )

        approval_req = HumanApprovalRequest(
            task_id=state["task_id"],
            required=True,
            status="pending",
            reason=reason_str,
            summary=summary_text,
            risk_level=risk_level,
            proposed_actions=proposed_actions,
            required_action="Review recommendations and approve, reject, or request reanalysis with feedback.",
        )

        self._log_trace(
            state=state,
            agent_role=None,
            event_type="human_gate",
            message=f"Workflow paused at Human Review Gate: {reason_str}",
            details=approval_req.model_dump(),
        )

        try:
            decision = interrupt(approval_req.model_dump())
        except RuntimeError:
            # Invoked outside a LangGraph Runnable stream (e.g. unit testing node directly)
            return {
                "human_approval": approval_req,
                "status": TaskStatus.WAITING_APPROVAL,
                "trace_events": state.get("trace_events", []),
            }

        if isinstance(decision, str):
            action = decision
            feedback = None
        elif isinstance(decision, dict):
            action = decision.get("action", "approve")
            feedback = decision.get("feedback")
        else:
            action = getattr(decision, "action", "approve")
            feedback = getattr(decision, "feedback", None)

        if action == "approve":
            approval_req.status = "approved"
            approval_req.feedback = feedback
            approval_req.decided_at = datetime.now(UTC)
            self._log_trace(
                state=state,
                agent_role=None,
                event_type="human_gate_resume",
                message=f"Human gate approved. Feedback: {feedback or 'None'}",
                details={"action": "approve", "feedback": feedback},
            )
            return {
                "human_approval": approval_req,
                "status": TaskStatus.COMPLETED,
                "trace_events": state.get("trace_events", []),
            }
        elif action == "reject":
            approval_req.status = "rejected"
            approval_req.feedback = feedback
            approval_req.decided_at = datetime.now(UTC)
            self._log_trace(
                state=state,
                agent_role=None,
                event_type="human_gate_resume",
                message=f"Human gate rejected. Feedback: {feedback or 'None'}",
                details={"action": "reject", "feedback": feedback},
            )
            return {
                "human_approval": approval_req,
                "status": TaskStatus.REJECTED,
                "trace_events": state.get("trace_events", []),
            }
        elif action == "request_reanalysis":
            approval_req.status = "revision_requested"
            approval_req.feedback = feedback
            approval_req.decided_at = datetime.now(UTC)
            new_retry_count = state.get("retry_count", 0) + 1
            self._log_trace(
                state=state,
                agent_role=None,
                event_type="human_gate_resume",
                message=f"Human gate requested reanalysis: {feedback or 'No feedback'}",
                details={"action": "request_reanalysis", "feedback": feedback, "retry_count": new_retry_count},
            )
            return {
                "human_approval": approval_req,
                "status": TaskStatus.RETRYING,
                "retry_count": new_retry_count,
                "critic_guidance": feedback or "",
                "specialist_results": {},
                "trace_events": state.get("trace_events", []),
            }
        else:
            return {
                "human_approval": approval_req,
                "status": TaskStatus.COMPLETED,
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

        # Synthesizer to Human Review Gate
        workflow.add_edge("synthesizer_node", "human_gate_node")

        # Human Review Gate routes to planner on reanalysis, else finishes
        workflow.add_conditional_edges(
            "human_gate_node",
            route_after_human_gate,
            {
                "planner_node": "planner_node",
                "__end__": END,
            },
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def run(
        self,
        initial_state: AegisState,
        on_node_finish: Optional[Callable[[AegisState], None]] = None,
    ) -> AegisState:
        """Executes the full compiled LangGraph workflow with streaming and interrupt support."""
        task_id = initial_state.get("task_id", "default")
        config = {"configurable": {"thread_id": task_id}}
        current_state = dict(initial_state)

        for event in self.graph.stream(initial_state, config=config):
            if isinstance(event, dict):
                for node_name, node_update in event.items():
                    if node_name == "__interrupt__":
                        current_state["status"] = TaskStatus.WAITING_APPROVAL
                        for intr in node_update:
                            if hasattr(intr, "value") and isinstance(intr.value, dict):
                                current_state["human_approval"] = HumanApprovalRequest.model_validate(intr.value)
                    elif isinstance(node_update, dict):
                        current_state.update(node_update)
            if on_node_finish:
                try:
                    on_node_finish(current_state)
                except Exception as exc:
                    logger.exception("Error in on_node_finish during run: %s", exc)
                    raise

        return current_state

    def resume(
        self,
        task_id: str,
        decision: Union[str, Dict[str, Any]],
        on_node_finish: Optional[Callable[[AegisState], None]] = None,
    ) -> AegisState:
        """Resumes an interrupted workflow at the human gate."""
        config = {"configurable": {"thread_id": task_id}}
        state_snapshot = self.graph.get_state(config)
        current_state = dict(state_snapshot.values)

        for event in self.graph.stream(Command(resume=decision), config=config):
            if isinstance(event, dict):
                for node_name, node_update in event.items():
                    if node_name == "__interrupt__":
                        current_state["status"] = TaskStatus.WAITING_APPROVAL
                        for intr in node_update:
                            if hasattr(intr, "value") and isinstance(intr.value, dict):
                                current_state["human_approval"] = HumanApprovalRequest.model_validate(intr.value)
                    elif isinstance(node_update, dict):
                        current_state.update(node_update)
            if on_node_finish:
                try:
                    on_node_finish(current_state)
                except Exception as exc:
                    logger.exception("Error in on_node_finish during run: %s", exc)
                    raise

        return current_state
