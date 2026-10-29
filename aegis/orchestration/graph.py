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
from aegis.models.llm_provider import BaseLLMProvider, get_llm_provider
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
        llm_provider: Optional[BaseLLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        trace_callback: Optional[Callable[[TraceEvent], None]] = None,
    ):
        self.llm = llm_provider or get_llm_provider()
        self.tool_registry = tool_registry or get_tool_registry()
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
        token_usage: int = 150,
        cost_usd: float = 0.0005,
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

    def _planner_node(self, state: AegisState) -> AegisState:
        t0 = time.perf_counter()
        query = state["query"]
        has_dataset = state.get("has_dataset", False)
        critic_feedback = state.get("critic_evaluation").retry_recommendations if state.get("critic_evaluation") else None

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

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=AgentRole.PLANNER,
            event_type="agent_finish",
            message=f"Planner generated structured plan with {len(plan.subtasks)} subtasks.",
            details={"subtasks": [st.title for st in plan.subtasks]},
            duration_ms=duration,
            token_usage=450,
        )

        state["plan"] = plan
        state["current_subtask_index"] = 0
        state["status"] = TaskStatus.EXECUTING
        if "specialist_results" not in state or state["specialist_results"] is None:
            state["specialist_results"] = []
        return state

    def _specialist_node(self, state: AegisState) -> AegisState:
        t0 = time.perf_counter()
        plan = state["plan"]
        idx = state.get("current_subtask_index", 0)

        if not plan or idx >= len(plan.subtasks):
            return state

        subtask = plan.subtasks[idx]
        agent_role = subtask.assigned_agent

        self._log_trace(
            state=state,
            agent_role=agent_role,
            event_type="agent_start",
            message=f"Dispatched {agent_role.value.capitalize()} Agent to execute: '{subtask.title}'",
            details={"required_tools": subtask.required_tools},
        )

        # Dynamic routing to specialist agent
        if agent_role == AgentRole.RESEARCH:
            result = self.research_agent.execute(subtask=subtask)
        elif agent_role == AgentRole.DATA_ANALYST:
            result = self.data_analyst_agent.execute(subtask=subtask, context={"content": state.get("dataset_content")})
        else:
            # Fallback to research
            result = self.research_agent.execute(subtask=subtask)

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
        state["specialist_results"].append(result)

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=agent_role,
            event_type="agent_finish",
            message=f"{agent_role.value.capitalize()} Agent completed: {result.output_summary[:100]}...",
            details={"findings_count": len(result.detailed_findings)},
            duration_ms=duration,
            token_usage=650,
        )

        state["current_subtask_index"] = idx + 1
        return state

    def _verification_node(self, state: AegisState) -> AegisState:
        t0 = time.perf_counter()
        state["status"] = TaskStatus.VERIFYING
        self._log_trace(
            state=state,
            agent_role=AgentRole.VERIFICATION,
            event_type="verify",
            message="Verification Agent auditing technical claims and cross-referencing RAG standards...",
        )

        results = state.get("specialist_results", [])
        v_res = self.verification_agent.execute(specialist_results=results, original_query=state["query"])

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=AgentRole.VERIFICATION,
            event_type="verify",
            message=(
                f"Verification completed. Reliability: {v_res.metrics.overall_reliability}%, "
                f"Evidence Coverage: {v_res.metrics.evidence_coverage}%, Hallucination Risk: {v_res.metrics.hallucination_risk}%"
            ),
            details=v_res.metrics.model_dump(),
            duration_ms=duration,
            token_usage=520,
        )

        state["verification_result"] = v_res
        return state

    def _critic_node(self, state: AegisState) -> AegisState:
        t0 = time.perf_counter()
        state["status"] = TaskStatus.CRITIQUING
        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="critic",
            message="Critic Agent reviewing findings for contradictions, evidence gaps, and reasoning rigor...",
        )

        results = state.get("specialist_results", [])
        v_res = state.get("verification_result")
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 2)

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
            token_usage=400,
        )

        state["critic_evaluation"] = critic_eval
        return state

    def _retry_node(self, state: AegisState) -> AegisState:
        state["status"] = TaskStatus.RETRYING
        state["retry_count"] = state.get("retry_count", 0) + 1
        critic = state.get("critic_evaluation")

        self._log_trace(
            state=state,
            agent_role=AgentRole.CRITIC,
            event_type="retry",
            message=f"Initiating Retry Cycle #{state['retry_count']}. Guidance: {critic.retry_recommendations if critic else 'Recalibrate'}",
            details={"retry_count": state["retry_count"]},
            duration_ms=10.0,
        )

        # Reset index to re-execute targeted subtasks
        state["current_subtask_index"] = 0
        return state

    def _synthesizer_node(self, state: AegisState) -> AegisState:
        t0 = time.perf_counter()
        state["status"] = TaskStatus.SYNTHESIZING
        self._log_trace(
            state=state,
            agent_role=AgentRole.SYNTHESIZER,
            event_type="agent_start",
            message="Synthesizer Agent assembling verified telemetry, evidence, and recommendations into final report...",
        )

        report = self.synthesizer_agent.execute(
            task_id=state["task_id"],
            query=state["query"],
            specialist_results=state.get("specialist_results", []),
            verification_result=state.get("verification_result"),
            critic_evaluation=state.get("critic_evaluation"),
            trace_events=state.get("trace_events", []),
            retries_count=state.get("retry_count", 0),
        )

        duration = (time.perf_counter() - t0) * 1000.0
        self._log_trace(
            state=state,
            agent_role=AgentRole.SYNTHESIZER,
            event_type="agent_finish",
            message=f"Final Report compiled successfully. Executive confidence: {report.executive_summary.overall_confidence * 100:.1f}%.",
            details={"title": report.title},
            duration_ms=duration,
            token_usage=800,
        )

        state["final_report"] = report
        state["status"] = TaskStatus.COMPLETED
        return state

    def _human_gate_node(self, state: AegisState) -> AegisState:
        report = state.get("final_report")
        conf = report.executive_summary.overall_confidence if report else 0.88

        # Create Human Approval Request
        approval_req = HumanApprovalRequest(
            task_id=state["task_id"],
            required=True,
            status="pending",
            reason=(
                "Mandatory operational sign-off: Predictive maintenance recommendation advises "
                "equipment throttling and bearing overhaul within 72 hours."
            ),
            risk_level="high" if conf < 0.75 else "medium",
            proposed_actions=[
                "Approve scheduled shutdown & bearing overhaul window",
                "Approve maintenance procurement budget",
                "Authorize automated SCADA trip limits",
            ],
        )

        state["human_approval"] = approval_req
        state["status"] = TaskStatus.WAITING_APPROVAL

        self._log_trace(
            state=state,
            agent_role=None,
            event_type="human_gate",
            message="Workflow paused at Human Review Gate awaiting operator sign-off.",
            details=approval_req.model_dump(),
        )
        return state

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
        result = self.graph.invoke(initial_state)
        return result
