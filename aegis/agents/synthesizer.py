"""Synthesizer Agent for compiling verified findings into an executive AI engineering report."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from aegis.agents.base import BaseAgent
from aegis.llm.pricing import calculate_cost
from aegis.llm.types import Message, TextBlock
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    CriticEvaluation,
    ExecutiveSummary,
    FinalReport,
    SynthesizerOutput,
    TraceEvent,
    VerificationResult,
)

if TYPE_CHECKING:
    from aegis.llm.base import LLMProvider
    from aegis.tools.registry import ToolRegistry


class SynthesizerAgent(BaseAgent):
    """
    Synthesizes only verified empirical findings into the FinalReport.
    Excludes unverified claims and places them into report limitations.
    Removes hard-coded templates, overall_confidence, and business_impact constants.
    """

    SYSTEM_PROMPT = (
        "You are the Aegis Lead Systems Synthesizer.\n"
        "Rules:\n"
        "1. Base findings and conclusions STRICTLY and EXCLUSIVELY on the provided VERIFIED findings.\n"
        "2. Do NOT invent, assume, or hallucinate findings that were excluded or marked unverified.\n"
        "3. Provide realistic, tailored engineering recommendations based only on the factual evidence.\n"
        "4. Summarize both literature evidence and data analysis findings clearly."
    )

    def __init__(self, llm_provider: Optional[LLMProvider] = None, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(role=AgentRole.SYNTHESIZER, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(
        self,
        task_id: str,
        query: str,
        specialist_results: Union[List[AgentExecutionResult], Dict[str, AgentExecutionResult]],
        verification_result: VerificationResult,
        critic_evaluation: Optional[CriticEvaluation] = None,
        trace_events: Optional[List[TraceEvent]] = None,
        retries_count: int = 0,
        max_retries: int = 2,
    ) -> FinalReport:
        t0 = time.perf_counter()
        traces = trace_events or []

        results_list: List[AgentExecutionResult] = (
            list(specialist_results.values())
            if isinstance(specialist_results, dict)
            else list(specialist_results)
        )

        # Collect supported claims and unverified/contradicted claims
        supported_claims = [c for c in verification_result.claims if c.status == "supported"]
        excluded_claims = [c for c in verification_result.claims if c.status != "supported"]

        # Limitations collection: specialist limitations + excluded claims
        report_limitations: List[str] = []
        for r in results_list:
            report_limitations.extend(r.limitations)

        for c in excluded_claims:
            report_limitations.append(f"Excluded unverified claim: '{c.claim}' ({c.status})")

        if retries_count >= max_retries and (critic_evaluation and not critic_evaluation.quality_pass):
            report_limitations.append(
                f"Maximum retry limit reached ({retries_count}/{max_retries}). Report synthesized from best available evidence."
            )

        # Prompt with ONLY verified findings
        verified_lines = [f"- {c.claim}" for c in supported_claims]
        prompt = (
            f"Original Engineering Query:\n\"{query}\"\n\n"
            f"VERIFIED FINDINGS (Status: Supported):\n"
            + ("\n".join(verified_lines) if verified_lines else "None - all claims were unverified or lacked evidence.")
            + f"\n\nAudit Notes:\n{verification_result.audit_notes}\n\n"
            "Synthesize these verified findings into the structured SynthesizerOutput schema."
        )

        struct_res = self.llm.structured(
            messages=[Message(role="user", content=[TextBlock(text=prompt)])],
            schema=SynthesizerOutput,
            system=self.SYSTEM_PROMPT,
        )
        self.last_structured_result = struct_res

        output_val = struct_res.value
        synth_in_tokens = struct_res.usage.input_tokens
        synth_out_tokens = struct_res.usage.output_tokens
        synth_cost = calculate_cost(struct_res.model, synth_in_tokens, synth_out_tokens) if struct_res.model else None

        if isinstance(output_val, FinalReport):
            # Direct FinalReport returned by test fake
            output_val.limitations.extend(report_limitations)
            output_val.executive_summary.overall_confidence = None
            output_val.executive_summary.business_impact = ""
            return output_val

        synth_output: SynthesizerOutput = output_val

        exec_summary = ExecutiveSummary(
            final_finding=synth_output.final_finding,
            key_conclusions=synth_output.key_conclusions,
            overall_confidence=None,  # Rule: remove overall_confidence
            business_impact="",  # Rule: remove business_impact
            actionable_recommendations=synth_output.actionable_recommendations,
        )

        evidence_section = {
            "title": "Empirical Evidence",
            "summary": synth_output.evidence_summary,
            "verified_claims": [c.claim for c in supported_claims],
            "total_claims_audited": len(verification_result.claims),
            "supported_count": len(supported_claims),
        }

        data_analysis_section = {
            "title": "Quantitative Data Analysis",
            "summary": synth_output.data_analysis_summary,
        }

        technical_recommendation = {
            "action": synth_output.recommended_action,
            "urgency": synth_output.urgency,
            "operational_steps": synth_output.operational_steps,
            "economic_impact": "",
        }

        total_exec_seconds = round(
            sum(t.duration_ms for t in traces) / 1000.0 if traces else (time.perf_counter() - t0),
            2,
        )
        total_tokens = sum(t.token_usage for t in traces) + (synth_in_tokens + synth_out_tokens)
        has_unpriced = any(
            (t.call_count > 0 or t.token_usage > 0) and t.cost_usd is None
            for t in traces
        )
        if synth_cost is None and (synth_in_tokens > 0 or synth_out_tokens > 0):
            has_unpriced = True

        if has_unpriced:
            total_cost_usd = None
        else:
            cost_sum = sum(t.cost_usd for t in traces if t.cost_usd is not None)
            if synth_cost is not None:
                cost_sum += synth_cost
            total_cost_usd = round(cost_sum, 6)

        return FinalReport(
            task_id=task_id,
            title=synth_output.title or f"Aegis Engineering Assessment: {query[:60]}",
            executive_summary=exec_summary,
            evidence_section=evidence_section,
            data_analysis_section=data_analysis_section,
            technical_recommendation=technical_recommendation,
            verification_section=verification_result,
            critic_evaluation=critic_evaluation,
            agent_execution_trace=traces,
            limitations=report_limitations,
            total_execution_time_seconds=total_exec_seconds,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            retries_count=retries_count,
        )
