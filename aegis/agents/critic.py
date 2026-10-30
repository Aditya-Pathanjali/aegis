"""Critic Agent for peer review, error detection, rubric evaluation, and retry governance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Union

from aegis.agents.base import BaseAgent
from aegis.llm.pricing import calculate_cost
from aegis.llm.types import Message, TextBlock
from aegis.models.llm_outputs import CriticRubricOutput
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    CriticEvaluation,
    VerificationResult,
)

if TYPE_CHECKING:
    from aegis.llm.base import LLMProvider
    from aegis.tools.registry import ToolRegistry


class CriticAgent(BaseAgent):
    """
    Evaluates specialist outputs and verification audits using rubric scoring.
    Computes quality_pass deterministically in code from scores and verification pass.
    """

    SYSTEM_PROMPT = (
        "You are the Aegis Senior Review Critic.\n"
        "Evaluate the specialist outputs and verification audit against four criteria (0-10):\n"
        "1. answers_question: Does the work directly and completely answer the engineering request?\n"
        "2. uses_task_data: Are empirical dataset tables, metrics, and RAG standards properly utilized?\n"
        "3. claims_verified: Are claims empirically supported with zero contradictions and zero numeric discrepancies?\n"
        "4. limitations_stated: Are assumptions, data gaps, and operational limitations transparently documented?\n\n"
        "If specific subtasks need re-running with better parameters or queries, list their IDs in retry_subtask_ids.\n"
        "If the plan decomposition itself is flawed, set replan=True.\n"
        "Provide actionable, precise guidance in the guidance field."
    )

    def __init__(self, llm_provider: Optional[LLMProvider] = None, tool_registry: Optional[ToolRegistry] = None):
        super().__init__(role=AgentRole.CRITIC, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(
        self,
        specialist_results: Union[List[AgentExecutionResult], Dict[str, AgentExecutionResult]],
        verification_result: VerificationResult,
        current_retry_count: int = 0,
        max_retries: int = 2,
    ) -> CriticEvaluation:
        results_list: List[AgentExecutionResult] = (
            list(specialist_results.values())
            if isinstance(specialist_results, dict)
            else list(specialist_results)
        )

        # Build context for Critic LLM
        summary_lines = []
        for r in results_list:
            findings_summary = "; ".join(f.statement for f in r.findings)
            summary_lines.append(
                f"- Subtask [{r.subtask_id}] ({r.agent_role.value}): {r.output_summary}\n"
                f"  Findings: {findings_summary or 'None'}\n"
                f"  Limitations: {'; '.join(r.limitations) or 'None'}"
            )

        v_metrics = verification_result.metrics
        unsupported_summary = "; ".join(verification_result.unsupported_claims) or "None"
        ratio_str = f"{v_metrics.supported_ratio:.2f}" if v_metrics.supported_ratio is not None else "N/A"

        prompt = (
            "Specialist Execution Results:\n"
            + "\n".join(summary_lines)
            + f"\n\nVerification Audit Summary:\n"
            f"- Status: {verification_result.status}\n"
            f"- Supported: {v_metrics.supported}/{len(verification_result.claims)} (Ratio: {ratio_str})\n"
            f"- Contradictions: {v_metrics.contradicted}\n"
            f"- Numeric Mismatches: {v_metrics.numeric_mismatches}\n"
            f"- Unsupported Claims: {unsupported_summary}\n"
            f"- Current Retry Count: {current_retry_count}/{max_retries}\n\n"
            "Score each rubric dimension (0-10) and determine if targeted retry or replan is needed."
        )

        resp = self.llm.structured(
            messages=[Message(role="user", content=[TextBlock(text=prompt)])],
            schema=CriticRubricOutput,
            system=self.SYSTEM_PROMPT,
        )
        rubric: CriticRubricOutput = resp.value

        scores = [
            rubric.answers_question,
            rubric.uses_task_data,
            rubric.claims_verified,
            rubric.limitations_stated,
        ]
        avg_score = round(sum(scores) / len(scores), 2)

        # Deterministic computation of quality pass in code:
        # LLM does NOT set quality_pass directly.
        verification_pass = verification_result.metrics.quality_pass
        has_retries = len(rubric.retry_subtask_ids) > 0 or rubric.replan
        has_low_scores = any(s < 6.0 for s in scores) or avg_score < 7.0

        quality_pass = verification_pass and not has_retries and not has_low_scores

        target_subtask = rubric.retry_subtask_ids[0] if rubric.retry_subtask_ids else None
        critic_cost = calculate_cost(resp.model, resp.usage.input_tokens, resp.usage.output_tokens) if resp.model else None
        self.last_structured_result = resp

        return CriticEvaluation(
            quality_pass=quality_pass,
            score=avg_score,
            rubric_scores={
                "answers_question": rubric.answers_question,
                "uses_task_data": rubric.uses_task_data,
                "claims_verified": rubric.claims_verified,
                "limitations_stated": rubric.limitations_stated,
            },
            retry_subtask_ids=rubric.retry_subtask_ids,
            replan=rubric.replan,
            guidance=rubric.guidance,
            issues_found=rubric.issues_found,
            critique_summary=rubric.guidance or ("Quality passed all rubric criteria." if quality_pass else "Revisions required."),
            retry_recommendations=rubric.guidance,
            target_subtask_for_retry=target_subtask,
            usage=resp.usage,
            metadata={
                "call_count": 1,
                "model": resp.model,
                "cost_usd": critic_cost,
            },
        )
