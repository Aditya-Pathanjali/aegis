"""Critic Agent for peer review, error detection, and retry governance."""

from __future__ import annotations

from typing import List

from aegis.agents.base import BaseAgent
from aegis.models.schemas import (
    AgentExecutionResult,
    AgentRole,
    CriticEvaluation,
    VerificationResult,
)


class CriticAgent(BaseAgent):
    """Critic agent placeholder (PR 6 implementation pending)."""

    def __init__(self, llm_provider=None, tool_registry=None):
        super().__init__(role=AgentRole.CRITIC, llm_provider=llm_provider, tool_registry=tool_registry)

    def execute(
        self,
        specialist_results: List[AgentExecutionResult],
        verification_result: VerificationResult,
        current_retry_count: int = 0,
        max_retries: int = 2,
    ) -> CriticEvaluation:
        return CriticEvaluation(
            quality_pass=True,
            score=0.0,
            issues_found=[],
            missing_evidence=[],
            weak_reasoning=[],
            contradictions=[],
            critique_summary="Critic not implemented yet (PR 6)",
            retry_recommendations=None,
            target_subtask_for_retry=None,
        )
