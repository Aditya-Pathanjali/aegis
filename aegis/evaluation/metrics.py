"""Evaluation benchmark metrics schemas."""

from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field


class ParadigmMetric(BaseModel):
    paradigm_name: str  # "Single Agent", "Multi-Agent", "Multi-Agent + Verification"
    task_success_rate: float = Field(ge=0.0, le=100.0)  # %
    answer_accuracy: float = Field(ge=0.0, le=100.0)  # %
    evidence_coverage: float = Field(ge=0.0, le=100.0)  # %
    hallucination_rate: float = Field(ge=0.0, le=100.0)  # %
    verification_accuracy: float = Field(ge=0.0, le=100.0)  # %
    average_latency_sec: float  # seconds
    token_api_cost_usd: float  # USD
    retries_count: int
    human_intervention_rate: float = Field(ge=0.0, le=100.0)  # %
    description: str = ""


class BenchmarkComparisonReport(BaseModel):
    benchmark_id: str
    run_timestamp: datetime = Field(default_factory=datetime.utcnow)
    tasks_evaluated_count: int = 3
    results: List[ParadigmMetric]
    comparative_summary: str
    key_findings: List[str]
