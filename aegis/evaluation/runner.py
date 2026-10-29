"""Benchmark execution runner (Phase 3)."""

from __future__ import annotations

from typing import List, Optional

from aegis.evaluation.metrics import BenchmarkComparisonReport


class EvaluationRunner:
    """Benchmark runner placeholder for Phase 3."""

    def __init__(self, llm_provider=None):
        self.llm = llm_provider

    def run_benchmark(self, task_subset: Optional[List[str]] = None) -> BenchmarkComparisonReport:
        raise NotImplementedError("Phase 3")
