"""Evaluation service caching and running architecture benchmarks."""

from __future__ import annotations

from typing import Optional

from aegis.evaluation.metrics import BenchmarkComparisonReport
from aegis.evaluation.runner import EvaluationRunner


class EvaluationService:
    """Manages benchmark runs and maintains benchmark result cache."""

    def __init__(self):
        self.runner = EvaluationRunner()
        self._latest_report: Optional[BenchmarkComparisonReport] = None

    def run_benchmark(self) -> BenchmarkComparisonReport:
        report = self.runner.run_benchmark()
        self._latest_report = report
        return report

    def get_latest_report(self) -> BenchmarkComparisonReport:
        if self._latest_report is None:
            self._latest_report = self.run_benchmark()
        return self._latest_report


_global_eval_service: Optional[EvaluationService] = None


def get_evaluation_service() -> EvaluationService:
    global _global_eval_service
    if _global_eval_service is None:
        _global_eval_service = EvaluationService()
    return _global_eval_service
