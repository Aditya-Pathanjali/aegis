"""Aegis Evaluation package."""

from aegis.evaluation.datasets import BENCHMARK_TASKS, BenchmarkTask
from aegis.evaluation.metrics import BenchmarkComparisonReport, ParadigmMetric
from aegis.evaluation.runner import EvaluationRunner

__all__ = [
    "BENCHMARK_TASKS",
    "BenchmarkTask",
    "BenchmarkComparisonReport",
    "ParadigmMetric",
    "EvaluationRunner",
]
