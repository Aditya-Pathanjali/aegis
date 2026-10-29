"""Evaluation benchmark tasks (Phase 3)."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class BenchmarkTask(BaseModel):
    id: str
    title: str
    query: str
    expected_findings: List[str]
    has_dataset: bool = True
    domain: str = "Industrial Engineering"
    complexity: str = "High"


BENCHMARK_TASKS: List[BenchmarkTask] = []
