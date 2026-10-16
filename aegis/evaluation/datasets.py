"""Evaluation benchmark tasks and ground-truth standards."""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    id: str
    title: str
    query: str
    expected_findings: List[str]
    has_dataset: bool = True
    domain: str = "Industrial Engineering"
    complexity: str = "High"


BENCHMARK_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        id="bench-pdm-01",
        title="Predictive Maintenance & Bearing Life for Slurry Pumps",
        query=(
            "Analyse whether predictive maintenance would be suitable for this industrial equipment. "
            "Use the uploaded sensor data and relevant research, identify technical feasibility, risks, "
            "and provide an implementation recommendation."
        ),
        expected_findings=[
            "ISO 10816 Zone D critical threshold breach (> 4.5 mm/s)",
            "Thermal-vibration correlation exceeding 0.95",
            "Remaining Useful Life under 7 operational days",
            "35% to 50% downtime reduction ROI",
        ],
        has_dataset=True,
        domain="Rotating Turbomachinery",
        complexity="High",
    ),
    BenchmarkTask(
        id="bench-thermal-02",
        title="High-Density Compute Cluster Thermal Runaway & Auto-Throttle Logic",
        query=(
            "Evaluate automated dynamic frequency scaling and coolant loop thresholds for a 48U high-density compute rack. "
            "Assess sensor telemetry against ASHRAE thermal guidelines and provide failure prevention logic."
        ),
        expected_findings=[
            "ASHRAE A1 environmental thermal threshold violation (> 27°C intake)",
            "Exponential fan power scaling above 80% duty cycle",
            "Dynamic P-state downclocking recommendation",
        ],
        has_dataset=True,
        domain="Datacenter Infrastructure",
        complexity="Medium",
    ),
    BenchmarkTask(
        id="bench-turbine-03",
        title="Offshore Wind Turbine Blade Structural Delamination & Anomaly Detection",
        query=(
            "Investigate acoustic emission sensor telemetry from a 6MW offshore wind turbine. "
            "Determine whether blade root micro-cracking is present and benchmark against IEC 61400-23 standards."
        ),
        expected_findings=[
            "IEC 61400-23 structural blade safety criteria",
            "Acoustic burst count escalation at pitch angle > 12 degrees",
            "Aeroelastic flutter risk under gust conditions",
        ],
        has_dataset=True,
        domain="Renewable Energy",
        complexity="High",
    ),
]
