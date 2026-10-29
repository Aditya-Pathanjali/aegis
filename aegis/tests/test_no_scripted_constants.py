"""Regression test to guarantee no scripted constants or hard-coded domain values appear in aegis/."""

import os
from pathlib import Path

from aegis.models.schemas import AgentRole, ExecutionPlan, SubTask, TaskStatus
from aegis.orchestration.graph import AegisOrchestrator
from aegis.orchestration.state import AegisState
from tests.fakes import FakeLLM

FORBIDDEN_PATTERNS = [
    "180," + "000",
    "18," + "500",
    "89" + ".4",
    "ISO" + " 10816",
    "ieee" + "xplore",
    "aegis" + "-engine-v1",
]

DOMAIN_TRIPWIRE_PATTERNS = [
    "mm" + "/s",
    "bear" + "ing",
    "down" + "time",
    "Zo" + "ne",
    "procure" + "ment",
    "R" + "OI",
]


def test_no_scripted_constants_in_aegis():
    """Verify that none of the forbidden scripted values or strings appear anywhere in aegis/."""
    aegis_dir = Path(__file__).resolve().parent.parent
    violations = []

    for root, dirs, files in os.walk(aegis_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for file in files:
            if file.endswith((".pyc", ".pyo", ".pyd")):
                continue
            file_path = Path(root) / file
            if file_path.resolve() == Path(__file__).resolve():
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                for line_no, line in enumerate(lines, start=1):
                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern in line:
                            violations.append(
                                f"{file_path.relative_to(aegis_dir)}:{line_no} contains forbidden pattern '{pattern}' -> {line.strip()}"
                            )
            except Exception as e:
                violations.append(f"Could not read {file_path}: {e}")

    assert not violations, "Forbidden scripted strings found in aegis/:\n" + "\n".join(violations)


def test_tripwire_no_domain_content_in_agents_and_orchestration():
    """Verify that aegis/agents/ and aegis/orchestration/ contain no fixed domain text or keywords."""
    aegis_dir = Path(__file__).resolve().parent.parent
    target_dirs = [aegis_dir / "agents", aegis_dir / "orchestration"]
    violations = []

    for target_dir in target_dirs:
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for file in files:
                if file.endswith((".pyc", ".pyo", ".pyd")):
                    continue
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    for line_no, line in enumerate(lines, start=1):
                        for pattern in DOMAIN_TRIPWIRE_PATTERNS:
                            if pattern.lower() in line.lower():
                                violations.append(
                                    f"{file_path.relative_to(aegis_dir)}:{line_no} contains tripwire pattern '{pattern}' -> {line.strip()}"
                                )
                except Exception as e:
                    violations.append(f"Could not read {file_path}: {e}")

    assert not violations, "Domain tripwire patterns found in agents/orchestration:\n" + "\n".join(violations)


def test_graph_outputs_strictly_derived_from_input_and_fake_llm():
    """
    Run the graph twice with different queries using a scripted FakeLLM,
    and assert that neither report contains text that didn't come from the query
    or the scripted responses.
    """
    query_1 = "Alpha query: evaluate turbine hydraulic pressure"
    plan_1 = ExecutionPlan(
        task_id="t-1",
        original_query=query_1,
        objective="Investigate hydraulic pressure sensor readings",
        subtasks=[
            SubTask(
                id="subtask-1",
                title="Review hydraulic pressure documentation",
                description="Look up expected bar pressure bounds",
                assigned_agent=AgentRole.RESEARCH,
            )
        ],
    )
    fake_llm_1 = FakeLLM(responses=[plan_1])
    orch_1 = AegisOrchestrator(llm_provider=fake_llm_1)
    state_1: AegisState = {
        "task_id": "t-1",
        "query": query_1,
        "has_dataset": False,
        "require_human_approval": False,
        "max_retries": 1,
        "retry_count": 0,
    }
    res_1 = orch_1.run(state_1)
    assert res_1["status"] == TaskStatus.COMPLETED
    report_1 = res_1["final_report"]

    query_2 = "Beta query: analyze solar inverter thermal load"
    plan_2 = ExecutionPlan(
        task_id="t-2",
        original_query=query_2,
        objective="Analyze solar inverter thermal thresholds",
        subtasks=[
            SubTask(
                id="subtask-1",
                title="Review solar inverter technical specs",
                description="Look up inverter temperature limits",
                assigned_agent=AgentRole.RESEARCH,
            )
        ],
    )
    fake_llm_2 = FakeLLM(responses=[plan_2])
    orch_2 = AegisOrchestrator(llm_provider=fake_llm_2)
    state_2: AegisState = {
        "task_id": "t-2",
        "query": query_2,
        "has_dataset": False,
        "require_human_approval": False,
        "max_retries": 1,
        "retry_count": 0,
    }
    res_2 = orch_2.run(state_2)
    assert res_2["status"] == TaskStatus.COMPLETED
    report_2 = res_2["final_report"]

    # Verify report 1 has no text from query 2 / plan 2
    report_1_json = report_1.model_dump_json()
    assert "solar inverter" not in report_1_json
    assert "thermal load" not in report_1_json

    # Verify report 2 has no text from query 1 / plan 1
    report_2_json = report_2.model_dump_json()
    assert "hydraulic pressure" not in report_2_json
    assert "turbine hydraulic" not in report_2_json

    # Verify neither report contains unscripted domain findings
    for r_json in [report_1_json, report_2_json]:
        for pattern in DOMAIN_TRIPWIRE_PATTERNS:
            assert pattern.lower() not in r_json.lower()
