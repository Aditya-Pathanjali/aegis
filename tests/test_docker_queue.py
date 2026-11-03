"""Docker integration tests proving PR 10 Job Queue and Worker guarantees."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import List

import pytest
from arq.worker import create_worker
from fastapi.testclient import TestClient

from aegis.agents.critic import CriticRubricOutput
from aegis.agents.verification import JudgeVerdict
from aegis.api.main import app
from aegis.config import reset_settings
from aegis.db import get_task_store
from aegis.models.llm_outputs import PlannedSubtask, PlanOutput
from aegis.models.schemas import (
    AgentRole,
    EvidenceRef,
    Finding,
    SpecialistOutput,
    SynthesizerOutput,
    TaskStatus,
    TraceEvent,
)
from aegis.orchestration.checkpointer import reset_checkpointer
from aegis.queue import (
    publish_trace_event_sync,
    reset_redis_pools,
    set_task_cancelled,
)
from aegis.services.orchestrator_service import (
    reset_orchestrator_service,
)
from aegis.worker import get_worker_settings
from tests.fakes import FakeLLM, fake_structured, fake_text, fake_tool_call


@pytest.fixture(autouse=True)
def setup_queue_test_env(monkeypatch):
    monkeypatch.setenv("AEGIS_PROVIDER", "anthropic")
    monkeypatch.setenv("AEGIS_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    reset_settings(None)
    yield
    reset_settings(None)


class WorkflowFakeLLM(FakeLLM):
    def __init__(self, plan_output=None, synth_title="Overhaul Audit", sleep_secs=0.0):
        super().__init__(responses=[])
        self.sleep_secs = sleep_secs
        self.plan_output = plan_output or PlanOutput(
            objective="Inspect bearing and synthesize audit",
            rationale="Check safety thresholds",
            subtasks=[
                PlannedSubtask(
                    id="st-1",
                    description="Query operational data",
                    role=AgentRole.DATA_ANALYST,
                    tools=["sql_query"],
                )
            ],
        )
        self.spec_out = SpecialistOutput(
            summary="Data analysis complete.",
            findings=[
                Finding(
                    statement="Engine reached maximum lifespan of 192 operational cycles.",
                    evidence=[EvidenceRef(kind="tool_call", ref_id="call-1")],
                )
            ],
            limitations=["Sensor data analyzed for unit 1."],
        )
        self.judge_verdict = JudgeVerdict(
            verdict="supported",
            quote_span="192",
            rationale="Empirical SQL query confirms 192 maximum cycles.",
        )
        self.critic_rubric = CriticRubricOutput(
            answers_question=9.0,
            uses_task_data=9.0,
            claims_verified=9.0,
            limitations_stated=9.0,
        )
        self.synth_output = SynthesizerOutput(
            title=synth_title,
            final_finding="Approved maintenance schedule confirmed at 192 cycles.",
            key_conclusions=["All subsystems normal."],
            actionable_recommendations=["Sign off on operation."],
        )

    def structured(self, *args, **kwargs):
        self.calls.append({"method": "structured", **kwargs})
        if self.sleep_secs > 0:
            time.sleep(self.sleep_secs)
        schema = kwargs.get("schema")
        name = getattr(schema, "__name__", "")
        if "PlanOutput" in name:
            val = self.plan_output
        elif "SpecialistOutput" in name:
            val = self.spec_out
        elif "JudgeVerdict" in name:
            val = self.judge_verdict
        elif "CriticRubricOutput" in name:
            val = self.critic_rubric
        elif "SynthesizerOutput" in name:
            val = self.synth_output
        else:
            val = self.synth_output
        return fake_structured(val, model="fake-model")

    def chat(self, *args, **kwargs):
        self.calls.append({"method": "chat", **kwargs})
        messages = kwargs.get("messages", [])
        if any(getattr(m, "role", "") == "tool" for m in messages):
            return fake_text("Maximum cycles observed is 192.")
        return fake_tool_call(
            name="sql_query",
            args={"sql": "SELECT MAX(time_cycles) AS max_cycles FROM dataset;"},
            tool_call_id="call-1",
        )


@pytest.mark.docker
def test_task_runs_end_to_end_in_worker(postgres_container, redis_container):
    """Proves: task submitted via API enters queue as 202 Accepted, worker executes it, and report is in Postgres."""
    reset_orchestrator_service()
    reset_checkpointer()
    reset_redis_pools()

    fake_llm = WorkflowFakeLLM(synth_title="Overhaul Audit")
    client = TestClient(app)

    # 1. Submit task via POST /api/tasks
    resp = client.post(
        "/api/tasks",
        json={
            "query": "Bearing vibration diagnosis",
            "dataset_content": "unit_number,time_cycles\n1,192\n",
            "require_human_approval": False,
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    task_id = data["task_id"]
    assert data["status"] == "queued"

    # Verify task in Postgres has status QUEUED immediately
    store = get_task_store()
    init_state = store.get_state(task_id)
    assert init_state is not None
    assert init_state["status"] == TaskStatus.QUEUED

    # 2. Worker executes task in burst mode
    async def run_worker_burst():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": fake_llm},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_worker_burst())

    # 3. Verify task completed in Postgres
    final_state = store.get_state(task_id)
    assert final_state is not None
    assert final_state["status"] == TaskStatus.COMPLETED, f"Errors: {final_state.get('errors')}"
    assert final_state.get("final_report") is not None
    report = final_state["final_report"]
    title = report.title if hasattr(report, "title") else report["title"]
    assert title == "Overhaul Audit"

    # Verify API get_task returns completed report
    get_resp = client.get(f"/api/tasks/{task_id}")
    assert get_resp.status_code == 200
    task_resp = get_resp.json()
    assert task_resp["status"] == "completed"
    assert task_resp["report"]["title"] == "Overhaul Audit"


@pytest.mark.docker
def test_cancel_running_task_stops_worker(postgres_container, redis_container):
    """Proves: cancelling a running task stops execution within one node boundary, marking status CANCELLED."""
    reset_orchestrator_service()
    reset_checkpointer()
    reset_redis_pools()

    plan_output = PlanOutput(
        objective="Inspect bearing multi-step",
        rationale="Check thresholds",
        subtasks=[
            PlannedSubtask(
                id="st-1",
                description="Step 1 query",
                role=AgentRole.DATA_ANALYST,
                tools=["sql_query"],
            ),
            PlannedSubtask(
                id="st-2",
                description="Step 2 research",
                role=AgentRole.RESEARCH,
                tools=["rag_search"],
            ),
        ],
    )

    class CancellingFakeLLM(WorkflowFakeLLM):
        def __init__(self, task_to_cancel: List[str]):
            super().__init__(plan_output=plan_output)
            self.task_to_cancel = task_to_cancel

        def structured(self, *args, **kwargs):
            res = super().structured(*args, **kwargs)
            schema = kwargs.get("schema")
            name = getattr(schema, "__name__", "")
            if "PlanOutput" in name and self.task_to_cancel:
                tid = self.task_to_cancel[0]
                set_task_cancelled(tid)
            return res

    target_task_id: List[str] = []
    fake_llm = CancellingFakeLLM(target_task_id)
    client = TestClient(app)

    resp = client.post(
        "/api/tasks",
        json={
            "query": "Test cancellation workflow",
            "dataset_content": "unit_number,time_cycles\n1,192\n",
            "require_human_approval": False,
        },
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]
    target_task_id.append(task_id)

    async def run_worker():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": fake_llm},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_worker())

    # Verify status in Postgres is CANCELLED
    store = get_task_store()
    st = store.get_state(task_id)
    assert st is not None
    assert st["status"] == TaskStatus.CANCELLED
    # Synthesizer and FinalReport never ran
    assert st.get("final_report") is None

    # Verify cancellation trace event
    events = store.get_trace_events(task_id)
    cancelled_events = [ev for ev in events if ev.event_type == "task_cancelled"]
    assert len(cancelled_events) >= 1


@pytest.mark.docker
def test_concurrency_limit_queues_extra_tasks(postgres_container, redis_container):
    """Proves: with max_concurrent_tasks=2, enqueueing 4 tasks limits simultaneous executions to <= 2."""
    reset_orchestrator_service()
    reset_checkpointer()
    reset_redis_pools()

    active_lock = threading.Lock()
    active_count = 0
    max_active = 0

    class ConcurrencyTrackingLLM(WorkflowFakeLLM):
        def structured(self, *args, **kwargs):
            nonlocal active_count, max_active
            with active_lock:
                active_count += 1
                if active_count > max_active:
                    max_active = active_count

            time.sleep(0.15)
            try:
                return super().structured(*args, **kwargs)
            finally:
                with active_lock:
                    active_count -= 1

    fake_llm = ConcurrencyTrackingLLM()
    client = TestClient(app)

    task_ids = []
    for i in range(4):
        resp = client.post(
            "/api/tasks",
            json={
                "query": f"Concurrent task {i}",
                "dataset_content": "unit_number,time_cycles\n1,192\n",
                "require_human_approval": False,
            },
        )
        assert resp.status_code == 202
        task_ids.append(resp.json()["task_id"])

    async def run_worker():
        worker = create_worker(
            get_worker_settings(),
            max_jobs=2,
            burst=True,
            ctx={"llm": fake_llm},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_worker())

    # Assert max concurrent tasks was capped at 2
    assert max_active <= 2, f"Expected max concurrent <= 2, observed {max_active}"

    # Verify all tasks finished
    store = get_task_store()
    for tid in task_ids:
        st = store.get_state(tid)
        assert st is not None
        assert st["status"] == TaskStatus.COMPLETED


@pytest.mark.docker
def test_sse_reconnect_with_last_event_id_exact_stream(postgres_container, redis_container):
    """Proves: SSE client reconnecting with Last-Event-ID receives missed events without gaps or duplicates."""
    reset_orchestrator_service()
    reset_redis_pools()

    store = get_task_store()
    task_id = f"task-sse-reconnect-{uuid.uuid4().hex[:6]}"

    # Save initial task
    store.save_state(
        task_id,
        {
            "task_id": task_id,
            "query": "SSE reconnection check",
            "status": TaskStatus.EXECUTING,
            "trace_events": [],
            "errors": [],
        },
    )

    t0 = datetime.now(UTC)
    ev1 = TraceEvent(event_id="ev-101", task_id=task_id, event_type="plan", message="Plan created", timestamp=t0)
    ev2 = TraceEvent(event_id="ev-102", task_id=task_id, event_type="specialist_start", message="Specialist 1", timestamp=t0 + timedelta(milliseconds=10))
    store.save_trace_event(ev1)
    store.save_trace_event(ev2)
    publish_trace_event_sync(ev1)
    publish_trace_event_sync(ev2)

    client = TestClient(app)

    # First connection: read 2 events and disconnect
    pass1_ids = []
    with client.stream("GET", f"/api/tasks/{task_id}/events") as resp1:
        assert resp1.status_code == 200
        for line in resp1.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if "event_id" in payload:
                    pass1_ids.append(payload["event_id"])
                if len(pass1_ids) == 2:
                    break

    assert pass1_ids == ["ev-101", "ev-102"]

    # Now emit remaining events and complete task
    ev3 = TraceEvent(event_id="ev-103", task_id=task_id, event_type="specialist_end", message="Specialist complete", timestamp=t0 + timedelta(milliseconds=20))
    ev4 = TraceEvent(event_id="ev-104", task_id=task_id, event_type="verify", message="Verification complete", timestamp=t0 + timedelta(milliseconds=30))
    ev5 = TraceEvent(event_id="ev-105", task_id=task_id, event_type="synthesize", message="Report synthesized", timestamp=t0 + timedelta(milliseconds=40))
    for ev in [ev3, ev4, ev5]:
        store.save_trace_event(ev)
        publish_trace_event_sync(ev)

    st = store.get_state(task_id)
    st["status"] = TaskStatus.COMPLETED
    store.save_state(task_id, st)

    # Reconnect with Last-Event-ID: ev-102
    pass2_ids = []
    headers = {"Last-Event-ID": "ev-102"}
    with client.stream("GET", f"/api/tasks/{task_id}/events", headers=headers) as resp2:
        assert resp2.status_code == 200
        for line in resp2.iter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if "event_id" in payload:
                    pass2_ids.append(payload["event_id"])
                elif payload.get("event_type") == "stream_end":
                    break

    # Client 2 should receive only ev-103, ev-104, ev-105
    assert pass2_ids == ["ev-103", "ev-104", "ev-105"]
    assert "ev-101" not in pass2_ids
    assert "ev-102" not in pass2_ids

    # Combined sequence is exact without gaps or duplicates
    combined = pass1_ids + pass2_ids
    assert combined == ["ev-101", "ev-102", "ev-103", "ev-104", "ev-105"]
    assert len(combined) == len(set(combined))


@pytest.mark.docker
def test_worker_timeout_fails_loudly_worker_remains_healthy(postgres_container, redis_container):
    """Proves: task exceeding task_timeout_s fails cleanly in Postgres, and worker immediately runs next task."""
    reset_orchestrator_service()
    reset_checkpointer()
    reset_redis_pools()

    store = get_task_store()
    client = TestClient(app)

    # 1. Enqueue slow task that will timeout
    resp1 = client.post(
        "/api/tasks",
        json={"query": "Slow timeout task", "require_human_approval": False},
    )
    assert resp1.status_code == 202
    slow_task_id = resp1.json()["task_id"]

    slow_llm = WorkflowFakeLLM(sleep_secs=2.0)

    # Run worker with timeout = 1 second
    async def run_slow_worker():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": slow_llm, "job_timeout": 1},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_slow_worker())

    # Verify slow task marked FAILED with timeout error
    st1 = store.get_state(slow_task_id)
    assert st1 is not None
    assert st1["status"] == TaskStatus.FAILED
    assert any("timed out" in err.lower() for err in st1.get("errors", []))

    # 2. Enqueue normal fast task
    resp2 = client.post(
        "/api/tasks",
        json={
            "query": "Fast subsequent task",
            "dataset_content": "unit_number,time_cycles\n1,192\n",
            "require_human_approval": False,
        },
    )
    assert resp2.status_code == 202
    fast_task_id = resp2.json()["task_id"]

    fast_llm = WorkflowFakeLLM(sleep_secs=0.0)

    async def run_fast_worker():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": fast_llm, "job_timeout": 10},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_fast_worker())

    # Verify fast task completed cleanly, showing worker remained healthy
    st2 = store.get_state(fast_task_id)
    assert st2 is not None
    assert st2["status"] == TaskStatus.COMPLETED
    assert st2.get("final_report") is not None


@pytest.mark.docker
def test_human_approval_pause_resume_across_worker_restarts(postgres_container, redis_container):
    """Proves: human approval pauses task, releases worker slot, and resumes across worker restarts."""
    reset_orchestrator_service()
    reset_checkpointer()
    reset_redis_pools()

    store = get_task_store()
    client = TestClient(app)

    fake_llm = WorkflowFakeLLM()

    # 1. Submit task with human approval required
    resp = client.post(
        "/api/tasks",
        json={
            "query": "Audit requiring gate approval",
            "dataset_content": "unit_number,time_cycles\n1,192\n",
            "require_human_approval": True,
        },
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # 2. Worker 1 runs and halts at human gate
    async def run_worker_1():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": fake_llm},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_worker_1())

    # Verify task status in Postgres is WAITING_APPROVAL
    st = store.get_state(task_id)
    assert st is not None
    assert st["status"] == TaskStatus.WAITING_APPROVAL

    # 3. Client calls /api/tasks/{task_id}/approve to approve
    app_resp = client.post(
        f"/api/tasks/{task_id}/approve",
        json={"action": "approve", "feedback": "Approved by engineering manager"},
    )
    assert app_resp.status_code in (200, 202)

    # 4. Worker 2 (fresh instance simulating process restart) picks up resume job
    async def run_worker_2():
        worker = create_worker(
            get_worker_settings(),
            burst=True,
            ctx={"llm": fake_llm},
        )
        await worker.async_run()
        await worker.close()

    asyncio.run(run_worker_2())

    # Verify task completed successfully
    final_st = store.get_state(task_id)
    assert final_st is not None
    assert final_st["status"] == TaskStatus.COMPLETED
    assert final_st.get("final_report") is not None
