"""Regression tests for the Phase 2 Part 1 review findings (no Docker needed)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from aegis.config import Settings
from aegis.tools.python_sandbox import _SANDBOX_INPUT_DIR, PythonExecTool
from aegis.worker import TaskCancelledException, TaskTimedOutException, _make_on_node, _StopGuardLLM
from tests.fakes import FakeLLM


def _settings(**overrides: Any) -> Settings:
    """Settings from explicit values only, ignoring any developer .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


def _parquet(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    f = dirpath / "dataset.parquet"
    f.write_bytes(b"PAR1")
    return f


def test_compose_mode_mounts_only_the_task_subpath_of_the_volume(tmp_path):
    """In compose the worker's paths don't exist on the Docker host, so the sandbox must mount the
    task's subpath of the shared volume (a bind mount silently became an empty directory)."""
    base = tmp_path / "tasks"
    pq = _parquet(base / "aegis_task_abc_123")
    s = _settings(task_data_dir=str(base), sandbox_task_volume="aegis_taskdata")

    (mount,) = PythonExecTool._dataset_mounts(pq, s)
    assert mount["Type"] == "volume"
    assert mount["Source"] == "aegis_taskdata"
    assert mount["Target"] == _SANDBOX_INPUT_DIR
    assert mount["ReadOnly"] is True
    assert mount["VolumeOptions"]["Subpath"] == "aegis_task_abc_123"


def test_compose_mode_refuses_a_dataset_outside_the_task_volume(tmp_path):
    pq = _parquet(tmp_path / "elsewhere")
    s = _settings(task_data_dir=str(tmp_path / "tasks"), sandbox_task_volume="aegis_taskdata")
    with pytest.raises(RuntimeError, match="outside task_data_dir"):
        PythonExecTool._dataset_mounts(pq, s)


def test_local_mode_bind_mounts_the_task_dir_read_only(tmp_path):
    pq = _parquet(tmp_path / "task")
    (mount,) = PythonExecTool._dataset_mounts(pq, _settings())
    assert mount["Type"] == "bind"
    assert mount["Source"] == str(pq.parent)
    assert mount["ReadOnly"] is True


def test_missing_dataset_fails_instead_of_running_on_an_empty_frame(tmp_path):
    with pytest.raises(RuntimeError, match="Task dataset missing"):
        PythonExecTool._dataset_mounts(tmp_path / "nope" / "dataset.parquet", _settings())


def test_task_volume_settings_must_be_set_together():
    with pytest.raises(ValueError, match="set together"):
        _settings(sandbox_task_volume="aegis_taskdata")


def test_empty_env_values_are_treated_as_unset(monkeypatch):
    """docker compose passes ${AEGIS_PROVIDER:-} as "" when unset; that crash-looped the worker and API."""
    monkeypatch.setenv("AEGIS_PROVIDER", "")
    monkeypatch.setenv("AEGIS_MODEL", "")
    s = _settings()
    assert s.provider is None and s.model is None


def test_stop_guard_blocks_llm_calls_after_timeout(monkeypatch):
    """asyncio.wait_for can't kill the task thread; the guard stops its next (paid) LLM call."""
    monkeypatch.setattr("aegis.worker.is_task_cancelled", lambda task_id: False)
    inner = FakeLLM(responses=["first"])
    timed_out = threading.Event()
    guarded = _StopGuardLLM(inner, "t1", timed_out)

    guarded.chat(messages=[])
    timed_out.set()
    with pytest.raises(TaskTimedOutException):
        guarded.chat(messages=[])
    with pytest.raises(TaskTimedOutException):
        guarded.structured(messages=[], schema=Settings)
    assert len(inner.calls) == 1
    assert guarded.model == inner.model  # still looks like the wrapped provider


def test_stop_guard_blocks_llm_calls_after_cancel(monkeypatch):
    monkeypatch.setattr("aegis.worker.is_task_cancelled", lambda task_id: True)
    inner = FakeLLM(responses=["never"])
    with pytest.raises(TaskCancelledException):
        _StopGuardLLM(inner, "t1", threading.Event()).chat(messages=[])
    assert inner.calls == []


def test_on_node_never_overwrites_state_after_timeout(monkeypatch):
    monkeypatch.setattr("aegis.worker.is_task_cancelled", lambda task_id: False)
    saved: list = []

    class Store:
        def save_state(self, task_id, state):
            saved.append(state)

    timed_out = threading.Event()
    on_node = _make_on_node(Store(), "t1", timed_out)
    on_node({"status": "executing"})
    timed_out.set()
    with pytest.raises(TaskTimedOutException):
        on_node({"status": "executing"})
    assert len(saved) == 1


def test_health_reports_sandbox_from_worker_heartbeat_not_docker(monkeypatch):
    """The API has no Docker socket; sandbox health comes from the worker's Redis heartbeat."""
    from aegis.api import routes

    monkeypatch.setattr(routes, "check_db_health", lambda url: True)
    monkeypatch.setattr(routes, "check_redis_health", lambda url: True)

    def no_docker(*a, **k):
        raise AssertionError("the API must not touch Docker")

    monkeypatch.setattr("docker.from_env", no_docker)

    monkeypatch.setattr("aegis.queue.read_sandbox_heartbeat", lambda: True)
    code, body = routes.compute_health_status()
    assert code == 200 and body["sandbox"] == "healthy"

    monkeypatch.setattr("aegis.queue.read_sandbox_heartbeat", lambda: False)
    code, body = routes.compute_health_status()
    assert code == 503 and body["sandbox"] == "down"
