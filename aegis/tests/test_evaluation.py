"""Unit tests for the evaluation benchmark endpoints."""

import pytest
from fastapi.testclient import TestClient

from aegis.api.main import app
from aegis.evaluation.runner import EvaluationRunner


def test_evaluation_runner_not_implemented():
    runner = EvaluationRunner()
    with pytest.raises(NotImplementedError, match="Phase 3"):
        runner.run_benchmark()


def test_evaluation_endpoints_return_501():
    client = TestClient(app)

    resp_run = client.post("/api/evaluation/run")
    assert resp_run.status_code == 501
    assert "Phase 3" in resp_run.json()["detail"]

    resp_results = client.get("/api/evaluation/results")
    assert resp_results.status_code == 501
    assert "Phase 3" in resp_results.json()["detail"]
