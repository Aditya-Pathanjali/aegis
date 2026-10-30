"""Tests for PR 3: TaskContext, per-task data, tools, AST security, and limits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aegis.api.main import app
from aegis.config import Settings, reset_settings
from aegis.tools.context import TaskContext
from aegis.tools.describe_tool import DescribeDatasetTool
from aegis.tools.python_sandbox import PythonExecTool, SecurityViolationError
from aegis.tools.sql_tool import SQLQueryTool


def test_ast_rejects_pd_io_common_os_getcwd():
    tool = PythonExecTool()
    code = "import pandas as pd\nx = pd.io.common.os.getcwd()"
    with pytest.raises(SecurityViolationError, match="Access to attribute 'os' is prohibited"):
        tool._validate_ast(code)

    res = tool.execute(code=code)
    assert res.success is False
    assert "SecurityViolationError" in res.error

    # Attributes starting with _ also rejected
    with pytest.raises(SecurityViolationError, match="Access to attribute '__class__' is prohibited"):
        tool._validate_ast("x = (1).__class__")


def test_python_exec_infinite_loop_times_out():
    tool = PythonExecTool(timeout_seconds=1.0)
    res = tool.execute(code="while True: pass")
    assert res.success is False
    assert "timed out" in res.error.lower()


def test_subprocess_cannot_see_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-test-key-99999")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-key-88888")

    tool = PythonExecTool()
    env = tool._clean_environment()
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env

    with tempfile.TemporaryDirectory() as tmp:
        res = tool._run_subprocess(
            "import os\nprint('KEY_SEEN=' + str(os.environ.get('ANTHROPIC_API_KEY')))",
            workdir=Path(tmp),
            timeout=5.0,
        )
        assert "KEY_SEEN=None" in res["stdout"]


def test_two_concurrent_tasks_see_different_datasets():
    df1 = pd.DataFrame({"sensor_id": [1, 2], "val": [10.0, 20.0]})
    df2 = pd.DataFrame({"sensor_id": [100, 200], "val": [500.0, 600.0]})

    ctx1 = TaskContext(dataset=df1, task_id="task-1")
    ctx2 = TaskContext(dataset=df2, task_id="task-2")

    try:
        # 1. SQL query isolation
        sql1 = SQLQueryTool(context=ctx1)
        sql2 = SQLQueryTool(context=ctx2)

        res1 = sql1.execute(sql="SELECT SUM(val) as total FROM dataset")
        res2 = sql2.execute(sql="SELECT SUM(val) as total FROM dataset")

        assert res1.success is True
        assert res1.data["rows"][0][0] == 30.0

        assert res2.success is True
        assert res2.data["rows"][0][0] == 1100.0

        # 2. Describe dataset isolation
        desc1 = DescribeDatasetTool(context=ctx1)
        desc2 = DescribeDatasetTool(context=ctx2)

        d1 = desc1.execute()
        d2 = desc2.execute()

        assert d1.data["rows"][0]["val"] == 10.0
        assert d2.data["rows"][0]["val"] == 500.0

        # 3. Python exec parquet isolation
        py1 = PythonExecTool(context=ctx1)
        py2 = PythonExecTool(context=ctx2)

        r1 = py1.execute(
            code="import pandas as pd\ndf = pd.read_parquet('dataset.parquet')\nprint('SUM:', df['val'].sum())"
        )
        r2 = py2.execute(
            code="import pandas as pd\ndf = pd.read_parquet('dataset.parquet')\nprint('SUM:', df['val'].sum())"
        )

        assert "SUM: 30.0" in r1.data["stdout"]
        assert "SUM: 1100.0" in r2.data["stdout"]

    finally:
        ctx1.cleanup()
        ctx2.cleanup()


def test_drop_and_multi_statement_sql_rejected():
    df = pd.DataFrame({"id": [1, 2], "name": ["Alpha", "Beta"]})
    ctx = TaskContext(dataset=df)
    try:
        tool = SQLQueryTool(context=ctx)

        # Valid SELECT
        valid_res = tool.execute(sql="SELECT * FROM dataset")
        assert valid_res.success is True
        assert valid_res.data["row_count"] == 2

        # DROP rejected by SQLite authorizer
        drop_res = tool.execute(sql="DROP TABLE dataset")
        assert drop_res.success is False
        assert "not authorized" in drop_res.error.lower() or "denied" in drop_res.error.lower()

        # INSERT / DELETE rejected
        insert_res = tool.execute(sql="INSERT INTO dataset VALUES (3, 'Gamma')")
        assert insert_res.success is False

        delete_res = tool.execute(sql="DELETE FROM dataset")
        assert delete_res.success is False

        # Multi-statement rejected
        multi_res = tool.execute(sql="SELECT 1; SELECT 2")
        assert multi_res.success is False
        assert "one statement at a time" in multi_res.error.lower()

    finally:
        ctx.cleanup()


def test_oversized_csv_returns_422(monkeypatch):
    from tests.fakes import FakeLLM

    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5", dataset_max_bytes=100))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("aegis.services.orchestrator_service.get_llm_provider", lambda: FakeLLM(model="test"))

    client = TestClient(app)

    # 1. Byte limit exceeded via JSON
    oversized_csv = "col1,col2\n" + "x" * 200
    resp_bytes = client.post(
        "/api/tasks",
        json={"query": "Analyze data", "dataset_content": oversized_csv},
    )
    assert resp_bytes.status_code == 422
    assert "exceeds limit" in resp_bytes.json()["detail"]

    # 2. Byte limit exceeded via multipart upload
    resp_file = client.post(
        "/api/tasks",
        data={"query": "Analyze upload"},
        files={"file": ("big.csv", oversized_csv.encode("utf-8"), "text/csv")},
    )
    assert resp_file.status_code == 422
    assert "exceeds limit" in resp_file.json()["detail"]

    # 3. Row limit exceeded
    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5", dataset_max_rows=2))
    row_csv = "id,val\n1,10\n2,20\n3,30\n"
    resp_rows = client.post(
        "/api/tasks",
        json={"query": "Analyze rows", "dataset_content": row_csv},
    )
    assert resp_rows.status_code == 422
    assert "row count" in resp_rows.json()["detail"]

    # 4. Col limit exceeded
    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5", dataset_max_cols=2))
    col_csv = "c1,c2,c3\n1,2,3\n"
    resp_cols = client.post(
        "/api/tasks",
        json={"query": "Analyze cols", "dataset_content": col_csv},
    )
    assert resp_cols.status_code == 422
    assert "column count" in resp_cols.json()["detail"]

    # 5. Column name sanitization
    reset_settings(Settings(provider="anthropic", model="claude-sonnet-5-5"))
    messy_csv = "  First-Col! , Second@Col # \n1,2\n"
    resp_clean = client.post(
        "/api/tasks",
        json={"query": "Analyze clean", "dataset_content": messy_csv},
    )
    assert resp_clean.status_code == 200


def test_ast_rejects_getattr_probe_and_alias():
    tool = PythonExecTool()

    # Direct getattr probe
    probe1 = "import pandas as pd\nm = getattr(pd.io.common, 'o' + 's')\nprint(m.getcwd())"
    with pytest.raises(SecurityViolationError, match="Identifier 'getattr' is prohibited|prohibited"):
        tool._validate_ast(probe1)

    res1 = tool.execute(code=probe1)
    assert res1.success is False
    assert "SecurityViolationError" in res1.error

    # Alias form probe
    probe2 = "import pandas as pd\ng = getattr\nm = g(pd.io.common, 'o' + 's')\nprint(m.getcwd())"
    with pytest.raises(SecurityViolationError, match="Identifier 'getattr' is prohibited|prohibited"):
        tool._validate_ast(probe2)

    res2 = tool.execute(code=probe2)
    assert res2.success is False
    assert "SecurityViolationError" in res2.error

    # Other forbidden identifiers
    for forbidden in ["setattr", "delattr", "vars", "globals", "locals", "dir", "type"]:
        code = f"x = {forbidden}"
        with pytest.raises(SecurityViolationError):
            tool._validate_ast(code)


def test_sql_recursive_cte_query_times_out():
    tool = SQLQueryTool(timeout_seconds=0.5)
    query = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    res = tool.execute(sql=query)
    assert res.success is False
    assert "timed out" in res.error.lower()


def test_runner_corrupt_parquet_reports_error():
    tool = PythonExecTool()
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        # Write corrupt parquet
        (workdir / "dataset.parquet").write_bytes(b"corrupt-parquet-data")
        with pytest.raises(RuntimeError, match="Failed to load dataset.parquet"):
            tool._run_subprocess("print('should not execute')", workdir=workdir, timeout=5.0)


def test_orchestrator_worker_fails_on_unparseable_csv():
    from aegis.models.schemas import TaskSubmission
    from aegis.services.orchestrator_service import OrchestratorService
    from tests.fakes import FakeLLM

    svc = OrchestratorService(llm_provider=FakeLLM(model="test"))
    # Give an invalid CSV with inconsistent column counts that causes pandas ParserError
    bad_submission = TaskSubmission(
        query="Test unparseable csv",
        dataset_content="col1,col2\n1,2\n3,4,5,6\n",
    )
    task_id = "test-bad-csv-task"
    initial_state = {
        "task_id": task_id,
        "query": bad_submission.query,
        "status": "pending",
        "plan": None,
        "results": {},
        "verification": None,
        "critique": None,
        "final_report": None,
        "telemetry": None,
        "human_approval": None,
        "errors": [],
        "iteration": 0,
        "phase": "init",
    }
    svc.state_store.save_state(task_id, initial_state)
    svc._run_workflow_worker(task_id, initial_state, bad_submission)
    saved = svc.state_store.get_state(task_id)
    assert saved["status"] == "failed"
    assert any("Failed to parse dataset" in err or "ParserError" in err for err in saved["errors"])
