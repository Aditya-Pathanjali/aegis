"""Unit tests for Aegis tools and security sandbox."""

import pytest
from aegis.models.schemas import AgentRole
from aegis.tools.python_sandbox import PythonSandboxTool, SecurityViolationError
from aegis.tools.registry import ToolRegistry, get_tool_registry
from aegis.tools.sql_tool import SQLQueryTool
from aegis.tools.web_search import WebSearchTool


def test_python_sandbox_math_and_pandas():
    sandbox = PythonSandboxTool()
    code = """
import numpy as np
import pandas as pd

x = [10, 20, 30]
mean_x = float(np.mean(x))
df = pd.DataFrame({'val': x})
total = int(df['val'].sum())
print(f"Calculated mean: {mean_x}, total: {total}")
"""
    result = sandbox.execute(code=code)
    assert result.success is True
    assert result.data["variables"]["mean_x"] == 20.0
    assert result.data["variables"]["total"] == 60
    assert "Calculated mean: 20.0" in result.data["stdout"]


def test_python_sandbox_security_rejections():
    sandbox = PythonSandboxTool()

    # Direct AST check raises SecurityViolationError
    with pytest.raises(SecurityViolationError):
        sandbox._validate_ast("import os\nos.system('dir')")

    with pytest.raises(SecurityViolationError):
        sandbox._validate_ast("import subprocess\nsubprocess.call(['ls'])")

    with pytest.raises(SecurityViolationError):
        sandbox._validate_ast("eval('1 + 1')")

    # execute() catches exception and returns structured failure record
    res = sandbox.execute(code="import os\nos.system('dir')")
    assert res.success is False
    assert "SecurityViolationError" in res.error


def test_sql_query_tool():
    tool = SQLQueryTool()
    res = tool.execute(query="SELECT status_flag, AVG(vibration_rms_x) FROM sensor_telemetry GROUP BY status_flag;")
    assert res.success is True
    assert res.data["row_count"] > 0
    assert "status_flag" in res.data["columns"]


def test_web_search_tool():
    tool = WebSearchTool()
    res = tool.execute(query="ISO 10816 vibration limits")
    assert res.success is True
    assert res.data["results_count"] > 0
    assert any("10816" in r["title"] for r in res.data["results"])


def test_tool_registry_permissions():
    reg = ToolRegistry()

    # Planner is not authorized to execute python_sandbox
    rec1 = reg.execute_tool(AgentRole.PLANNER, "python_sandbox", {"code": "x = 1"})
    assert rec1.success is False
    assert "Security Policy Violation" in rec1.error

    # Data Analyst IS authorized
    rec2 = reg.execute_tool(AgentRole.DATA_ANALYST, "python_sandbox", {"code": "result = 42"})
    assert rec2.success is True
    assert rec2.output["variables"]["result"] == 42
