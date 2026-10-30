"""Unit tests for Aegis tools and security sandbox."""

import pytest

from aegis.models.schemas import AgentRole
from aegis.tools.python_sandbox import PythonExecTool, SecurityViolationError
from aegis.tools.registry import ToolRegistry
from aegis.tools.sql_tool import SQLQueryTool


def test_python_sandbox_math_and_pandas():
    sandbox = PythonExecTool()
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
    sandbox = PythonExecTool()

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
    import pandas as pd

    from aegis.tools.context import TaskContext

    df = pd.DataFrame({"status_flag": ["OK", "WARN"], "metric_val": [10.0, 20.0]})
    ctx = TaskContext(dataset=df)
    tool = SQLQueryTool(context=ctx)
    res = tool.execute(sql="SELECT status_flag, AVG(metric_val) FROM dataset GROUP BY status_flag;")
    assert res.success is True
    assert res.data["row_count"] > 0
    assert "status_flag" in res.data["columns"]
    ctx.cleanup()


def test_tool_registry_permissions():
    reg = ToolRegistry()

    # Planner is not authorized to execute python_exec
    rec1 = reg.execute_tool(AgentRole.PLANNER, "python_exec", {"code": "x = 1"})
    assert rec1.success is False
    assert "Security Policy Violation" in rec1.error

    # Data Analyst is authorized to execute python_exec
    rec2 = reg.execute_tool(AgentRole.DATA_ANALYST, "python_exec", {"code": "result = 42"})
    assert rec2.success is True

    # Obsolete aliases are not registered
    assert reg.get_tool("python_sandbox") is None
    assert reg.get_tool("rag_retriever") is None
    rec_old1 = reg.execute_tool(AgentRole.DATA_ANALYST, "python_sandbox", {"code": "x = 1"})
    assert rec_old1.success is False
    rec_old2 = reg.execute_tool(AgentRole.RESEARCH, "rag_retriever", {"query": "test"})
    assert rec_old2.success is False
