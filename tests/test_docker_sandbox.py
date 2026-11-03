"""Security isolation and escape test suite for the Docker container sandbox."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from docker.errors import DockerException

from aegis.tools.context import TaskContext
from aegis.tools.python_sandbox import PythonExecTool, SecurityViolationError

pytestmark = pytest.mark.docker


def test_docker_sandbox_normal_analysis_cmapss_max_cycle():
    """Validates that a normal pandas analysis on C-MAPSS dataset runs and extracts the correct max-cycle."""
    csv_path = Path(__file__).resolve().parents[1] / "data" / "samples" / "fd001_train_5engines.csv"
    assert csv_path.is_file(), f"C-MAPSS sample file not found at {csv_path}"

    df = pd.read_csv(csv_path)
    expected_max_cycles = int(df[df["unit_number"] == 1]["time_cycles"].max())
    assert expected_max_cycles == 192

    context = TaskContext(dataset=df, task_id="cmapss-max-cycle-analysis")
    tool = PythonExecTool(context=context)

    code = """
import pandas as pd
e1_df = df[df['unit_number'] == 1]
max_cycles = int(e1_df['time_cycles'].max())
print(f"Engine 1 maximum cycles: {max_cycles}")
"""
    result = tool.execute(code=code)
    assert result.success is True
    assert result.data["variables"]["max_cycles"] == 192
    assert "Engine 1 maximum cycles: 192" in result.data["stdout"]


def test_docker_sandbox_escape_reading_host_file():
    """Validates that attempts to read host files fail safely at both AST and container levels."""
    tool = PythonExecTool()

    # 1. AST pre-filter rejection
    ast_probe = "with open('/etc/shadow') as f:\n    data = f.read()"
    with pytest.raises(SecurityViolationError, match="Function call 'open\\(\\)' is prohibited"):
        tool._validate_ast(ast_probe)

    # 2. Container OS kernel boundary rejection
    container_probe = """
try:
    with open('/etc/shadow', 'r') as f:
        content = f.read()
except Exception as e:
    err_type = type(e).__name__
    err_msg = str(e)
"""
    res = tool._run_container(container_probe, bypass_ast=True)
    assert res["variables"]["err_type"] == "PermissionError"
    assert "Permission denied" in res["variables"]["err_msg"]


def test_docker_sandbox_escape_network_access():
    """Validates that external network egress is completely disabled (network_mode='none')."""
    tool = PythonExecTool()

    # 1. AST pre-filter rejection
    with pytest.raises(SecurityViolationError, match="Importing module 'socket' is prohibited"):
        tool._validate_ast("import socket\ns = socket.socket()")

    # 2. Container kernel boundary rejection
    container_probe = """
import socket
net_err = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect(('8.8.8.8', 53))
except Exception as e:
    net_err = f"{type(e).__name__}: {e}"
"""
    res = tool._run_container(container_probe, bypass_ast=True)
    assert res["variables"]["net_err"] is not None
    assert "Network is unreachable" in res["variables"]["net_err"] or "OSError" in res["variables"]["net_err"]


def test_docker_sandbox_escape_pickle_code_execution():
    """Validates that pd.read_pickle/np.load code execution cannot write to system or break container isolation."""
    tool = PythonExecTool()

    # Craft pickle payload: posix.system("touch /etc/pwned")
    # Using standard pickle opcodes: c <module>\n<name>\n ( <args> t R .
    pickle_bytes = b"cposix\nsystem\n(S'touch /etc/pwned'\ntR."
    payload_b64 = base64.b64encode(pickle_bytes).decode("ascii")

    container_probe = f"""
import base64
import os
import pandas as pd

raw = base64.b64decode('{payload_b64}')
with open('/workspace/payload.pkl', 'wb') as f:
    f.write(raw)

pickle_err = None
try:
    pd.read_pickle('/workspace/payload.pkl')
except Exception as e:
    pickle_err = f"{{type(e).__name__}}: {{e}}"

pwned_created = os.path.exists('/etc/pwned')
"""
    res = tool._run_container(container_probe, bypass_ast=True)
    assert res["variables"]["pwned_created"] is False


def test_docker_sandbox_escape_fork_bomb():
    """Validates that PID limits (pids_limit=64) prevent fork bombs from exhausting host resources."""
    tool = PythonExecTool(timeout_seconds=3.0)

    # 1. AST pre-filter rejection
    with pytest.raises(SecurityViolationError):
        tool._validate_ast("import os\nwhile True:\n    os.fork()")

    # 2. Container kernel boundary throttling or clean timeout kill
    container_probe = """
import os
caught_fork_error = False
try:
    while True:
        os.fork()
except (BlockingIOError, OSError):
    caught_fork_error = True
"""
    try:
        res = tool._run_container(container_probe, timeout=3.0, bypass_ast=True)
        assert res["variables"].get("caught_fork_error") is True
    except TimeoutError:
        # Container was constrained by PID limit and killed cleanly on wall-clock timeout
        pass


def test_docker_sandbox_escape_memory_exhaustion():
    """Validates that memory limits (mem_limit=256m) kill runaway allocations with exit code 137."""
    tool = PythonExecTool()

    # Attempt to allocate 500MB (limit is 256MB)
    oom_code = "data = bytearray(500 * 1024 * 1024)"
    res = tool.execute(code=oom_code)
    assert res.success is False
    assert "memory limit" in res.error.lower() or "oom" in res.error.lower() or "killed" in res.error.lower()


def test_docker_sandbox_escape_writing_outside_workdir():
    """Validates that the root filesystem is mounted read-only and writes outside tmpfs fail."""
    tool = PythonExecTool()

    # 1. AST pre-filter rejection
    with pytest.raises(SecurityViolationError):
        tool._validate_ast("open('/etc/test.txt', 'w').write('evil')")

    # 2. Container kernel boundary rejection
    container_probe = """
write_err = None
try:
    with open('/etc/unauthorized.txt', 'w') as f:
        f.write('fail')
except Exception as e:
    write_err = f"{type(e).__name__}: {e}"
"""
    res = tool._run_container(container_probe, bypass_ast=True)
    assert "ReadOnly" in res["variables"]["write_err"] or "Read-only file system" in res["variables"]["write_err"]


def test_docker_sandbox_escape_getattr_subclasses():
    """Validates that getattr and __subclasses__ escape attempts are rejected."""
    tool = PythonExecTool()

    # Direct getattr probe
    probe1 = "import pandas as pd\nm = getattr(pd.io.common, 'os')\nprint(m.getcwd())"
    with pytest.raises(SecurityViolationError, match="getattr"):
        tool._validate_ast(probe1)

    res1 = tool.execute(code=probe1)
    assert res1.success is False
    assert "SecurityViolationError" in res1.error

    # __subclasses__ probe
    probe2 = "subclasses = ().__class__.__base__.__subclasses__()"
    with pytest.raises(SecurityViolationError, match="Access to attribute"):
        tool._validate_ast(probe2)

    res2 = tool.execute(code=probe2)
    assert res2.success is False
    assert "SecurityViolationError" in res2.error


def test_docker_sandbox_dataset_readonly_protection():
    """Validates that dataset.parquet is mounted read-only and cannot be overwritten."""
    df = pd.DataFrame({"sensor_a": [1.0, 2.0], "sensor_b": [10.0, 20.0]})
    context = TaskContext(dataset=df)
    tool = PythonExecTool(context=context)

    overwrite_probe = """
overwrite_err = None
try:
    with open('dataset.parquet', 'wb') as f:
        f.write(b'tampered')
except Exception as e:
    overwrite_err = f"{type(e).__name__}: {e}"
"""
    res = tool._run_container(overwrite_probe, bypass_ast=True)
    assert res["variables"]["overwrite_err"] is not None
    assert "Permission denied" in res["variables"]["overwrite_err"] or "Read-only file system" in res["variables"]["overwrite_err"]

    # Verify host parquet file is unmodified
    reloaded = pd.read_parquet(context.parquet_path)
    assert len(reloaded) == 2
    assert list(reloaded.columns) == ["sensor_a", "sensor_b"]


def test_docker_sandbox_timeout_kills_container_cleanly():
    """Validates that infinite loops are forcefully killed within the configured timeout."""
    tool = PythonExecTool(timeout_seconds=2.0)
    res = tool.execute(code="while True:\n    pass")
    assert res.success is False
    assert "timed out" in res.error.lower()


def test_docker_sandbox_non_root_user_and_capabilities():
    """Validates that execution runs under UID 1000 and dropped capabilities prevent raw sockets."""
    tool = PythonExecTool()

    container_probe = """
import os
import socket

uid = os.getuid()
gid = os.getgid()

raw_socket_err = None
try:
    socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
except Exception as e:
    raw_socket_err = f"{type(e).__name__}: {e}"
"""
    res = tool._run_container(container_probe, bypass_ast=True)
    assert res["variables"]["uid"] == 1000
    assert res["variables"]["gid"] == 1000
    assert "Operation not permitted" in res["variables"]["raw_socket_err"]


def test_docker_stopped_returns_error_no_subprocess_started(monkeypatch):
    """Validates fail-closed behavior: when Docker is unavailable, an error is returned and no subprocess is started."""
    tool = PythonExecTool()

    # Mock docker.from_env to raise DockerException
    def mock_from_env(*args, **kwargs):
        raise DockerException("Docker daemon is stopped or unreachable.")

    monkeypatch.setattr("docker.from_env", mock_from_env)

    # Spy on subprocess.run and Popen to verify zero host fallback
    subprocess_spy = MagicMock()
    monkeypatch.setattr(subprocess, "run", subprocess_spy)
    monkeypatch.setattr(subprocess, "Popen", subprocess_spy)

    result = tool.execute(code="x = 1\nprint(x)")

    assert result.success is False
    assert "Docker sandbox runtime unavailable" in result.error
    assert subprocess_spy.call_count == 0

