"""Sandboxed Python code execution tool with AST security validation and subprocess isolation."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from aegis.config import get_settings
from aegis.tools.base import BaseTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext


class SecurityViolationError(Exception):
    """Raised when forbidden modules, builtins, or attribute chains are detected in AST."""
    pass


class SafeCodeValidator(ast.NodeVisitor):
    """
    AST validator to prevent dangerous operations.
    Note: AST filtering is defense-in-depth and not an absolute security boundary.
    """

    FORBIDDEN_MODULES = {
        "os", "subprocess", "sys", "shutil", "socket", "pty", "commands",
        "multiprocessing", "threading", "ctypes", "winreg", "posix", "importlib",
    }

    FORBIDDEN_CALLS = {
        "eval", "exec", "__import__", "compile", "open", "breakpoint", "input",
    }

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(f"Importing module '{alias.name}' is prohibited for security.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(f"Importing from module '{node.module}' is prohibited for security.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(f"Function call '{node.func.id}()' is prohibited inside the sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr_name = node.attr
        if attr_name.startswith("_") or attr_name in {"os", "sys", "subprocess"}:
            raise SecurityViolationError(f"Access to attribute '{attr_name}' is prohibited for security.")
        self.generic_visit(node)


_RUNNER_TEMPLATE = r"""
import io
import json
import sys
from pathlib import Path

# Isolated execution namespace
ns = {}
try:
    import pandas as pd
    import numpy as np
    ns["pd"] = pd
    ns["pandas"] = pd
    ns["np"] = np
    ns["numpy"] = np
    parquet_path = Path("dataset.parquet")
    if parquet_path.exists():
        _df = pd.read_parquet(parquet_path)
        ns["df"] = _df
        ns["dataset"] = _df
except Exception:
    pass

code_file = Path("__script__.py")
if not code_file.exists():
    sys.stderr.write("User script file missing.\n")
    sys.exit(1)

user_code = code_file.read_text(encoding="utf-8")

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()
old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = stdout_buf
sys.stderr = stderr_buf

exec_err = None
try:
    exec(user_code, ns)
except Exception as e:
    exec_err = f"{type(e).__name__}: {e}"
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr

stdout_str = stdout_buf.getvalue()
stderr_str = stderr_buf.getvalue()

extracted_vars = {}
for k, v in ns.items():
    if k.startswith("_") or k in ("pd", "pandas", "np", "numpy", "df", "dataset"):
        continue
    if isinstance(v, (int, float, str, bool, list, dict)):
        try:
            json.dumps(v)
            extracted_vars[k] = v
        except Exception:
            pass

payload = {
    "stdout": stdout_str,
    "stderr": stderr_str,
    "variables": extracted_vars,
    "error": exec_err,
}

Path("__results__.json").write_text(json.dumps(payload), encoding="utf-8")

if exec_err:
    sys.stderr.write(exec_err + "\n")
    sys.exit(1)
"""


class PythonExecTool(BaseTool):
    """Executes validated Python code in an isolated subprocess with access to task datasets."""

    name = "python_exec"
    description = (
        "Executes Python code in an isolated subprocess (python -I) with access to dataset.parquet, pandas, and numpy. "
        "Captures stdout, stderr, and extracted output variables."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute inside the sandbox"},
        },
        "required": ["code"],
    }

    def __init__(
        self,
        context: Optional[TaskContext] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.context = context
        self.timeout_seconds = timeout_seconds

    def _validate_ast(self, code_str: str) -> None:
        tree = ast.parse(code_str)
        validator = SafeCodeValidator()
        validator.visit(tree)

    def _clean_environment(self) -> Dict[str, str]:
        """Provides a sanitized environment containing no API keys or platform secrets."""
        clean_env: Dict[str, str] = {}
        allowed_keys = {"SYSTEMROOT", "SystemDrive", "PATH", "TEMP", "TMP"}
        for k, v in os.environ.items():
            if k in allowed_keys and not k.endswith("_API_KEY") and not k.startswith("AEGIS_"):
                clean_env[k] = v
        return clean_env

    def _run_subprocess(self, code: str, workdir: Path, timeout: float) -> Dict[str, Any]:
        runner_file = workdir / "__runner__.py"
        script_file = workdir / "__script__.py"
        results_file = workdir / "__results__.json"

        runner_file.write_text(_RUNNER_TEMPLATE, encoding="utf-8")
        script_file.write_text(code, encoding="utf-8")

        def _posix_limits():
            if sys.platform != "win32":
                try:
                    import resource
                    resource.setrlimit(resource.RLIMIT_CPU, (int(timeout), int(timeout) + 1))
                    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
                except Exception:
                    pass

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner_file)],
                cwd=str(workdir),
                env=self._clean_environment(),
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=_posix_limits if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(f"Execution timed out after {timeout} seconds limit.") from e

        settings = get_settings()
        limit = settings.tool_output_limit

        stdout = ""
        stderr = ""
        variables: Dict[str, Any] = {}
        err = None

        if results_file.exists():
            try:
                data = json.loads(results_file.read_text(encoding="utf-8"))
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                variables = data.get("variables", {})
                err = data.get("error")
            except Exception:
                stdout = proc.stdout
                stderr = proc.stderr
        else:
            stdout = proc.stdout
            stderr = proc.stderr

        # Truncate output to limit if necessary
        if len(stdout) > limit:
            stdout = stdout[:limit] + f"\n... [output truncated at {limit} characters]"
        if len(stderr) > limit:
            stderr = stderr[:limit] + f"\n... [output truncated at {limit} characters]"

        if proc.returncode != 0 or err:
            error_msg = err or stderr or f"Process failed with exit code {proc.returncode}"
            raise RuntimeError(error_msg.strip())

        return {
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "variables": variables,
            "status": "success",
        }

    def _run(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        # 1. AST Validation
        self._validate_ast(code)

        settings = get_settings()
        timeout = self.timeout_seconds or settings.sandbox_timeout

        # 2. Subprocess execution
        if self.context is not None:
            return self._run_subprocess(code, self.context.workdir, timeout)

        with tempfile.TemporaryDirectory(prefix="aegis_py_exec_") as tmpdir:
            return self._run_subprocess(code, Path(tmpdir), timeout)


class PythonSandboxTool(PythonExecTool):
    """Backward compatibility wrapper for PythonSandboxTool."""

    name = "python_sandbox"
