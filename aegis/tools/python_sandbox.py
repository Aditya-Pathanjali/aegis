"""Sandboxed Python code execution tool with AST security validation and Docker container isolation."""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import requests
from docker.errors import ImageNotFound  # type: ignore[import-not-found]

import docker  # type: ignore[import-untyped]
from aegis.config import get_settings
from aegis.tools.base import BaseTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext

logger = logging.getLogger(__name__)


class SecurityViolationError(Exception):
    """Raised when forbidden modules, builtins, or attribute chains are detected in AST."""
    pass


class SafeCodeValidator(ast.NodeVisitor):
    """
    AST validator to prevent dangerous operations as a first-line pre-filter.
    Note: AST filtering is defense-in-depth; the Docker container is the true security boundary.
    """

    FORBIDDEN_MODULES = {
        "os", "subprocess", "sys", "shutil", "socket", "pty", "commands",
        "multiprocessing", "threading", "ctypes", "winreg", "posix", "importlib",
        "builtins", "urllib", "requests", "http",
    }

    FORBIDDEN_CALLS = {
        "eval", "exec", "__import__", "compile", "open", "breakpoint", "input",
        "getattr", "setattr", "delattr", "vars", "globals", "locals", "dir", "type",
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

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(f"Identifier '{node.id}' is prohibited inside the sandbox.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(f"Function call '{node.func.id}()' is prohibited inside the sandbox.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr_name = node.attr
        if attr_name.startswith("_") or attr_name in {"os", "sys", "subprocess"} or attr_name in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(f"Access to attribute '{attr_name}' is prohibited for security.")
        self.generic_visit(node)


_DOCKER_RUNNER_TEMPLATE = r"""
import io
import json
import sys
import base64
from pathlib import Path

# Isolated execution namespace
ns = {}
try:
    import pandas as pd
    import numpy as np
    import scipy
    ns["pd"] = pd
    ns["pandas"] = pd
    ns["np"] = np
    ns["numpy"] = np
    ns["scipy"] = scipy
except ImportError:
    pass

parquet_path = Path("/workspace/dataset.parquet")
if not parquet_path.exists():
    parquet_path = Path("dataset.parquet")

if parquet_path.exists():
    try:
        import pandas as pd
        _df = pd.read_parquet(parquet_path)
        ns["df"] = _df
        ns["dataset"] = _df
    except Exception as e:
        err_msg = f"Failed to load dataset.parquet: {e}\n"
        sys.stderr.write(err_msg)
        payload = {
            "stdout": "",
            "stderr": err_msg,
            "variables": {},
            "error": err_msg.strip(),
        }
        sys.stdout.write(f"__AEGIS_RESULT_START__{json.dumps(payload)}__AEGIS_RESULT_END__\n")
        sys.stdout.flush()
        sys.exit(1)

try:
    user_code = base64.b64decode(sys.argv[1]).decode("utf-8")
except Exception as e:
    err_msg = f"Failed to decode user code: {e}\n"
    sys.stderr.write(err_msg)
    payload = {
        "stdout": "",
        "stderr": err_msg,
        "variables": {},
        "error": err_msg.strip(),
    }
    sys.stdout.write(f"__AEGIS_RESULT_START__{json.dumps(payload)}__AEGIS_RESULT_END__\n")
    sys.stdout.flush()
    sys.exit(1)

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
    if k.startswith("_") or k in ("pd", "pandas", "np", "numpy", "scipy", "df", "dataset"):
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

sys.stdout.write(f"__AEGIS_RESULT_START__{json.dumps(payload)}__AEGIS_RESULT_END__\n")
sys.stdout.flush()

if exec_err:
    sys.stderr.write(exec_err + "\n")
    sys.stderr.flush()
    sys.exit(1)
"""


class PythonExecTool(BaseTool):
    """Executes validated Python code in an isolated Docker container with access to task datasets."""

    name = "python_exec"
    description = (
        "Executes Python code in a locked-down disposable Docker container with access to dataset.parquet, "
        "pandas, numpy, and scipy. Zero network access, read-only rootfs, non-root user, dropped capabilities, "
        "and strict resource limits. Captures stdout, stderr, and output variables."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute inside the container sandbox"},
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

    def _get_docker_client(self) -> Any:
        """Connects to the Docker daemon and validates availability. Fail closed if unavailable."""
        try:
            client = docker.from_env()  # type: ignore[attr-defined]
            if not client.ping():
                raise RuntimeError("Docker daemon responded with non-ping status.")
            return client
        except Exception as e:
            raise RuntimeError(
                f"Docker sandbox runtime unavailable: {e}. "
                "Python code execution requires an active Docker daemon and cannot fall back to subprocess."
            ) from e

    def _run_container(
        self,
        code: str,
        workdir: Optional[Path] = None,
        timeout: Optional[float] = None,
        bypass_ast: bool = False,
    ) -> Dict[str, Any]:
        """Runs the code in a new, locked-down Docker container with fail-closed security guarantees."""
        if not bypass_ast:
            self._validate_ast(code)

        settings = get_settings()
        actual_timeout = timeout or self.timeout_seconds or settings.sandbox_timeout
        client = self._get_docker_client()

        # Check sandbox image exists
        image_tag = settings.sandbox_image
        try:
            client.images.get(image_tag)
        except ImageNotFound:
            raise RuntimeError(
                f"Docker sandbox image '{image_tag}' not found. "
                f"Please build it using: docker build -f docker/sandbox.Dockerfile -t {image_tag} ."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to inspect Docker sandbox image '{image_tag}': {e}") from e

        # Prepare dataset volume mount if available
        volumes: Dict[str, Dict[str, str]] = {}
        parquet_file: Optional[Path] = None
        if self.context is not None and getattr(self.context, "parquet_path", None) is not None:
            p_path = getattr(self.context, "parquet_path")
            if p_path is not None:
                candidate = Path(p_path).resolve()
                if candidate.exists():
                    parquet_file = candidate
        elif self.context is not None and getattr(self.context, "workdir", None) is not None:
            candidate = (Path(self.context.workdir) / "dataset.parquet").resolve()
            if candidate.exists():
                parquet_file = candidate
        elif workdir is not None:
            candidate = (Path(workdir) / "dataset.parquet").resolve()
            if candidate.exists():
                parquet_file = candidate

        if parquet_file and parquet_file.exists():
            volumes[str(parquet_file)] = {
                "bind": "/workspace/dataset.parquet",
                "mode": "ro",
            }

        b64_code = base64.b64encode(code.encode("utf-8")).decode("ascii")
        container = None

        try:
            container = client.containers.create(
                image=image_tag,
                command=["python3", "-c", _DOCKER_RUNNER_TEMPLATE, b64_code],
                network_mode="none",
                read_only=True,
                tmpfs={
                    "/workspace": "rw,noexec,nosuid,size=64m,uid=1000,gid=1000,mode=1777",
                    "/tmp": "rw,noexec,nosuid,size=64m,uid=1000,gid=1000,mode=1777",
                },
                user="1000:1000",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit=settings.sandbox_mem_limit,
                nano_cpus=int(settings.sandbox_cpus * 1_000_000_000),
                pids_limit=settings.sandbox_pids_limit,
                volumes=volumes,
                environment={"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            )

            container.start()

            try:
                res = container.wait(timeout=int(actual_timeout))
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, TimeoutError) as e:
                try:
                    container.kill()
                except Exception:
                    pass
                raise TimeoutError(f"Execution timed out after {actual_timeout} seconds limit.") from e

            status_code = res.get("StatusCode", 0) if isinstance(res, dict) else getattr(res, "StatusCode", 0)

            # Check for OOM kill
            if status_code == 137:
                raise RuntimeError("Process was killed (OOM / memory limit of 256MB exceeded).")

            stdout_raw = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr_raw = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            # Parse structured payload marker if emitted
            match = re.search(r"__AEGIS_RESULT_START__(.*?)__AEGIS_RESULT_END__", stdout_raw, re.DOTALL)
            if match:
                payload_str = match.group(1).strip()
                try:
                    data = json.loads(payload_str)
                    stdout = data.get("stdout", "")
                    stderr = data.get("stderr", "")
                    variables = data.get("variables", {})
                    err = data.get("error")
                except Exception:
                    stdout = stdout_raw
                    stderr = stderr_raw
                    variables = {}
                    err = None
            else:
                stdout = stdout_raw
                stderr = stderr_raw
                variables = {}
                err = None

            # Truncate output to limit if necessary
            limit = settings.tool_output_limit
            if len(stdout) > limit:
                stdout = stdout[:limit] + f"\n... [output truncated at {limit} characters]"
            if len(stderr) > limit:
                stderr = stderr[:limit] + f"\n... [output truncated at {limit} characters]"

            if status_code != 0 or err:
                error_msg = err or stderr or f"Process failed with exit code {status_code}"
                raise RuntimeError(error_msg.strip())

            return {
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
                "variables": variables,
                "status": "success",
            }

        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    def _run_subprocess(
        self,
        code: str,
        workdir: Optional[Path] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Backwards-compatible interface routing directly to the container sandbox.
        Zero subprocess fallback is performed.
        """
        return self._run_container(code, workdir=workdir, timeout=timeout, bypass_ast=True)

    def _run(self, code: str = "", **kwargs: Any) -> Dict[str, Any]:
        # 1. AST Validation
        self._validate_ast(code)

        settings = get_settings()
        timeout = self.timeout_seconds or settings.sandbox_timeout

        # 2. Container execution
        return self._run_container(code, timeout=timeout)
