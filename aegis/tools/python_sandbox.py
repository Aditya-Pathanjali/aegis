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
from docker.types import Mount  # type: ignore[import-not-found]

import docker  # type: ignore[import-untyped]
from aegis.config import get_settings
from aegis.tools.base import BaseTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext

logger = logging.getLogger(__name__)

# Where the task's dataset directory is mounted (read-only) inside every sandbox container.
_SANDBOX_INPUT_DIR = "/input"


def sandbox_runtime_ok() -> bool:
    """True if the Docker daemon answers and the sandbox image is present."""
    try:
        client = docker.from_env(timeout=5)  # type: ignore[attr-defined]
        client.ping()
        client.images.get(get_settings().sandbox_image)
        return True
    except Exception as exc:
        logger.warning("Sandbox runtime unavailable: %s", exc)
        return False


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

import os
_expected = os.environ.get("AEGIS_DATASET_PATH")
parquet_path = Path(_expected) if _expected else None
if parquet_path is not None and not parquet_path.is_file():
    # A task dataset was mounted but is not a readable file: never run on an empty frame.
    err_msg = f"Task dataset not found in sandbox at {parquet_path}\n"
    sys.stderr.write(err_msg)
    payload = {"stdout": "", "stderr": err_msg, "variables": {}, "error": err_msg.strip()}
    sys.stdout.write(f"__AEGIS_RESULT_START__{json.dumps(payload)}__AEGIS_RESULT_END__\n")
    sys.stdout.flush()
    sys.exit(1)

if parquet_path is not None:
    # Agent code reads 'dataset.parquet' from the cwd; the link target stays read-only.
    try:
        Path("/workspace/dataset.parquet").symlink_to(parquet_path)
    except FileExistsError:
        pass
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

    @staticmethod
    def _dataset_mounts(parquet_file: Path, settings: Any) -> list:
        """Read-only mount of ONLY this task's directory at /input.

        Bind-mount paths are resolved by the Docker daemon on the host. When the worker runs in a
        container its temp paths don't exist there, and Docker silently mounts an empty directory,
        so in that deployment each task dir lives on a named volume and we mount its subpath.
        """
        if not parquet_file.is_file():
            raise RuntimeError(f"Task dataset missing at {parquet_file}; refusing to run without it.")
        task_dir = parquet_file.parent
        if settings.sandbox_task_volume:
            base = Path(settings.task_data_dir).resolve()
            try:
                subpath = task_dir.relative_to(base)
            except ValueError as e:
                raise RuntimeError(
                    f"Task dir {task_dir} is outside task_data_dir {base}; cannot mount it from "
                    f"volume '{settings.sandbox_task_volume}'."
                ) from e
            return [
                Mount(
                    target=_SANDBOX_INPUT_DIR,
                    source=settings.sandbox_task_volume,
                    type="volume",
                    read_only=True,
                    subpath=subpath.as_posix(),
                )
            ]
        return [Mount(target=_SANDBOX_INPUT_DIR, source=str(task_dir), type="bind", read_only=True)]

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

        parquet_file: Optional[Path] = None
        if self.context is not None and self.context.parquet_path is not None:
            parquet_file = Path(self.context.parquet_path).resolve()
        elif workdir is not None and (Path(workdir) / "dataset.parquet").is_file():
            parquet_file = (Path(workdir) / "dataset.parquet").resolve()
        mounts = self._dataset_mounts(parquet_file, settings) if parquet_file is not None else []
        env = {"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if parquet_file is not None:
            env["AEGIS_DATASET_PATH"] = f"{_SANDBOX_INPUT_DIR}/{parquet_file.name}"

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
                mounts=mounts,
                environment=env,
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
                raise RuntimeError(
                    f"Process was killed (OOM / memory limit of {settings.sandbox_mem_limit} exceeded)."
                )

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
