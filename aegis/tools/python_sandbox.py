"""Sandboxed Python code execution tool with safety validation and stdout capture."""

from __future__ import annotations

import ast
import io
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd

from aegis.tools.base import BaseTool


class SecurityViolationError(Exception):
    """Raised when forbidden modules or builtins are detected in AST."""
    pass


class SafeCodeValidator(ast.NodeVisitor):
    """AST validator to prevent dangerous operations in sandboxed code."""

    FORBIDDEN_MODULES = {
        "os", "subprocess", "sys", "shutil", "socket", "pty", "commands",
        "multiprocessing", "threading", "ctypes", "winreg", "posix"
    }

    FORBIDDEN_CALLS = {
        "eval", "exec", "__import__", "compile", "open", "breakpoint", "input"
    }

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(f"Importing module '{alias.name}' is prohibited for security.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module in self.FORBIDDEN_MODULES:
                raise SecurityViolationError(f"Importing from module '{node.module}' is prohibited for security.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(f"Function call '{node.func.id}()' is prohibited inside the sandbox.")
        self.generic_visit(node)


class PythonSandboxTool(BaseTool):
    """Executes validated Python code in an isolated execution namespace with data science libraries."""

    name = "python_sandbox"
    description = (
        "Executes Python code in a safe sandbox environment. "
        "Has access to pandas, numpy, scipy, math, json. Captures stdout, return expressions, and statistics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "context_data": {"type": "object", "description": "Optional dictionary of variables or datasets to inject"},
        },
        "required": ["code"],
    }

    def __init__(self, timeout_seconds: float = 8.0):
        self.timeout_seconds = timeout_seconds

    def _validate_ast(self, code_str: str) -> None:
        tree = ast.parse(code_str)
        validator = SafeCodeValidator()
        validator.visit(tree)

    def _run(self, code: str, context_data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # 1. AST Safety Check
        self._validate_ast(code)

        # 2. Prepare Safe Environment
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base = name.split(".")[0]
            if base in SafeCodeValidator.FORBIDDEN_MODULES:
                raise SecurityViolationError(f"Importing module '{name}' is prohibited for security.")
            return __import__(name, globals, locals, fromlist, level)

        safe_globals: Dict[str, Any] = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
                "enumerate": enumerate, "float": float, "format": format, "int": int,
                "len": len, "list": list, "map": map, "max": max, "min": min,
                "print": lambda *args, **kwargs: print(*args, file=stdout_capture, **kwargs),
                "range": range, "round": round, "set": set, "sorted": sorted,
                "str": str, "sum": sum, "tuple": tuple, "zip": zip,
                "isinstance": isinstance, "issubclass": issubclass, "Exception": Exception,
                "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
                "__import__": _safe_import,
            },
            "pd": pd,
            "pandas": pd,
            "np": np,
            "numpy": np,
        }

        # Inject context variables
        local_scope: Dict[str, Any] = {}
        if context_data:
            for k, v in context_data.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    local_scope[k] = pd.DataFrame(v)
                else:
                    local_scope[k] = v

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        start_time = time.time()
        try:
            # Execute
            exec(code, safe_globals, local_scope)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()

        # Extract useful results from local_scope
        extracted_results: Dict[str, Any] = {}
        for var_name, val in local_scope.items():
            if var_name.startswith("_") or var_name in (context_data or {}):
                continue
            if isinstance(val, (int, float, str, bool, list, dict)):
                extracted_results[var_name] = val
            elif isinstance(val, pd.DataFrame):
                extracted_results[var_name] = {
                    "shape": list(val.shape),
                    "columns": list(val.columns),
                    "summary": val.describe().to_dict() if len(val) > 0 else {},
                    "preview": val.head(5).to_dict(orient="records"),
                }
            elif isinstance(val, np.ndarray):
                extracted_results[var_name] = val.tolist()

        return {
            "stdout": stdout_val.strip(),
            "stderr": stderr_val.strip(),
            "variables": extracted_results,
            "execution_duration_sec": round(time.time() - start_time, 4),
            "status": "success",
        }
