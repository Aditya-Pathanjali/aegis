"""Task-scoped data context and tool instance management."""

from __future__ import annotations

import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from aegis.tools.base import BaseTool


class TaskContext:
    """
    Encapsulates isolated per-task state:
    - Parsed dataset (DataFrame)
    - Private in-memory SQLite connection loaded with table 'dataset'
    - Temporary working directory containing 'dataset.parquet'
    - Context-bound tool instances
    """

    def __init__(
        self,
        dataset: Optional[pd.DataFrame] = None,
        task_id: Optional[str] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.dataset: Optional[pd.DataFrame] = dataset
        self._tempdir = tempfile.TemporaryDirectory(prefix=f"aegis_task_{self.task_id[:8]}_")
        self.workdir: Path = Path(self._tempdir.name)
        self.parquet_path: Optional[Path] = None

        if self.dataset is not None:
            self.parquet_path = self.workdir / "dataset.parquet"
            self.dataset.to_parquet(self.parquet_path, index=False)

        # In-memory SQLite connection isolated per task
        self.sqlite_conn: sqlite3.Connection = sqlite3.connect(":memory:", check_same_thread=False)
        if self.dataset is not None:
            self.dataset.to_sql("dataset", self.sqlite_conn, if_exists="replace", index=False)

        # Bind tools to this task context
        from aegis.tools.describe_tool import DescribeDatasetTool
        from aegis.tools.python_sandbox import PythonExecTool, PythonSandboxTool
        from aegis.tools.rag_tool import RAGRetrieverTool, RAGSearchTool
        from aegis.tools.sql_tool import SQLQueryTool

        py_tool = PythonExecTool(context=self)
        rag_tool = RAGSearchTool()

        self.tools: Dict[str, BaseTool] = {
            "rag_search": rag_tool,
            "rag_retriever": RAGRetrieverTool(),
            "describe_dataset": DescribeDatasetTool(context=self),
            "sql_query": SQLQueryTool(context=self),
            "python_exec": py_tool,
            "python_sandbox": PythonSandboxTool(context=self),
        }

    def cleanup(self) -> None:
        """Closes SQLite connection and removes temporary working directory."""
        if hasattr(self, "sqlite_conn") and self.sqlite_conn is not None:
            try:
                self.sqlite_conn.close()
            except Exception:
                pass
        if hasattr(self, "_tempdir") and self._tempdir is not None:
            try:
                self._tempdir.cleanup()
            except Exception:
                pass

    def __enter__(self) -> TaskContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
