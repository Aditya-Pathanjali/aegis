"""SQL execution tool utilizing isolated in-memory relational engine."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from aegis.tools.base import BaseTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext

ALLOWED_SQLITE_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}


def _readonly_authorizer(action: int, arg1: Any, arg2: Any, dbname: Any, source: Any) -> int:
    """Enforces strictly read-only SELECT operations; denies DDL and DML writes."""
    if action in ALLOWED_SQLITE_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class SQLQueryTool(BaseTool):
    """Executes read-only SQL queries on in-memory relational tables populated from task datasets."""

    name = "sql_query"
    description = (
        "Executes analytical read-only SQL queries (SELECT, aggregations, GROUP BY, window functions) "
        "against the 'dataset' table in an isolated in-memory SQLite database."
    )
    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "Read-only SQL query (SELECT) to execute against the 'dataset' table",
            },
        },
        "required": ["sql"],
    }

    def __init__(self, context: Optional[TaskContext] = None):
        self.context = context
        if context is not None:
            self.conn = context.sqlite_conn
        else:
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)

        # Enforce read-only operations via authorizer
        self.conn.set_authorizer(_readonly_authorizer)

    def load_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Loads a pandas dataframe into a relational table."""
        # Temporarily disable authorizer to load data into memory
        self.conn.set_authorizer(None)
        try:
            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
        finally:
            self.conn.set_authorizer(_readonly_authorizer)

    def _run(self, sql: Optional[str] = None, query: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        sql_cmd = sql or query
        if not sql_cmd or not sql_cmd.strip():
            raise ValueError("No SQL query provided to execute.")

        cursor = self.conn.cursor()
        cursor.execute(sql_cmd)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(500)  # Capped at max 500 rows as required

        return {
            "columns": columns,
            "row_count": len(rows),
            "rows": rows,
            "query": sql_cmd,
        }
