"""SQL execution tool utilizing isolated in-memory relational engine."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pandas as pd

from aegis.tools.base import BaseTool


class SQLQueryTool(BaseTool):
    """Executes SQL queries on in-memory relational tables populated from datasets."""

    name = "sql_query"
    description = (
        "Executes analytical SQL queries (SELECT, aggregations, GROUP BY, window functions) "
        "against structured telemetry and engineering tables in an in-memory SQLite database."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "SQL query to execute"},
        },
        "required": ["query"],
    }

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._init_default_tables()

    def _init_default_tables(self) -> None:
        """Seeds default sensor telemetry table for queries."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_telemetry (
                timestamp TEXT,
                vibration_rms_x REAL,
                vibration_rms_y REAL,
                bearing_temperature_c REAL,
                operating_pressure_bar REAL,
                shaft_speed_rpm INT,
                status_flag TEXT
            );
        """)
        # Insert sample industrial readings
        sample_rows = [
            ("2026-03-01 00:00:00", 1.2, 1.1, 58.2, 12.1, 1495, "NORMAL"),
            ("2026-03-05 00:00:00", 1.8, 1.6, 62.4, 12.0, 1492, "NORMAL"),
            ("2026-03-10 00:00:00", 2.9, 2.7, 69.1, 11.8, 1490, "WARNING"),
            ("2026-03-15 00:00:00", 3.8, 3.5, 76.5, 11.5, 1482, "ALERT"),
            ("2026-03-20 00:00:00", 5.2, 4.9, 84.3, 11.2, 1475, "CRITICAL"),
        ]
        cursor.executemany("""
            INSERT INTO sensor_telemetry VALUES (?, ?, ?, ?, ?, ?, ?);
        """, sample_rows)
        self.conn.commit()

    def load_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Loads a pandas dataframe into a relational table."""
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)

    def _run(self, query: str) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return {
            "columns": columns,
            "row_count": len(rows),
            "rows": rows[:100],  # cap at 100 rows for response payload
            "query": query,
        }
