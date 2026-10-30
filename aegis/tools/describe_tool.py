"""Tool for inspecting columns, dtypes, null counts, stats, and preview of uploaded dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd

from aegis.tools.base import BaseTool

if TYPE_CHECKING:
    from aegis.tools.context import TaskContext


class DescribeDatasetTool(BaseTool):
    """Summarizes dataset structure, types, null counts, numeric statistics, and first 5 rows."""

    name = "describe_dataset"
    description = (
        "Returns schema and descriptive statistics of the uploaded task dataset: "
        "columns, dtypes, null counts, numeric summary statistics, and first 5 rows."
    )
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, context: Optional[TaskContext] = None):
        self.context = context

    def _run(self) -> Dict[str, Any]:
        if self.context is None or self.context.dataset is None:
            raise ValueError("No dataset loaded for this task.")

        df = self.context.dataset
        numeric_desc = df.describe().to_dict()
        formatted_stats: Dict[str, Dict[str, Any]] = {}
        for col, stats in numeric_desc.items():
            formatted_stats[str(col)] = {
                stat: (float(val) if pd.notnull(val) else None) for stat, val in stats.items()
            }

        return {
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "null_counts": {str(c): int(n) for c, n in df.isnull().sum().items()},
            "numeric_stats": formatted_stats,
            "rows": df.head(5).to_dict(orient="records"),
        }
