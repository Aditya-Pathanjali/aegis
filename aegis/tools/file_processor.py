"""Dataset and document processor tool."""

from __future__ import annotations

import io
import json
from typing import Any, Dict
import pandas as pd

from aegis.tools.base import BaseTool


class DocumentProcessorTool(BaseTool):
    """Parses, inspects, and validates tabular datasets (CSV, JSON) and technical text files."""

    name = "document_processor"
    description = (
        "Parses uploaded telemetry files, CSV datasets, and JSON reports. "
        "Outputs structural schema, column distributions, null values, and summary statistics."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_content": {"type": "string", "description": "Raw string content of the file or CSV"},
            "file_type": {"type": "string", "description": "csv, json, or text", "default": "csv"},
        },
        "required": ["file_content"],
    }

    def _run(self, file_content: str, file_type: str = "csv") -> Dict[str, Any]:
        file_type = file_type.lower()
        if file_type == "csv":
            df = pd.read_csv(io.StringIO(file_content))
            return {
                "format": "csv",
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "null_counts": df.isnull().sum().to_dict(),
                "summary_stats": df.describe().to_dict() if not df.empty else {},
                "sample_rows": df.head(5).to_dict(orient="records"),
            }
        elif file_type == "json":
            data = json.loads(file_content)
            if isinstance(data, list):
                return {
                    "format": "json_array",
                    "item_count": len(data),
                    "sample": data[:3],
                }
            return {
                "format": "json_object",
                "keys": list(data.keys()),
            }
        else:
            lines = file_content.splitlines()
            return {
                "format": "text",
                "line_count": len(lines),
                "char_count": len(file_content),
                "preview": lines[:10],
            }
