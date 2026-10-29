"""Model pricing registry and token cost computation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    input_per_mtok_usd: float = Field(ge=0.0)
    output_per_mtok_usd: float = Field(ge=0.0)
    source_url: str
    checked_on: str


def load_pricing_registry(pricing_file: str = "aegis/config/pricing.yaml") -> Dict[str, ModelPricing]:
    """Load and validate model pricing definitions from a YAML file.

    Raises ValueError if the YAML structure or schema is invalid.
    """
    path = Path(pricing_file)
    if not path.is_file():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed pricing YAML file in '{pricing_file}': {e}") from e

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Malformed pricing YAML file in '{pricing_file}': root must be a mapping")

    registry: Dict[str, ModelPricing] = {}
    for model_id, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed pricing entry for model '{model_id}' in '{pricing_file}': entry must be a mapping")
        registry[model_id] = ModelPricing.model_validate(entry)

    return registry


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_file: str = "aegis/config/pricing.yaml",
) -> Optional[float]:
    """Calculate the estimated USD cost for token usage. Returns None if unpriced."""
    registry = load_pricing_registry(pricing_file)
    pricing = registry.get(model)
    if pricing is None:
        return None

    cost = (input_tokens * pricing.input_per_mtok_usd + output_tokens * pricing.output_per_mtok_usd) / 1_000_000.0
    return round(cost, 6)
