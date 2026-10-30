"""Model pricing registry and token cost computation."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field

DEFAULT_PRICING_FILE = Path(__file__).resolve().parents[1] / "config" / "pricing.yaml"


class ModelPricing(BaseModel):
    input_per_mtok_usd: float = Field(ge=0.0)
    output_per_mtok_usd: float = Field(ge=0.0)
    source_url: str
    checked_on: str


@functools.lru_cache(maxsize=32)
def _load_pricing_cached(file_path: str) -> Dict[str, ModelPricing]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pricing file not found at: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed pricing YAML file in '{file_path}': {e}") from e

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Malformed pricing YAML file in '{file_path}': root must be a mapping")

    registry: Dict[str, ModelPricing] = {}
    for model_id, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed pricing entry for model '{model_id}' in '{file_path}': entry must be a mapping")
        registry[model_id] = ModelPricing.model_validate(entry)

    return registry


def clear_pricing_cache() -> None:
    """Clear cached pricing registries."""
    _load_pricing_cached.cache_clear()


def resolve_pricing_path(pricing_file: Optional[str] = None) -> Path:
    """Resolve pricing file path relative to package/project root or absolute path."""
    if not pricing_file:
        return DEFAULT_PRICING_FILE

    p = Path(pricing_file)
    if p.is_file():
        return p.resolve()

    # If relative, check relative to project root (parents[2]) or package (parents[1])
    repo_root_rel = (Path(__file__).resolve().parents[2] / p).resolve()
    if repo_root_rel.is_file():
        return repo_root_rel

    pkg_rel = (Path(__file__).resolve().parents[1] / p).resolve()
    if pkg_rel.is_file():
        return pkg_rel

    return p.resolve()


def load_pricing_registry(pricing_file: Optional[str] = None) -> Dict[str, ModelPricing]:
    """Load and validate model pricing definitions from a YAML file.

    Raises FileNotFoundError if the file does not exist.
    Raises ValueError if the YAML structure or schema is invalid.
    """
    path = resolve_pricing_path(pricing_file)
    return _load_pricing_cached(str(path))


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing_file: Optional[str] = None,
) -> Optional[float]:
    """Calculate the estimated USD cost for token usage. Returns None if unpriced."""
    registry = load_pricing_registry(pricing_file)
    pricing = registry.get(model)
    if pricing is None:
        return None

    cost = (input_tokens * pricing.input_per_mtok_usd + output_tokens * pricing.output_per_mtok_usd) / 1_000_000.0
    return round(cost, 6)
