"""Model pricing registry and token cost computation."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_PRICING_FILE = Path(__file__).resolve().parents[1] / "config" / "pricing.yaml"


class ModelPricing(BaseModel):
    input_per_mtok_usd: float = Field(ge=0.0)
    output_per_mtok_usd: float = Field(ge=0.0)
    # Optional per-request tier: a request whose prompt exceeds the threshold is billed
    # entirely at the long-context rates (e.g. Claude Haiku 5.5 above 100,000 tokens).
    long_context_threshold_tokens: Optional[int] = Field(default=None, gt=0)
    long_context_input_per_mtok_usd: Optional[float] = Field(default=None, ge=0.0)
    long_context_output_per_mtok_usd: Optional[float] = Field(default=None, ge=0.0)
    source_url: str
    checked_on: str

    @model_validator(mode="after")
    def _tier_all_or_none(self) -> "ModelPricing":
        tier = (
            self.long_context_threshold_tokens,
            self.long_context_input_per_mtok_usd,
            self.long_context_output_per_mtok_usd,
        )
        if any(v is not None for v in tier) and any(v is None for v in tier):
            raise ValueError("long_context_* fields must be set together or not at all")
        return self


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
    """USD cost of ONE request. Returns None if unpriced.

    Never pass tokens summed over several requests: tiers apply per request, so add up the
    per-request costs with sum_costs() instead.
    """
    registry = load_pricing_registry(pricing_file)
    pricing = registry.get(model)
    if pricing is None:
        return None

    in_rate, out_rate = pricing.input_per_mtok_usd, pricing.output_per_mtok_usd
    if (
        pricing.long_context_threshold_tokens is not None
        and pricing.long_context_input_per_mtok_usd is not None
        and pricing.long_context_output_per_mtok_usd is not None
        and input_tokens > pricing.long_context_threshold_tokens
    ):
        in_rate, out_rate = pricing.long_context_input_per_mtok_usd, pricing.long_context_output_per_mtok_usd

    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
    return round(cost, 6)


def sum_costs(costs: Iterable[Optional[float]]) -> Optional[float]:
    """Total of per-request costs; None if there were none or any request was unpriced."""
    values = list(costs)
    if not values or any(c is None for c in values):
        return None
    return round(sum(c for c in values if c is not None), 6)
