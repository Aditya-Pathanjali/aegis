"""Pricing registry: the shipped pricing.yaml, per-request long-context tiers, and cost summing."""

from __future__ import annotations

import pydantic
import pytest
import yaml

from aegis.llm.pricing import (
    ModelPricing,
    calculate_cost,
    clear_pricing_cache,
    load_pricing_registry,
    sum_costs,
)


def test_shipped_pricing_file_loads_and_prices_the_configured_models():
    clear_pricing_cache()
    registry = load_pricing_registry()
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"):
        assert model in registry
        assert registry[model].source_url.startswith("https://")


def _write_tiered(tmp_path):
    path = tmp_path / "pricing.yaml"
    path.write_text(
        yaml.safe_dump({
            "tiered": {
                "input_per_mtok_usd": 0.10,
                "output_per_mtok_usd": 0.50,
                "long_context_threshold_tokens": 100_000,
                "long_context_input_per_mtok_usd": 0.50,
                "long_context_output_per_mtok_usd": 2.50,
                "source_url": "https://example.com/pricing",
                "checked_on": "2026-11-02",
            }
        }),
        encoding="utf-8",
    )
    clear_pricing_cache()
    return str(path)


def test_long_context_tier_applies_per_request_above_threshold(tmp_path):
    f = _write_tiered(tmp_path)
    # At the threshold: base rates. One token over: the whole request at the higher rates.
    assert calculate_cost("tiered", 100_000, 1_000, f) == pytest.approx(0.0105)
    assert calculate_cost("tiered", 100_001, 1_000, f) == pytest.approx(0.0525)  # rounded to 6 dp


def test_summed_requests_are_not_repriced_at_the_tier(tmp_path):
    """Three 50K-token requests cost 3x a 50K request, not one 150K long-context request."""
    f = _write_tiered(tmp_path)
    per_call = [calculate_cost("tiered", 50_000, 0, f) for _ in range(3)]
    assert sum_costs(per_call) == pytest.approx(0.015)
    assert calculate_cost("tiered", 150_000, 0, f) == pytest.approx(0.075)  # what summing tokens would bill


def test_sum_costs_is_none_when_any_request_is_unpriced():
    assert sum_costs([0.01, None]) is None
    assert sum_costs([]) is None


def test_partial_long_context_tier_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        ModelPricing(
            input_per_mtok_usd=0.1,
            output_per_mtok_usd=0.5,
            long_context_threshold_tokens=100_000,
            source_url="https://example.com",
            checked_on="2026-11-02",
        )
