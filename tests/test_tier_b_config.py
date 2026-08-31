from __future__ import annotations

import json

import pytest

from serve_bench.tier_b.config import TierBConfig
from serve_bench.tier_b.launcher import cap_for_target
from serve_bench.tier_b.runner import _cost, baseline_itl_ms_from_tier_a

TOML = """
engine = "vllm"
base_url = "http://localhost"
model = "m"
tokenizer_name = "t"
target_prompt_tokens = 512
max_tokens = 128
qps = 20.0
duration_s = 120.0
ttft_slo_ms = 500.0
"""


def test_from_toml_parses_required_fields_and_defaults(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text(TOML)
    c = TierBConfig.from_toml(p)
    assert c.engine == "vllm" and c.qps == 20.0 and c.ttft_slo_ms == 500.0
    assert c.num_workers == 4
    assert c.itl_slo_ms is None  # unset => derive from the Tier A baseline
    assert c.itl_baseline_multiple == 2.0
    assert c.uds is None


def test_example_config_parses_and_derives_a_cap():
    c = TierBConfig.from_toml("examples/tier_b_vllm.toml")
    assert c.engine == "vllm"
    cap = cap_for_target(c.target_concurrency, c.num_workers)
    assert cap > c.target_concurrency / c.num_workers


# --- Tier A baseline lookup -------------------------------------------------

def _tier_a_doc(itl_p99: float | None) -> dict:
    agg: dict = {"count": 20}
    if itl_p99 is not None:
        # Deliberately distinct values so a p50/p99 mix-up cannot pass.
        agg["itl"] = {"p50_ms": 24.8, "p90_ms": 34.4, "p95_ms": 37.3,
                      "p99_ms": itl_p99, "mean_ms": 26.0, "min_ms": 16.6, "max_ms": 55.2}
    return {"run_at": "x", "aggregate": agg}


def test_baseline_is_read_from_the_itl_TAIL_not_the_median(tmp_path):
    """Smooth goodput tests the MAX interval across a response, so the threshold has to
    come from the tail. A median-derived threshold scored 1/392 on the first real run."""
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_tier_a_doc(43.6)))
    assert baseline_itl_ms_from_tier_a(p) == pytest.approx(43.6)


def test_missing_itl_summary_returns_none_rather_than_raising(tmp_path):
    """An older Tier A result is still a valid run, and the caller may have supplied an
    explicit itl_slo_ms. Whether a missing baseline is fatal is resolve_slo's call."""
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_tier_a_doc(None)))
    assert baseline_itl_ms_from_tier_a(p) is None


# --- cost -------------------------------------------------------------------


def test_cost_is_none_when_no_tokens_qualified():
    """Zero well-served tokens makes $/MTok at the SLO undefined, not free."""
    assert _cost(3.60, 10.0, 0) is None


def test_cost_matches_the_tier_a_derivation():
    # $3.60/hr = $0.001/s; 10s = $0.01; over 1M tokens => $0.01/MTok.
    assert _cost(3.60, 10.0, 1_000_000) == pytest.approx(0.01)
