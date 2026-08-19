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


def test_unknown_keys_are_dropped_matching_tier_a(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text(TOML + '\nnot_a_field = 3\n')
    assert TierBConfig.from_toml(p).engine == "vllm"


def test_missing_required_field_raises(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text('engine = "vllm"\n')
    with pytest.raises(TypeError):
        TierBConfig.from_toml(p)


def test_concurrency_cap_is_not_a_config_field():
    """Sizing the cap by hand is how Poisson burstiness gets misreported as backpressure.
    It must be derived."""
    assert "concurrency_cap" not in {f for f in TierBConfig.__dataclass_fields__}


def test_example_config_parses_and_derives_a_cap():
    c = TierBConfig.from_toml("examples/tier_b_vllm.toml")
    assert c.engine == "vllm"
    cap = cap_for_target(c.target_concurrency, c.num_workers)
    assert cap > c.target_concurrency / c.num_workers


# --- Tier A baseline lookup -------------------------------------------------

def _tier_a_doc(itl_p50: float | None) -> dict:
    agg: dict = {"count": 20}
    if itl_p50 is not None:
        agg["itl"] = {"p50_ms": itl_p50, "p90_ms": itl_p50, "p95_ms": itl_p50,
                      "p99_ms": itl_p50, "mean_ms": itl_p50, "min_ms": 0.0, "max_ms": itl_p50}
    return {"run_at": "x", "aggregate": agg}


def test_baseline_is_read_from_pooled_itl_p50(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_tier_a_doc(12.5)))
    assert baseline_itl_ms_from_tier_a(p) == pytest.approx(12.5)


def test_missing_itl_summary_returns_none_rather_than_raising(tmp_path):
    """An older Tier A result is still a valid run, and the caller may have supplied an
    explicit itl_slo_ms. Whether a missing baseline is fatal is resolve_slo's call."""
    p = tmp_path / "a.json"
    p.write_text(json.dumps(_tier_a_doc(None)))
    assert baseline_itl_ms_from_tier_a(p) is None


# --- cost -------------------------------------------------------------------

def test_cost_is_none_when_no_price_disclosed():
    assert _cost(0.0, 10.0, 1000) is None


def test_cost_is_none_when_no_tokens_qualified():
    """Zero well-served tokens makes $/MTok at the SLO undefined, not free."""
    assert _cost(3.60, 10.0, 0) is None


def test_cost_matches_the_tier_a_derivation():
    # $3.60/hr = $0.001/s; 10s = $0.01; over 1M tokens => $0.01/MTok.
    assert _cost(3.60, 10.0, 1_000_000) == pytest.approx(0.01)


def test_cost_at_slo_is_never_cheaper_than_at_throughput():
    """Badly served tokens still cost money, so the SLO price is the higher number."""
    at_throughput = _cost(2.49, 60.0, 100_000)
    at_slo = _cost(2.49, 60.0, 60_000)
    assert at_slo > at_throughput
