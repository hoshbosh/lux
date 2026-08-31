from __future__ import annotations

from statistics import mean

import pytest

from serve_bench.metrics import Metrics
from serve_bench.tier_b.goodput import (
    REASON_ITL,
    REASON_NO_ITL,
    REASON_TTFT,
    GoodputConfig,
    compute_goodput,
    evaluate_request,
    resolve_slo,
)


def _m(ttft_ms: float, itl_ms: list[float]) -> Metrics:
    itl = [v / 1000 for v in itl_ms]
    return Metrics(
        ttft=ttft_ms / 1000,
        tpot=mean(itl[1:]) if len(itl) >= 2 else None,
        e2e=ttft_ms / 1000 + sum(itl),
        itl=itl,
        prompt_tokens=100,
        completion_tokens=len(itl) + 1,
        token_count_warning=False,
    )


def _slo(ttft_ms=200.0, itl_ms=50.0):
    return resolve_slo(GoodputConfig(ttft_slo_ms=ttft_ms, itl_slo_ms=itl_ms), None)


# --- the two gaming paradoxes ----------------------------------------------

def test_token_delay_paradox_buffering_fails_per_chunk_but_passes_on_average():
    """30 fast chunks hide 2 half-second stalls. An averaged ITL check waves this
    through; the per-chunk maximum must not."""
    itl_ms = [1.0] * 30 + [500.0, 500.0]
    m = _m(ttft_ms=50.0, itl_ms=itl_ms)
    slo = _slo(itl_ms=50.0)

    # Precondition: the averaged check really would have passed. If this ever fails the
    # test has stopped exercising the paradox.
    assert mean(itl_ms) < slo.itl_ms, "test no longer demonstrates the averaging hole"

    v = evaluate_request(m, slo)
    assert not v.passed
    assert REASON_ITL in v.reasons
    assert v.worst_itl_s == pytest.approx(0.5)


def test_request_abandonment_paradox_cannot_improve_the_fraction():
    """Shedding the requests that were going to miss must not raise goodput. The
    denominator counts backpressure drops precisely so the numerator cannot be gamed."""
    good = [_m(50.0, [10.0] * 20) for _ in range(80)]
    borderline = [_m(50.0, [10.0] * 19 + [400.0]) for _ in range(20)]
    config = GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0)

    honest = compute_goodput(good + borderline, 0, 0, 10.0, config)
    # Same offered load; the 20 doomed requests are shed before they reach the server.
    abandoning = compute_goodput(good, 0, 20, 10.0, config)

    assert honest.denominator == abandoning.denominator == 100
    assert abandoning.goodput_fraction <= honest.goodput_fraction
    assert abandoning.goodput_fraction == pytest.approx(0.8)


# --- the three conditions ---------------------------------------------------


def test_both_failures_are_both_recorded():
    v = evaluate_request(_m(500.0, [400.0] * 5), _slo())
    assert set(v.reasons) == {REASON_TTFT, REASON_ITL}


def test_a_clean_request_passes_with_no_reasons():
    v = evaluate_request(_m(50.0, [10.0] * 20), _slo())
    assert v.passed and v.reasons == []


def test_single_chunk_response_fails_rather_than_passing_vacuously():
    """No interval means no evidence for condition (c). Treating that as a pass would
    make 'emit one chunk' a winning strategy."""
    v = evaluate_request(_m(50.0, []), _slo())
    assert not v.passed
    assert v.reasons == [REASON_NO_ITL]
    assert v.worst_itl_s is None


def test_exclude_first_itl_changes_a_first_interval_stall():
    m = _m(50.0, [400.0] + [10.0] * 10)
    slo = _slo()
    assert not evaluate_request(m, slo, exclude_first_itl=False).passed
    assert evaluate_request(m, slo, exclude_first_itl=True).passed


# --- SLO resolution ---------------------------------------------------------

def test_explicit_itl_slo_overrides_the_baseline():
    slo = resolve_slo(GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=33.0), baseline_itl_ms=10.0)
    assert slo.itl_ms == 33.0 and slo.itl_source == "explicit"


def test_baseline_is_multiplied_and_the_derivation_is_disclosed():
    slo = resolve_slo(GoodputConfig(ttft_slo_ms=200.0, itl_baseline_multiple=2.0),
                      baseline_itl_ms=12.5)
    assert slo.itl_ms == pytest.approx(25.0)
    assert "12.500" in slo.itl_source and "x2.0" in slo.itl_source


def test_no_threshold_available_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="itl_slo_ms"):
        resolve_slo(GoodputConfig(ttft_slo_ms=200.0), baseline_itl_ms=None)


def test_denominator_includes_failures_and_backpressure_drops():
    r = compute_goodput([_m(50.0, [10.0] * 5)] * 10, num_failed=3,
                        num_dropped_backpressure=7, span_s=10.0,
                        config=GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.denominator == 20
    assert r.goodput_fraction == pytest.approx(0.5)


def test_num_dropped_slo_is_finally_populated():
    """Every lower layer carries this as None on purpose. This is the layer that fills it."""
    metrics = [_m(50.0, [10.0] * 5)] * 8 + [_m(900.0, [10.0] * 5)] * 2
    r = compute_goodput(metrics, 0, 0, 10.0,
                        GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.num_dropped_slo == 2
    assert r.num_dropped_slo is not None


def test_dropped_slo_counts_no_itl_evidence_that_the_failure_counters_miss():
    metrics = [_m(50.0, [10.0] * 5)] * 5 + [_m(50.0, [])] * 3
    r = compute_goodput(metrics, 0, 0, 10.0,
                        GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    # Summing the failure counters would give 0 here and undercount by 3.
    assert r.num_slo_fail_ttft + r.num_slo_fail_itl == 0
    assert r.num_dropped_slo == 3


def test_rescoring_a_saved_run_matches_scoring_it_live(tmp_path):
    """The saved-result path must go through the same compute_goodput as the live path,
    so a swept number and a measured number are comparable."""
    import json as _json

    from serve_bench.tier_b.runner import rescore_goodput

    metrics = [_m(50.0, [10.0] * 20) for _ in range(30)] + [_m(50.0, [10.0] * 19 + [400.0]) for _ in range(10)]
    cfg = GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0)
    live = compute_goodput(metrics, num_failed=2, num_dropped_backpressure=3, span_s=10.0, config=cfg)

    doc = {
        "load": {
            "summary": {"num_failed": 2, "in_window_span_s": 10.0},
            "dropped": {"backpressure": 3},
            "per_request": [
                {"ttft_ms": m.ttft * 1000, "tpot_ms": m.tpot * 1000 if m.tpot else None,
                 "e2e_ms": m.e2e * 1000, "itl_ms": [v * 1000 for v in m.itl],
                 "prompt_tokens": m.prompt_tokens, "completion_tokens": m.completion_tokens,
                 "token_count_warning": m.token_count_warning}
                for m in metrics
            ],
        }
    }
    path = tmp_path / "b.json"
    path.write_text(_json.dumps(doc))

    again = rescore_goodput(path, cfg)
    assert again.num_goodput_ok == live.num_goodput_ok == 30
    assert again.denominator == live.denominator == 45
    assert again.goodput_fraction == pytest.approx(live.goodput_fraction)


def test_rescoring_a_pre_per_request_result_says_so(tmp_path):
    import json as _json

    from serve_bench.tier_b.runner import rescore_goodput

    path = tmp_path / "old.json"
    path.write_text(_json.dumps({"load": {"summary": {}, "dropped": {}}}))
    with pytest.raises(ValueError, match="per_request"):
        rescore_goodput(path, GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
