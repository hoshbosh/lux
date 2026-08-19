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


def test_token_delay_paradox_also_defeats_a_p99_over_chunks():
    """Even a p99 across 200 chunks tolerates a single stall. The max must not."""
    itl_ms = [1.0] * 199 + [900.0]
    m = _m(ttft_ms=50.0, itl_ms=itl_ms)
    slo = _slo(itl_ms=50.0)
    assert sorted(itl_ms)[int(0.99 * len(itl_ms)) - 1] < slo.itl_ms
    assert not evaluate_request(m, slo).passed


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

def test_ttft_failure_alone():
    v = evaluate_request(_m(500.0, [10.0] * 10), _slo(ttft_ms=200.0))
    assert not v.passed and v.reasons == [REASON_TTFT]


def test_itl_failure_alone():
    v = evaluate_request(_m(50.0, [10.0] * 9 + [400.0]), _slo())
    assert not v.passed and v.reasons == [REASON_ITL]


def test_both_failures_are_both_recorded():
    v = evaluate_request(_m(500.0, [400.0] * 5), _slo())
    assert set(v.reasons) == {REASON_TTFT, REASON_ITL}


def test_a_clean_request_passes_with_no_reasons():
    v = evaluate_request(_m(50.0, [10.0] * 20), _slo())
    assert v.passed and v.reasons == []


def test_threshold_is_inclusive_at_the_boundary():
    assert evaluate_request(_m(200.0, [50.0] * 5), _slo(200.0, 50.0)).passed


# --- edge cases -------------------------------------------------------------

def test_single_chunk_response_fails_rather_than_passing_vacuously():
    """No interval means no evidence for condition (c). Treating that as a pass would
    make 'emit one chunk' a winning strategy."""
    v = evaluate_request(_m(50.0, []), _slo())
    assert not v.passed
    assert v.reasons == [REASON_NO_ITL]
    assert v.worst_itl_s is None


def test_two_chunk_response_is_evaluable():
    assert evaluate_request(_m(50.0, [10.0]), _slo()).passed
    assert not evaluate_request(_m(50.0, [400.0]), _slo()).passed


def test_no_itl_evidence_is_counted_and_surfaced():
    r = compute_goodput([_m(50.0, [])] * 5, 0, 0, 10.0,
                        GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.num_no_itl_evidence == 5
    assert r.num_goodput_ok == 0
    assert r.worst_itl is None


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


def test_nonpositive_slos_are_rejected():
    with pytest.raises(ValueError):
        resolve_slo(GoodputConfig(ttft_slo_ms=0.0, itl_slo_ms=50.0), None)
    with pytest.raises(ValueError):
        resolve_slo(GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=-1.0), None)


# --- aggregate reporting ----------------------------------------------------

def test_denominator_includes_failures_and_backpressure_drops():
    r = compute_goodput([_m(50.0, [10.0] * 5)] * 10, num_failed=3,
                        num_dropped_backpressure=7, span_s=10.0,
                        config=GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.denominator == 20
    assert r.goodput_fraction == pytest.approx(0.5)


def test_goodput_qps_uses_the_measurement_span():
    r = compute_goodput([_m(50.0, [10.0] * 5)] * 40, 0, 0, 20.0,
                        GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.goodput_qps == pytest.approx(2.0)


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


def test_backpressure_and_slo_drops_are_never_summed():
    r = compute_goodput([_m(900.0, [10.0] * 5)] * 4, 0, 6, 10.0,
                        GoodputConfig(ttft_slo_ms=200.0, itl_slo_ms=50.0))
    assert r.num_dropped_backpressure == 6
    assert r.num_dropped_slo == 4
    assert not hasattr(r, "num_dropped_total")
