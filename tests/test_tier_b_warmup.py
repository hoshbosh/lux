from __future__ import annotations

import pytest

from serve_bench.adapter.base import RequestResult
from serve_bench.tier_b.warmup import (
    WarmupConfig,
    detect_steady_state,
)
from serve_bench.tier_b.worker import RequestOutcome

T0 = 1000.0  # arbitrary perf_counter origin; nothing may depend on its value


def _outcome(index: int, start_off: float, duration: float, ttft: float) -> RequestOutcome:
    start = T0 + start_off
    end = start + duration
    return RequestOutcome(
        index=index,
        queue_start_s=start,
        queue_wait_s=0.0,
        start_s=start,
        end_s=end,
        result=RequestResult(
            t_start=start,
            chunk_timestamps=[start + ttft, start + ttft + 0.01],
            prompt_tokens=10,
            completion_tokens=2,
            token_count_warning=False,
            success=True,
        ),
        error=None,
    )


def _steady_run(span_s: float, ttft_fn, rate_hz: float = 10.0, duration: float = 1.0):
    """A run at constant arrival rate; TTFT supplied per-request by ttft_fn(offset)."""
    outcomes = []
    n = int(span_s * rate_hz)
    for i in range(n):
        off = i / rate_hz
        outcomes.append(_outcome(i, off, duration, ttft_fn(off)))
    return outcomes


def test_stable_run_converges_shortly_after_first_full_window():
    outcomes = _steady_run(40.0, lambda off: 0.05)
    r = detect_steady_state(outcomes, T0, 40.0)
    assert r.converged
    assert not r.hit_cutoff
    # Cannot converge before a full trailing window exists, and should not need much
    # more than that once the in-flight ramp has left the window.
    assert 10.0 <= r.steady_state_offset_s <= 13.0
    assert r.measurement_span_s == pytest.approx(40.0 - r.steady_state_offset_s)


def test_warmup_then_stable_is_detected_after_the_warmup():
    # First 15s: TTFT alternates wildly (JIT/CUDA-graph capture). After: flat.
    def ttft(off: float) -> float:
        if off < 15.0:
            return 0.5 if int(off * 10) % 2 == 0 else 0.05
        return 0.05

    outcomes = _steady_run(60.0, ttft)
    r = detect_steady_state(outcomes, T0, 60.0)
    assert r.converged
    # Must not certify stability while the noisy stretch is still inside the window.
    assert r.steady_state_offset_s >= 25.0


def test_never_stable_hits_cutoff_and_reports_not_converged():
    outcomes = _steady_run(60.0, lambda off: 0.05 if int(off * 10) % 2 == 0 else 0.6)
    r = detect_steady_state(outcomes, T0, 60.0, WarmupConfig(hard_cutoff_s=30.0))
    assert not r.converged
    assert r.hit_cutoff
    assert r.steady_state_offset_s == 30.0


def test_not_converged_offset_is_run_end_when_run_is_shorter_than_cutoff():
    outcomes = _steady_run(20.0, lambda off: 0.05 if int(off * 10) % 2 == 0 else 0.6)
    r = detect_steady_state(outcomes, T0, 20.0, WarmupConfig(hard_cutoff_s=120.0))
    assert not r.converged
    assert not r.hit_cutoff  # the run ended first; the cutoff was never reached
    assert r.steady_state_offset_s == 20.0


def test_in_flight_plateau_alone_does_not_certify_stability():
    """The exact vLLM failure the two-condition rule exists for: arrival rate and
    duration are constant, so depth plateaus immediately, while TTFT is still falling."""
    def ttft(off: float) -> float:
        return max(0.05, 0.9 - 0.02 * off)  # decays until ~42s

    outcomes = _steady_run(60.0, ttft)
    r = detect_steady_state(outcomes, T0, 60.0)
    assert r.converged
    assert r.steady_state_offset_s > 20.0, "converged while TTFT was still trending down"


def test_dropped_arrivals_are_not_counted_as_in_flight():
    outcomes = _steady_run(30.0, lambda off: 0.05)
    dropped = RequestOutcome(
        index=9999, queue_start_s=T0 + 1.0, queue_wait_s=0.5,
        start_s=None, end_s=None, result=None, error="dropped",
    )
    baseline = detect_steady_state(outcomes, T0, 30.0)
    withdrop = detect_steady_state(outcomes + [dropped], T0, 30.0)
    assert withdrop.steady_state_offset_s == baseline.steady_state_offset_s


def test_too_few_ttft_samples_blocks_stability():
    # 1 request every 4s => fewer than min_ttft_samples in a 10s window.
    outcomes = _steady_run(60.0, lambda off: 0.05, rate_hz=0.25, duration=1.0)
    r = detect_steady_state(outcomes, T0, 60.0, WarmupConfig(min_ttft_samples=5))
    assert not r.converged
    assert all(s.ttft_cv is None for s in r.samples)


def test_samples_are_recorded_for_diagnosis():
    outcomes = _steady_run(40.0, lambda off: 0.05)
    r = detect_steady_state(outcomes, T0, 40.0)
    assert r.samples, "convergence must be shown, not just asserted"
    assert r.samples[-1].stable
    assert all(s.offset_s >= 10.0 for s in r.samples)


def test_zero_outcomes_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        detect_steady_state([], T0, 10.0)


def test_result_is_independent_of_perf_counter_origin():
    a = detect_steady_state(_steady_run(40.0, lambda off: 0.05), T0, 40.0)
    global T0_SHIFT
    outcomes = []
    for i in range(400):
        off = i / 10.0
        start = 5_000_000.0 + off
        outcomes.append(
            RequestOutcome(
                index=i, queue_start_s=start, queue_wait_s=0.0,
                start_s=start, end_s=start + 1.0,
                result=RequestResult(
                    t_start=start, chunk_timestamps=[start + 0.05, start + 0.06],
                    prompt_tokens=10, completion_tokens=2,
                    token_count_warning=False, success=True,
                ),
                error=None,
            )
        )
    b = detect_steady_state(outcomes, 5_000_000.0, 40.0)
    assert a.steady_state_offset_s == b.steady_state_offset_s


def test_growing_queue_depth_is_not_steady_state():
    """The drift gate's other half. An engine falling behind has a queue that climbs
    steadily; within any short window that climb is small next to the mean, so CV alone
    would certify it as a plateau."""
    outcomes = []
    for i in range(600):
        off = i / 10.0
        outcomes.append(_outcome(i, off, 1.0 + 0.1 * off, 0.05))  # service time grows
    r = detect_steady_state(outcomes, T0, 60.0)
    assert not (r.converged and r.steady_state_offset_s < 25.0), (
        "certified a plateau while in-flight depth was still climbing"
    )


def test_drift_is_reported_even_on_the_converged_sample():
    outcomes = _steady_run(40.0, lambda off: 0.05)
    r = detect_steady_state(outcomes, T0, 40.0)
    last = r.samples[-1]
    assert last.in_flight_drift is not None and last.ttft_drift is not None
    assert last.in_flight_drift < 0.10 and last.ttft_drift < 0.10


def test_low_concurrency_run_can_still_converge():
    """Regression: a fixed in-flight CV threshold is a hidden demand for high mean depth.
    At mean depth ~2, Poisson dispersion alone puts CV near 0.7, so an absolute 0.10
    threshold would make a perfectly steady low-QPS run never converge."""
    outcomes = _steady_run(60.0, lambda off: 0.05, rate_hz=2.0, duration=1.0)
    r = detect_steady_state(outcomes, T0, 60.0)
    assert r.converged, "steady low-concurrency run failed to converge"


def test_in_flight_dispersion_is_judged_against_the_poisson_baseline():
    outcomes = _steady_run(40.0, lambda off: 0.05)
    r = detect_steady_state(outcomes, T0, 40.0)
    s = r.samples[-1]
    assert s.in_flight_cv_expected is not None
    assert s.in_flight_cv <= 2.0 * s.in_flight_cv_expected
