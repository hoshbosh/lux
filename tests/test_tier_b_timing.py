import pytest

from serve_bench.tier_b.arrivals import Arrival
from serve_bench.tier_b.timing import summarize_timing


def make_arrival(index=0, scheduled=0.0, actual=0.0) -> Arrival:
    return Arrival(
        index=index,
        scheduled_offset_s=scheduled,
        scheduled_time_s=scheduled,
        actual_time_s=actual,
        lateness_s=actual - scheduled,
    )


def _from_pairs(scheduled: list[float], actual: list[float]) -> list[Arrival]:
    return [make_arrival(i, s, a) for i, (s, a) in enumerate(zip(scheduled, actual))]


def test_lateness_stats_exact():
    # Lateness of exactly [1, 2, 3, 4] ms => mean 2.5 ms, max 4.0 ms.
    arrivals = [make_arrival(i, 0.0, v) for i, v in enumerate([0.001, 0.002, 0.003, 0.004])]
    stats = summarize_timing(arrivals)
    assert stats.count == 4
    assert stats.lateness.mean_ms == pytest.approx(2.5)
    assert stats.lateness.max_ms == pytest.approx(4.0)
    assert stats.lateness.min_ms == pytest.approx(1.0)


def test_interarrival_error_exact():
    # Scheduled gaps [0.1, 0.1]; actual gaps [0.105, 0.105]; errors [+5ms, +5ms].
    # Likely cause of failure: computing error against the FIRST arrival (cumulative
    # drift) rather than pairwise between consecutive arrivals.
    arrivals = _from_pairs([0.0, 0.1, 0.2], [0.0, 0.105, 0.210])
    stats = summarize_timing(arrivals)
    assert stats.interarrival_abs_error.max_ms == pytest.approx(5.0, abs=1e-6)
    assert stats.interarrival_signed_error_mean_ms == pytest.approx(5.0, abs=1e-6)
    # Lateness accumulates even though each individual gap was only 5ms off.
    assert stats.lateness.max_ms == pytest.approx(10.0, abs=1e-6)


def test_interarrival_error_zero_for_perfect_timing():
    times = [i * 0.1 for i in range(10)]
    stats = summarize_timing(_from_pairs(times, times))
    assert stats.lateness.mean_ms == pytest.approx(0.0)
    assert stats.lateness.max_ms == pytest.approx(0.0)
    assert stats.interarrival_abs_error.max_ms == pytest.approx(0.0)
    assert stats.interarrival_signed_error_mean_ms == pytest.approx(0.0)


def test_signed_error_distinguishes_drift_from_jitter():
    # Symmetric jitter: gaps alternate early/late, so the signed mean cancels toward
    # zero while the absolute error does not. This is why both are reported.
    arrivals = _from_pairs([0.0, 0.1, 0.2, 0.3], [0.0, 0.105, 0.200, 0.305])
    stats = summarize_timing(arrivals)
    assert stats.interarrival_abs_error.mean_ms == pytest.approx(5.0, abs=1e-6)
    assert stats.interarrival_signed_error_mean_ms == pytest.approx(1.667, abs=0.01)


def test_summarize_empty_raises():
    with pytest.raises(ValueError, match="zero arrivals"):
        summarize_timing([])


def test_summarize_single_arrival():
    # One arrival has no pairs, so there is no gap to be wrong about.
    stats = summarize_timing([make_arrival(0, 0.0, 0.002)])
    assert stats.count == 1
    assert stats.lateness.p50_ms == pytest.approx(2.0)
    assert stats.interarrival_abs_error.max_ms == 0.0
    assert stats.interarrival_signed_error_mean_ms == 0.0
