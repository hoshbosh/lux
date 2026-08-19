import pytest

from serve_bench.stats import summarize


def test_percentiles_never_exceed_the_observed_max():
    # THE regression this module's method choice exists to prevent. With 20 samples the
    # "exclusive" default (statistics.quantiles' default) extrapolates past the ends and
    # reports p99 = 20.79ms against a slowest observed request of 20ms.
    # Likely cause of failure: dropping method="inclusive" from the quantiles call.
    values_s = [i / 1000 for i in range(1, 21)]
    s = summarize(values_s)

    assert s.max_ms == pytest.approx(20.0)
    assert s.p99_ms <= s.max_ms
    assert s.p95_ms <= s.max_ms
    assert s.p50_ms <= s.max_ms


def test_percentiles_never_fall_below_the_observed_min():
    # The same extrapolation bites at the bottom end: exclusive would report a p50 fine
    # but low percentiles below the fastest observed request.
    values_s = [i / 1000 for i in range(1, 21)]
    s = summarize(values_s)

    assert s.min_ms == pytest.approx(1.0)
    assert s.p50_ms >= s.min_ms


def test_percentiles_are_ordered():
    values_s = [i / 1000 for i in range(1, 101)]
    s = summarize(values_s)
    assert s.min_ms <= s.p50_ms <= s.p95_ms <= s.p99_ms <= s.max_ms


def test_known_percentiles_on_uniform_data():
    # 1..100ms inclusive: with inclusive interpolation p50 lands midway between the 50th
    # and 51st values, and p99 lands on the 100th.
    values_s = [i / 1000 for i in range(1, 101)]
    s = summarize(values_s)

    assert s.mean_ms == pytest.approx(50.5)
    assert s.p50_ms == pytest.approx(50.5)
    assert s.p95_ms == pytest.approx(95.05)
    assert s.p99_ms == pytest.approx(99.01)


def test_seconds_convert_to_milliseconds():
    s = summarize([1.5, 1.5])
    assert s.mean_ms == pytest.approx(1500.0)
    assert s.max_ms == pytest.approx(1500.0)


def test_single_value_returns_that_value_for_every_percentile():
    # statistics.quantiles needs >= 2 points; one sample has no distribution to speak of.
    s = summarize([0.007])
    assert s.mean_ms == s.p50_ms == s.p95_ms == s.p99_ms == s.min_ms == s.max_ms
    assert s.p99_ms == pytest.approx(7.0)


def test_empty_raises():
    with pytest.raises(ValueError, match="zero values"):
        summarize([])


def test_p90_is_ordered_between_p50_and_p95():
    from serve_bench.stats import summarize
    s = summarize([v / 1000 for v in range(1, 101)])
    assert s.min_ms <= s.p50_ms <= s.p90_ms <= s.p95_ms <= s.p99_ms <= s.max_ms


def test_p90_value():
    from serve_bench.stats import summarize
    s = summarize([v / 1000 for v in range(1, 101)])
    assert s.p90_ms == pytest.approx(90.1, abs=1.0)
