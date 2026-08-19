import json
import math

import pytest

from serve_bench.stats import StatSummary
from serve_bench.tier_b.calibrate import (
    CalibrationConfig,
    CalibrationPoint,
    CalibrationResult,
    run_calibration,
    select_ceiling,
)
from serve_bench.tier_b.timing import ArrivalTimingStats


def make_point(concurrency: int, passed: bool, p99_ms: float = 0.0) -> CalibrationPoint:
    summary = StatSummary(
        mean_ms=p99_ms, p50_ms=p99_ms, p90_ms=p99_ms, p95_ms=p99_ms, p99_ms=p99_ms,
        min_ms=0.0, max_ms=p99_ms
    )
    return CalibrationPoint(
        target_concurrency=concurrency,
        offered_qps=float(concurrency),
        num_arrivals=1000,
        achieved_qps=float(concurrency),
        peak_in_flight=concurrency,
        mean_in_flight=float(concurrency),
        timing=ArrivalTimingStats(
            count=1000,
            lateness=summary,
            interarrival_abs_error=summary,
            interarrival_signed_error_mean_ms=0.0,
        ),
        passed=passed,
        low_sample_warning=False,
    )


def make_result(points: list[CalibrationPoint], config: CalibrationConfig | None = None) -> CalibrationResult:
    ceiling, is_lower_bound = select_ceiling(points)
    return CalibrationResult(
        config=config or CalibrationConfig(),
        points=points,
        per_worker_ceiling=ceiling,
        ceiling_is_lower_bound=is_lower_bound,
    )


# --- knee selection ---------------------------------------------------------------


def test_ceiling_is_level_before_first_failure():
    points = [make_point(c, passed=c <= 50) for c in (10, 25, 50, 100, 200)]
    assert select_ceiling(points) == (50, False)


def test_no_failure_reports_lower_bound():
    # Nothing failed, so the knee is somewhere above the swept range. The ceiling is a
    # floor on the truth, and the caller must be told so before deriving a worker count.
    points = [make_point(c, passed=True) for c in (10, 25, 50)]
    assert select_ceiling(points) == (50, True)


def test_first_level_failing_yields_no_ceiling():
    points = [make_point(c, passed=False) for c in (10, 25, 50)]
    assert select_ceiling(points) == (None, False)


def test_pass_after_fail_is_ignored():
    # Level 100 fails, then 200 passes — noise, not recovery.
    # Likely cause of failure: taking the MAXIMUM passing level (would return 200) rather
    # than the last one before the first failure. 200 is a concurrency the generator has
    # already been shown to mistime at a lower load.
    points = [
        make_point(10, passed=True),
        make_point(50, passed=True),
        make_point(100, passed=False),
        make_point(200, passed=True),
    ]
    assert select_ceiling(points) == (50, False)


def test_empty_points_yields_no_ceiling():
    assert select_ceiling([]) == (None, True)


# --- worker count derivation ------------------------------------------------------


@pytest.mark.parametrize(
    "target,ceiling,expected",
    [(256, 64, 4), (250, 64, 4), (257, 64, 5), (64, 64, 1), (1, 64, 1)],
)
def test_workers_for_is_ceil_division(target, ceiling, expected):
    result = make_result([make_point(ceiling, passed=True)])
    assert result.workers_for(target) == expected
    assert result.workers_for(target) == max(1, math.ceil(target / ceiling))


def test_workers_for_raises_without_a_ceiling():
    # Silently defaulting to 1 worker here would hand back a load generator that is known
    # to mistime, wearing the authority of a calibration experiment.
    result = make_result([make_point(10, passed=False)])
    with pytest.raises(ValueError, match="cannot derive a worker count"):
        result.workers_for(256)


# --- Little's Law -----------------------------------------------------------------


def test_qps_derives_from_littles_law():
    # 64 chunks x 20ms = 1.28s per request; holding 128 in flight needs 100 QPS.
    config = CalibrationConfig(chunks_per_request=64, itl_s=0.02)
    assert config.request_duration_s == pytest.approx(1.28)
    assert config.qps_for(128) == pytest.approx(100.0)
    # Identity form: concurrency = qps x service time.
    assert config.qps_for(128) * config.request_duration_s == pytest.approx(128)


# --- config validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"concurrency_levels": []}, "must not be empty"),
        ({"concurrency_levels": [10, 0]}, "must all be > 0"),
        ({"concurrency_levels": [-5]}, "must all be > 0"),
        ({"itl_s": 0.0}, "must both be > 0"),
        ({"chunks_per_request": 0}, "must both be > 0"),
    ],
)
def test_invalid_config_raises(kwargs, match):
    config = CalibrationConfig(duration_s=0.1, **kwargs)
    with pytest.raises(ValueError, match=match):
        run_calibration(config)


# --- serialization ----------------------------------------------------------------


def test_to_json_carries_curve_and_disclosure():
    result = make_result([make_point(10, passed=True, p99_ms=0.5), make_point(50, passed=False, p99_ms=9.0)])
    doc = json.loads(result.to_json())

    assert doc["experiment"] == "tier_b_worker_calibration"
    assert doc["per_worker_ceiling"] == 10
    assert doc["ceiling_is_lower_bound"] is False
    # The full curve survives, so the knee can be re-derived under a different threshold
    # without re-running the sweep.
    assert [p["target_concurrency"] for p in doc["points"]] == [10, 50]
    assert doc["points"][1]["interarrival_abs_error"]["p99_ms"] == pytest.approx(9.0)
    # The optimistic-bound caveat must travel with the number, not live only in a docstring.
    assert "optimistic" in doc["disclosure"].lower()
    assert doc["config"]["itl_ms"] == pytest.approx(20.0)


# --- end-to-end smoke -------------------------------------------------------------


def test_sweep_runs_end_to_end():
    # Deliberately asserts nothing about pass/fail: a real sweep's timing error depends on
    # host load and would flake. The threshold rule is covered by the pure tests above;
    # this only proves the plumbing (schedule -> shard -> run_arrivals -> stats) holds.
    config = CalibrationConfig(
        concurrency_levels=[4, 2],  # unsorted on purpose
        duration_s=0.5,
        chunks_per_request=2,
        itl_s=0.005,
        seed=7,
    )
    result = run_calibration(config)

    # Levels are swept in ascending order regardless of input order — the knee rule reads
    # the list positionally, so an unsorted sweep would corrupt it.
    assert [p.target_concurrency for p in result.points] == [2, 4]
    for point in result.points:
        assert point.num_arrivals > 0
        assert point.peak_in_flight > 0
        assert point.timing.count == point.num_arrivals
        assert point.offered_qps == pytest.approx(point.target_concurrency / 0.01)
        # Little's Law sanity: mean depth should not exceed the target by much.
        assert point.mean_in_flight <= point.target_concurrency * 2


def test_zero_arrival_level_raises():
    # 1 concurrent / 10s per request = 0.1 QPS over 0.05s => no arrivals at all.
    # A silent empty level would show up as a passing point with no data behind it.
    config = CalibrationConfig(
        concurrency_levels=[1], duration_s=0.05, chunks_per_request=1, itl_s=10.0
    )
    with pytest.raises(RuntimeError, match="zero arrivals"):
        run_calibration(config)
