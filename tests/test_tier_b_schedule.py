import statistics

import pytest

from serve_bench.tier_b.schedule import ArrivalScheduleConfig, build_schedule, shard_schedule


def make_config(qps=10.0, duration_s=10.0, seed=42, max_requests=100_000) -> ArrivalScheduleConfig:
    return ArrivalScheduleConfig(
        qps=qps, duration_s=duration_s, seed=seed, max_requests=max_requests
    )


def _gaps(offsets: list[float]) -> list[float]:
    return [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]


def test_schedule_deterministic_for_same_seed():
    # Likely cause of failure: seeding the global `random` module instead of a
    # local random.Random(seed).
    a = build_schedule(make_config(seed=42))
    b = build_schedule(make_config(seed=42))
    assert a.offsets == b.offsets
    assert len(a.offsets) > 0


def test_schedule_differs_for_different_seed():
    a = build_schedule(make_config(seed=42))
    b = build_schedule(make_config(seed=43))
    assert a.offsets != b.offsets


def test_schedule_monotonic_and_bounded():
    s = build_schedule(make_config(qps=50, duration_s=20, seed=0))
    assert all(s.offsets[i] > s.offsets[i - 1] for i in range(1, len(s.offsets)))
    assert all(0.0 < o < 20.0 for o in s.offsets)
    # No arrival is pinned at the origin — the first gap is a genuine sample.
    assert s.offsets[0] > 0.0


def test_schedule_mean_rate_matches_qps():
    # 100 QPS x 1000s => expected 100_000 arrivals, sigma = sqrt(100_000) ~= 316.
    # A +/-2000 band is ~6 sigma, so this will not flake.
    # Likely cause of failure: passing 1/qps to expovariate instead of qps, which
    # inflates the mean gap by qps^2 (mean gap 100s, ~10 arrivals total).
    s = build_schedule(make_config(qps=100, duration_s=1000, seed=7, max_requests=200_000))
    assert 98_000 < len(s.offsets) < 102_000
    assert statistics.mean(_gaps(s.offsets)) == pytest.approx(0.01, rel=0.02)


def test_schedule_gaps_are_exponential_not_uniform():
    # The coefficient of variation of an exponential distribution is exactly 1.0;
    # a fixed 1/qps spacing has CV 0.0. This is the test that catches someone
    # "simplifying" the schedule to uniform spacing — which would pass every other
    # test in this file while destroying the entire point of an open-loop tier.
    s = build_schedule(make_config(qps=10, duration_s=10_000, seed=1))
    gaps = _gaps(s.offsets)
    assert len(gaps) > 50_000
    cv = statistics.stdev(gaps) / statistics.mean(gaps)
    assert cv == pytest.approx(1.0, abs=0.05)
    # Exponential gaps are bursty: some near-zero, some many times the mean.
    assert min(gaps) < 0.1 * statistics.mean(gaps)
    assert max(gaps) > 5 * statistics.mean(gaps)


def test_schedule_truncated_at_max_requests():
    # 1000 QPS x 100s would be ~100_000 arrivals; the cap stops it at 500.
    s = build_schedule(make_config(qps=1000, duration_s=100, max_requests=500))
    assert len(s.offsets) == 500
    assert s.truncated is True


def test_schedule_not_truncated_under_cap():
    s = build_schedule(make_config(qps=10, duration_s=10, max_requests=200_000))
    assert s.truncated is False


def test_schedule_invalid_params_raise():
    for bad in [
        make_config(qps=0),
        make_config(qps=-1.0),
        make_config(duration_s=0),
        make_config(duration_s=-5.0),
        make_config(max_requests=0),
    ]:
        with pytest.raises(ValueError):
            build_schedule(bad)


def test_shard_partitions_without_overlap_or_loss():
    s = build_schedule(make_config(qps=100, duration_s=1000, seed=3))
    s.offsets = s.offsets[:100]  # trim to an exact multiple of 4

    shards = [shard_schedule(s, i, 4) for i in range(4)]
    index_sets = [{i for i, _ in shard} for shard in shards]

    assert set().union(*index_sets) == set(range(100))
    assert sum(len(x) for x in index_sets) == 100  # disjoint
    assert all(len(shard) == 25 for shard in shards)
    for shard in shards:
        offs = [o for _, o in shard]
        assert offs == sorted(offs)


def test_shard_of_one_is_identity():
    s = build_schedule(make_config(seed=11))
    assert shard_schedule(s, 0, 1) == list(enumerate(s.offsets))


def test_shard_schedule_independent_of_shard_count():
    # Merging any sharding must reproduce the single global arrival process exactly.
    # Likely cause of failure: reseeding per worker, which would make the aggregate
    # offered load depend on how many workers happen to be running.
    s = build_schedule(make_config(qps=50, duration_s=20, seed=5))
    expected = list(enumerate(s.offsets))

    for num_shards in (3, 7):
        merged = []
        for i in range(num_shards):
            merged.extend(shard_schedule(s, i, num_shards))
        merged.sort(key=lambda pair: pair[0])
        assert merged == expected


def test_shard_index_out_of_range_raises():
    s = build_schedule(make_config())
    with pytest.raises(ValueError):
        shard_schedule(s, 4, 4)
    with pytest.raises(ValueError):
        shard_schedule(s, -1, 4)
    with pytest.raises(ValueError):
        shard_schedule(s, 0, 0)
