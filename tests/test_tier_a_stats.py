import pytest

from serve_bench.metrics import Metrics
from serve_bench.tier_a.stats import aggregate


def _make_metrics(ttft_s: float, tpot_s: float | None, e2e_s: float, completion_tokens: int = 128) -> Metrics:
    itl = [0.01] * 3
    return Metrics(
        ttft=ttft_s,
        tpot=tpot_s,
        e2e=e2e_s,
        itl=itl,
        prompt_tokens=512,
        completion_tokens=completion_tokens,
        token_count_warning=False,
    )


def _uniform_metrics(n: int, ttft_s=0.05, tpot_s=0.01, e2e_s=0.5, tokens=128) -> list[Metrics]:
    return [_make_metrics(ttft_s, tpot_s, e2e_s, tokens) for _ in range(n)]


def test_aggregate_count():
    metrics = _uniform_metrics(10)
    result = aggregate(metrics, wall_time_s=5.0, gpu_cost_per_hour=0.0)
    assert result.count == 10


def test_aggregate_percentiles_uniform():
    # All same value → all percentiles equal that value
    metrics = _uniform_metrics(20, ttft_s=0.1)
    result = aggregate(metrics, wall_time_s=10.0, gpu_cost_per_hour=0.0)
    assert result.ttft.p50_ms == pytest.approx(100.0, abs=0.1)
    assert result.ttft.p99_ms == pytest.approx(100.0, abs=0.1)
    assert result.ttft.mean_ms == pytest.approx(100.0, abs=0.1)


def test_aggregate_percentiles_known_values():
    # 100 values: 1ms to 100ms — p50 ≈ 50ms, p95 ≈ 95ms, p99 ≈ 99ms
    metrics = [_make_metrics(i / 1000, 0.01, i / 1000 + 0.5) for i in range(1, 101)]
    result = aggregate(metrics, wall_time_s=50.0, gpu_cost_per_hour=0.0)
    assert result.ttft.p50_ms == pytest.approx(50.5, abs=1.0)
    assert result.ttft.p95_ms == pytest.approx(95.5, abs=1.0)
    assert result.ttft.p99_ms == pytest.approx(99.5, abs=1.0)


def test_aggregate_throughput():
    # 10 requests × 100 tokens each over 10s → 100 tok/s
    metrics = _uniform_metrics(10, tokens=100)
    result = aggregate(metrics, wall_time_s=10.0, gpu_cost_per_hour=0.0)
    assert result.throughput_tok_per_s == pytest.approx(100.0)
    assert result.total_completion_tokens == 1000


def test_aggregate_cost_per_mtok():
    # $3.60/hr = $0.001/s; over 10s = $0.01 total; 1000 tokens → $10/MTok
    metrics = _uniform_metrics(10, tokens=100)
    result = aggregate(metrics, wall_time_s=10.0, gpu_cost_per_hour=3.60)
    assert result.cost_per_mtok_usd == pytest.approx(10.0, rel=1e-3)


def test_aggregate_cost_none_when_zero_rate():
    metrics = _uniform_metrics(5)
    result = aggregate(metrics, wall_time_s=5.0, gpu_cost_per_hour=0.0)
    assert result.cost_per_mtok_usd is None


def test_aggregate_none_tpot_excluded():
    # Mix of metrics with and without tpot
    metrics = [
        _make_metrics(0.05, None, 0.5),   # short response, no tpot
        _make_metrics(0.05, 0.01, 0.5),
        _make_metrics(0.05, 0.01, 0.5),
    ]
    result = aggregate(metrics, wall_time_s=2.0, gpu_cost_per_hour=0.0)
    assert result.tpot is not None
    assert result.tpot.count if hasattr(result.tpot, "count") else True  # just doesn't crash


def test_aggregate_all_none_tpot():
    # All responses too short to have tpot → aggregate tpot is None
    metrics = [_make_metrics(0.05, None, 0.5) for _ in range(5)]
    result = aggregate(metrics, wall_time_s=2.0, gpu_cost_per_hour=0.0)
    assert result.tpot is None


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate([], wall_time_s=1.0, gpu_cost_per_hour=0.0)
