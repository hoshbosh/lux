from __future__ import annotations

import pytest

from serve_bench.adapter.base import GenerationConfig, RequestResult
from serve_bench.tier_b.launcher import (
    AdapterSpec,
    LoadConfig,
    assemble_result,
    cap_for_target,
    rebase_outcomes,
    run_load,
)
from serve_bench.tier_b.schedule import ArrivalScheduleConfig
from serve_bench.tier_b.synthetic import SyntheticProfile
from serve_bench.tier_b.warmup import WarmupConfig
from serve_bench.tier_b.worker import MeasurementWindow, RequestOutcome, WorkerConfig, WorkerResult

GEN = GenerationConfig(max_tokens=32)


def _wcfg() -> WorkerConfig:
    return WorkerConfig(concurrency_cap=8, prompt="hi", generation=GEN,
                        window=MeasurementWindow(0.0, None))


def _outcome(index: int, start: float, dur: float, ttft: float, t0: float) -> RequestOutcome:
    s = t0 + start
    return RequestOutcome(
        index=index, queue_start_s=s, queue_wait_s=0.0, start_s=s, end_s=s + dur,
        result=RequestResult(
            t_start=s, chunk_timestamps=[s + ttft, s + ttft + 0.01],
            prompt_tokens=8, completion_tokens=2, token_count_warning=False, success=True,
        ),
        error=None,
    )


def _worker_result(t0_perf: float, t0_wall: float, outcomes: list[RequestOutcome]) -> WorkerResult:
    return WorkerResult(
        config=_wcfg(), t0_perf=t0_perf, t0_wall=t0_wall,
        dispatch_wall_s=1.0, total_wall_s=1.0, achieved_qps=1.0,
        num_dispatched=len(outcomes), num_success=len(outcomes), num_failed=0,
        num_dropped_backpressure=0, num_incomplete_at_window_end=0, num_outside_window=0,
        num_dropped_slo=None, outcomes=outcomes, labels=[], per_request_metrics=[],
        token_count_warnings=0, num_fire_exceptions=0, queue_wait=None, timing=None,
    )


def _load_config(**kw) -> LoadConfig:
    base = dict(
        schedule=ArrivalScheduleConfig(qps=10.0, duration_s=30.0, seed=1),
        prompt="hi", generation=GEN, num_workers=2, concurrency_cap=8,
        warmup=WarmupConfig(window_s=2.0, sample_interval_s=0.1, min_ttft_samples=3),
    )
    base.update(kw)
    return LoadConfig(**base)


# --- cap sizing -------------------------------------------------------------

def test_cap_for_target_covers_the_observed_poisson_peaks():
    # The B2 sweep measured these peaks for these targets on one worker.
    assert cap_for_target(200, 1) >= 242
    assert cap_for_target(500, 1) >= 598
    assert cap_for_target(100, 1) >= 131


def test_rebase_is_invariant_to_each_workers_perf_origin():
    """Two workers that started at the same wall instant but hold wildly different
    perf_counter origins must land on the same rebased timeline. This is the whole
    reason t0_wall and t0_perf are captured as a pair."""
    a = _worker_result(t0_perf=10.0, t0_wall=1000.0, outcomes=[_outcome(0, 1.0, 0.5, 0.05, 10.0)])
    b = _worker_result(t0_perf=9e6, t0_wall=1000.0, outcomes=[_outcome(1, 1.0, 0.5, 0.05, 9e6)])
    origin = 1000.0  # earliest worker START, a real wall timestamp
    assert rebase_outcomes(a, origin)[0].start_s == pytest.approx(1.0)
    assert rebase_outcomes(b, origin)[0].start_s == pytest.approx(1.0)


def _stable_worker(t0_perf: float, t0_wall: float, n: int, offset: float = 0.0) -> WorkerResult:
    return _worker_result(
        t0_perf, t0_wall,
        [_outcome(i, offset + i * 0.1, 0.3, 0.05, t0_perf) for i in range(n)],
    )


def test_assemble_merges_workers_and_applies_one_window():
    a = _stable_worker(100.0, 5000.0, 200)
    b = _stable_worker(9e6, 5000.0, 200)
    r = assemble_result([a, b], _load_config())
    assert r.num_workers == 2
    assert r.num_dispatched == 400
    assert r.num_success + r.num_outside_window + r.num_failed == 400
    assert r.num_success > 0
    assert r.window.start_offset_s == r.warmup.steady_state_offset_s


def test_pre_steady_state_requests_fall_outside_the_window():
    r = assemble_result([_stable_worker(100.0, 5000.0, 300)], _load_config())
    assert r.warmup.converged
    assert r.num_outside_window > 0, "warmup requests must be excluded, not measured"


def test_worker_start_skew_is_measured_and_reported():
    a = _stable_worker(100.0, 5000.0, 100)
    b = _stable_worker(100.0, 5000.5, 100)  # started 500ms later in wall time
    r = assemble_result([a, b], _load_config())
    assert r.worker_skew_s == pytest.approx(0.5, abs=1e-6)


def test_result_serializes_and_keeps_drop_causes_separate():
    import json
    r = assemble_result([_stable_worker(100.0, 5000.0, 200)], _load_config())
    doc = json.loads(r.to_json())
    assert doc["dropped"]["slo"] is None
    assert "backpressure" in doc["dropped"]
    assert "combined" not in doc["dropped"] and "total" not in doc["dropped"]
    assert doc["warmup"]["converged"] is True


# --- real multi-process run -------------------------------------------------

def test_run_load_spawns_workers_and_returns_a_measured_window():
    spec = AdapterSpec(
        engine="synthetic",
        synthetic_profile=SyntheticProfile(ttft_s=0.02, itl_s=0.005, num_chunks=8),
    )
    config = _load_config(
        schedule=ArrivalScheduleConfig(qps=25.0, duration_s=4.0, seed=7),
        num_workers=2,
        warmup=WarmupConfig(window_s=0.5, sample_interval_s=0.05, min_ttft_samples=3),
    )
    r = run_load(spec, config)

    assert r.num_workers == 2
    assert r.num_dispatched > 50
    assert r.num_success > 0
    assert r.num_fire_exceptions == 0
    assert r.ttft is not None and r.ttft.p50_ms > 0
    # The barrier is the whole point: workers must start within milliseconds.
    assert r.worker_skew_s < 0.25, f"start skew {r.worker_skew_s*1000:.0f}ms — barrier failed"


def test_end_to_end_load_then_goodput_flips_on_the_itl_threshold():
    """Full offline path: spawn workers against the synthetic adapter, detect warmup,
    then score goodput with a threshold above and below the known synthetic ITL."""
    from serve_bench.tier_b.goodput import GoodputConfig, compute_goodput

    spec = AdapterSpec(
        engine="synthetic",
        synthetic_profile=SyntheticProfile(ttft_s=0.02, itl_s=0.005, num_chunks=16),
    )
    config = _load_config(
        schedule=ArrivalScheduleConfig(qps=25.0, duration_s=4.0, seed=11),
        num_workers=2,
        warmup=WarmupConfig(window_s=0.5, sample_interval_s=0.05, min_ttft_samples=3),
    )
    load = run_load(spec, config)
    assert load.num_success > 0

    def score(itl_slo_ms: float):
        return compute_goodput(
            metrics=load.metrics, num_failed=load.num_failed,
            num_dropped_backpressure=load.num_dropped_backpressure,
            span_s=load.in_window_span_s,
            config=GoodputConfig(ttft_slo_ms=500.0, itl_slo_ms=itl_slo_ms),
        )

    generous = score(20.0)   # well above the synthetic 5ms ITL
    strict = score(1.0)      # well below it

    assert generous.num_goodput_ok > 0
    assert strict.num_goodput_ok == 0
    assert generous.goodput_qps > strict.goodput_qps
    # The hook every lower layer carries as None is populated at this layer, both ways.
    assert generous.num_dropped_slo == generous.num_completed - generous.num_goodput_ok
    assert strict.num_dropped_slo == strict.num_completed
