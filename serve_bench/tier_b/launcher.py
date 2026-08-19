"""Multi-process load launcher: N worker processes over one shared arrival schedule.

Tier B is multi-process because a single asyncio event loop saturates and starts
injecting its own latency into the offered load, which corrupts exactly the tail the
benchmark exists to measure. This module owns the three problems that only appear once
there is more than one process.

**1. Adapters are not picklable.** A live ``httpx.AsyncClient`` and a tokenizer cannot
cross a process boundary, so children receive an ``AdapterSpec`` — a plain description —
and construct the adapter themselves. That also puts tokenizer load time inside the
child, before the start barrier, where it cannot skew the schedule.

**2. Workers must start together.** Each worker reads its own ``t0_perf`` when it begins
dispatching. Spawning is not instantaneous and tokenizer loads vary by hundreds of
milliseconds, so without synchronization worker 3 would begin its schedule after worker 1
and the union of their arrivals would no longer be the Poisson process that was
scheduled. A ``multiprocessing.Barrier`` across the children (the parent is not a party
to it, so parent-side bookkeeping cannot become the sync point) releases them together
once every adapter is built.

**3. perf_counter is not comparable across processes.** Every worker returns the
``t0_wall``/``t0_perf`` pair read at the same instant, and outcomes are rebased onto a
common wall-clock origin with ``wall = t0_wall + (perf - t0_perf)`` before anything is
merged. Per-request metrics need no rebasing — TTFT and ITL are differences within a
single request, so they are invariant to the origin.

**Workers measure everything; the window is applied here.** Children run with a wide-open
``MeasurementWindow`` and return raw outcomes. Warmup detection needs the merged
cross-worker in-flight series, which no single child can see, so the real window is
computed in the parent and every outcome is re-classified under it. This is the reason
``worker.py`` keeps its raw outcomes rather than only its counts.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field, replace

from ..adapter.base import EngineAdapter, GenerationConfig
from ..metrics import Metrics, compute_metrics
from ..stats import StatSummary, summarize
from .schedule import ArrivalScheduleConfig, build_schedule, shard_schedule
from .synthetic import SyntheticAdapter, SyntheticProfile
from .warmup import WarmupConfig, WarmupResult, detect_steady_state
from .worker import (
    DROPPED_BACKPRESSURE,
    FAILED,
    INCOMPLETE_AT_WINDOW_END,
    OUTSIDE_WINDOW,
    SUCCESS,
    MeasurementWindow,
    RequestOutcome,
    WorkerConfig,
    WorkerResult,
    classify_outcome,
    run_worker,
)

logger = logging.getLogger(__name__)


@dataclass
class AdapterSpec:
    """Everything a child needs to build its own adapter. Must stay picklable."""

    engine: str  # "vllm" | "llamacpp" | "synthetic"
    base_url: str = ""
    model: str = ""
    tokenizer_name: str = ""
    uds: str | None = None
    synthetic_profile: SyntheticProfile | None = None
    seed: int = 0


def build_adapter(spec: AdapterSpec, worker_id: int) -> EngineAdapter:
    """Construct an adapter inside the child process.

    ``worker_id`` offsets the synthetic seed so N workers are not correlated replicas of
    one another; real adapters ignore it.
    """
    if spec.engine == "synthetic":
        return SyntheticAdapter(spec.synthetic_profile, seed=spec.seed + worker_id)

    from transformers import AutoTokenizer  # imported here: children only, and slow

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_name)
    if spec.engine == "vllm":
        from ..adapter.vllm import VLLMAdapter

        return VLLMAdapter(spec.base_url, spec.model, tokenizer, uds=spec.uds)
    if spec.engine == "llamacpp":
        from ..adapter.llamacpp import LlamaCppAdapter

        return LlamaCppAdapter(spec.base_url, spec.model, tokenizer, uds=spec.uds)
    raise ValueError(f"Unknown engine: {spec.engine!r}")


@dataclass
class LoadConfig:
    schedule: ArrivalScheduleConfig
    prompt: str
    generation: GenerationConfig
    num_workers: int = 4
    # Per worker. Sized at the burst peak, not the mean: Poisson arrivals overshoot mean
    # concurrency by roughly 3*sqrt(mean), and a cap set at the mean converts ordinary
    # burstiness into fake backpressure drops. See `cap_for_target`.
    concurrency_cap: int = 64
    max_queue_wait_s: float | None = None
    warmup: WarmupConfig = field(default_factory=WarmupConfig)


def cap_for_target(target_concurrency: int, num_workers: int, sigmas: float = 5.0) -> int:
    """Per-worker semaphore size for a target *mean* concurrency.

    In-flight depth under Poisson arrivals is Poisson-distributed, so its standard
    deviation is sqrt(mean). The obvious choice of 3 sigma is wrong here: the quantity
    being covered is the run's *maximum* depth, an extreme-value statistic over thousands
    of samples, and a longer run draws a higher max from the same distribution. The B2
    sweep bears that out — 3 sigma fits the mid-range (target 200 peaked at 242, against
    a 3-sigma prediction of 242) but under-covers the top (target 500 peaked at 598,
    3-sigma predicts 567).

    5 sigma is the default because the two errors are not symmetric. An oversized cap
    merely fails to bind, costing nothing but a little memory; an undersized one clips
    ordinary burstiness, reports the clipping as backpressure drops, and truncates the
    latency tail the benchmark exists to measure.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    per_worker = target_concurrency / num_workers
    return max(1, int(per_worker + sigmas * (per_worker ** 0.5)) + 1)


@dataclass
class TierBLoadResult:
    config: LoadConfig
    warmup: WarmupResult
    window: MeasurementWindow
    num_workers: int
    origin_wall: float
    worker_skew_s: float  # spread of worker start instants; barrier quality

    num_dispatched: int
    num_success: int
    num_failed: int
    num_dropped_backpressure: int
    num_incomplete_at_window_end: int
    num_outside_window: int
    num_dropped_slo: int | None  # always None here; the goodput layer decides SLOs
    num_fire_exceptions: int
    token_count_warnings: int

    in_window_span_s: float
    achieved_qps_in_window: float
    metrics: list[Metrics]
    ttft: StatSummary | None
    tpot: StatSummary | None
    e2e: StatSummary | None

    def to_json(self) -> str:
        import json

        def _s(x: StatSummary | None) -> dict | None:
            if x is None:
                return None
            return {
                "mean_ms": x.mean_ms, "p50_ms": x.p50_ms, "p90_ms": x.p90_ms,
                "p95_ms": x.p95_ms,
                "p99_ms": x.p99_ms, "min_ms": x.min_ms, "max_ms": x.max_ms,
            }

        return json.dumps(
            {
                "component": "tier_b_load",
                "config": {
                    "qps": self.config.schedule.qps,
                    "duration_s": self.config.schedule.duration_s,
                    "seed": self.config.schedule.seed,
                    "num_workers": self.num_workers,
                    "concurrency_cap_per_worker": self.config.concurrency_cap,
                    "max_queue_wait_ms": (
                        self.config.max_queue_wait_s * 1000
                        if self.config.max_queue_wait_s is not None else None
                    ),
                },
                "warmup": {
                    "converged": self.warmup.converged,
                    "hit_cutoff": self.warmup.hit_cutoff,
                    "steady_state_offset_s": self.warmup.steady_state_offset_s,
                    "measurement_span_s": self.warmup.measurement_span_s,
                    "note": (
                        "Detected post-hoc from merged per-request start/end times. "
                        "converged=false means the offset is the hard cutoff, not a "
                        "detection, and the run may not reflect steady state."
                    ),
                },
                "window": {
                    "start_offset_s": self.window.start_offset_s,
                    "end_offset_s": self.window.end_offset_s,
                },
                "workers": {"count": self.num_workers, "start_skew_ms": self.worker_skew_s * 1000},
                "summary": {
                    "num_dispatched": self.num_dispatched,
                    "num_success": self.num_success,
                    "num_failed": self.num_failed,
                    "num_incomplete_at_window_end": self.num_incomplete_at_window_end,
                    "num_outside_window": self.num_outside_window,
                    "num_fire_exceptions": self.num_fire_exceptions,
                    "token_count_warnings": self.token_count_warnings,
                    "in_window_span_s": self.in_window_span_s,
                    "achieved_qps_in_window": self.achieved_qps_in_window,
                },
                "dropped": {
                    "backpressure": self.num_dropped_backpressure,
                    "slo": self.num_dropped_slo,
                    "note": (
                        "Backpressure drops never reached the server. SLO drops are "
                        "decided by the goodput layer and are not evaluated here."
                    ),
                },
                "latency": {"ttft": _s(self.ttft), "tpot": _s(self.tpot), "e2e": _s(self.e2e)},
            },
            indent=2,
        )


def rebase_outcomes(result: WorkerResult, origin_wall: float) -> list[RequestOutcome]:
    """Translate one worker's perf-domain outcomes onto a shared wall-clock origin.

    ``wall = t0_wall + (perf - t0_perf)``, then expressed as seconds since
    ``origin_wall``. The embedded ``RequestResult`` is left untouched: every metric
    derived from it is a difference within one request and so is origin-invariant, and
    rewriting those timestamps would only create a second, redundant source of truth.
    """
    delta = result.t0_wall - result.t0_perf - origin_wall
    return [
        replace(
            o,
            queue_start_s=o.queue_start_s + delta,
            start_s=o.start_s + delta if o.start_s is not None else None,
            end_s=o.end_s + delta if o.end_s is not None else None,
        )
        for o in result.outcomes
    ]


def assemble_result(
    worker_results: list[WorkerResult], config: LoadConfig
) -> TierBLoadResult:
    """Merge worker results, detect warmup, and re-classify under the real window.

    Pure: no processes, no clock reads. The multi-process path and any single-process or
    replay path produce identical numbers through this function.
    """
    if not worker_results:
        raise ValueError("Cannot assemble a result from zero workers")

    # min of t0_wall, NOT of (t0_wall - t0_perf). The latter is the wall time that would
    # correspond to perf_counter zero — an arbitrary per-process anchor, not an instant
    # any worker started at. Using it puts every offset on a different, meaningless base.
    origin_wall = min(r.t0_wall for r in worker_results)
    worker_skew_s = max(r.t0_wall for r in worker_results) - origin_wall
    if worker_skew_s > 0.25:
        logger.warning(
            "Worker start skew is %.0fms. The union of worker schedules is no longer the "
            "Poisson process that was scheduled; suspect a slow adapter build outside the "
            "start barrier.",
            worker_skew_s * 1000,
        )

    merged: list[RequestOutcome] = []
    for r in worker_results:
        merged.extend(rebase_outcomes(r, origin_wall))
    merged.sort(key=lambda o: o.queue_start_s)

    ends = [o.end_s for o in merged if o.end_s is not None]
    run_span_s = max(ends) if ends else max(o.queue_start_s for o in merged)

    warmup = detect_steady_state(merged, 0.0, run_span_s, config.warmup)
    window = MeasurementWindow(
        start_offset_s=warmup.steady_state_offset_s, end_offset_s=run_span_s
    )

    counts = {SUCCESS: 0, FAILED: 0, DROPPED_BACKPRESSURE: 0,
              INCOMPLETE_AT_WINDOW_END: 0, OUTSIDE_WINDOW: 0}
    metrics: list[Metrics] = []
    token_count_warnings = 0

    for o in merged:
        label = classify_outcome(o, 0.0, window)
        if label == SUCCESS:
            assert o.result is not None
            try:
                m = compute_metrics(o.result)
            except ValueError:
                # Same demotion rule as the worker: a success with no chunks is not a
                # zero-latency measurement.
                counts[FAILED] += 1
                continue
            metrics.append(m)
            if m.token_count_warning:
                token_count_warnings += 1
        counts[label] += 1

    in_window_span_s = max(0.0, run_span_s - window.start_offset_s)
    completed = counts[SUCCESS] + counts[FAILED]

    tpots = [m.tpot for m in metrics if m.tpot is not None]
    return TierBLoadResult(
        config=config,
        warmup=warmup,
        window=window,
        num_workers=len(worker_results),
        origin_wall=origin_wall,
        worker_skew_s=worker_skew_s,
        num_dispatched=sum(r.num_dispatched for r in worker_results),
        num_success=counts[SUCCESS],
        num_failed=counts[FAILED],
        num_dropped_backpressure=counts[DROPPED_BACKPRESSURE],
        num_incomplete_at_window_end=counts[INCOMPLETE_AT_WINDOW_END],
        num_outside_window=counts[OUTSIDE_WINDOW],
        num_dropped_slo=None,
        num_fire_exceptions=sum(r.num_fire_exceptions for r in worker_results),
        token_count_warnings=token_count_warnings,
        in_window_span_s=in_window_span_s,
        achieved_qps_in_window=completed / in_window_span_s if in_window_span_s > 0 else 0.0,
        metrics=metrics,
        ttft=summarize([m.ttft for m in metrics]) if metrics else None,
        tpot=summarize(tpots) if tpots else None,
        e2e=summarize([m.e2e for m in metrics]) if metrics else None,
    )


def _worker_entry(spec, offsets, worker_config, barrier, queue, worker_id) -> None:
    """Child process entry point. Module-level so ``spawn`` can import it."""
    import asyncio

    try:
        adapter = build_adapter(spec, worker_id)
        # Every adapter (and tokenizer) is built before any worker starts dispatching.
        barrier.wait()
        result = asyncio.run(run_worker(offsets, adapter, worker_config))
        queue.put((worker_id, result, None))
    except Exception as exc:  # noqa: BLE001 - must reach the parent, not just the log
        try:
            barrier.abort()  # release siblings rather than hanging them at the barrier
        except Exception:
            pass
        queue.put((worker_id, None, f"{type(exc).__name__}: {exc}"))


def run_load(spec: AdapterSpec, config: LoadConfig) -> TierBLoadResult:
    """Run the full multi-process Tier B load and assemble one result."""
    if config.num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    schedule = build_schedule(config.schedule)
    logger.info(
        "Tier B: %d arrivals at %.1f QPS over %.0fs across %d workers (cap %d each)",
        len(schedule.offsets), config.schedule.qps, config.schedule.duration_s,
        config.num_workers, config.concurrency_cap,
    )

    # Wide-open window in the children; the real one is computed in assemble_result once
    # the merged cross-worker in-flight series exists.
    worker_config = WorkerConfig(
        concurrency_cap=config.concurrency_cap,
        prompt=config.prompt,
        generation=config.generation,
        window=MeasurementWindow(start_offset_s=0.0, end_offset_s=None),
        max_queue_wait_s=config.max_queue_wait_s,
    )

    # "spawn", not the Linux default "fork": forking a process that may already hold an
    # event loop or HTTP client is a documented source of deadlocks, and spawn matches
    # macOS behaviour so the same code path is exercised on both dev and benchmark hosts.
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(config.num_workers)
    queue: mp.Queue = ctx.Queue()

    procs = []
    for wid in range(config.num_workers):
        offsets = shard_schedule(schedule, wid, config.num_workers)
        p = ctx.Process(
            target=_worker_entry,
            args=(spec, offsets, worker_config, barrier, queue, wid),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Drain before joining: a child blocks on a full pipe until its result is read, so
    # joining first would deadlock on any run big enough to matter.
    results: list[WorkerResult] = []
    errors: list[str] = []
    for _ in range(config.num_workers):
        wid, result, error = queue.get()
        if error is not None:
            errors.append(f"worker {wid}: {error}")
        else:
            results.append(result)

    for p in procs:
        p.join()

    if errors:
        raise RuntimeError("Tier B workers failed:\n  " + "\n  ".join(errors))

    return assemble_result(results, config)
