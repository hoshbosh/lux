"""One asyncio worker: sharded arrival schedule in, real adapter calls out.

This is the layer that turns the timing-only scheduler in ``arrivals.py`` into actual
load. It owns exactly three concerns, all of which ``arrivals.py`` deliberately refused:
backpressure, measurement-window filtering, and per-request metric collection.

Timing invariants are inherited unchanged from ``arrivals.py`` and must stay that way:

1. Every timestamp in this module comes from ``time.perf_counter()``. ``time.time()``
   does not appear anywhere in the timing path — it can step under NTP mid-run and is
   not comparable with perf_counter. The single wall-clock read of the run lives in
   ``run_arrivals`` and reaches the caller as ``WorkerResult.t0_wall``, paired with
   ``t0_perf`` from the same instant.

2. ``perf_counter`` is not comparable across processes, so the measurement window is
   expressed as OFFSETS from the schedule origin, never as absolute timestamps. Every
   worker driving the same schedule shares those offsets, and a future multi-process
   layer maps each worker's perf-domain values onto a common epoch with
   ``wall = t0_wall + (perf - t0_perf)``.

**Window filtering happens after the run, not during it.** Requests outside the window
are still issued — the offered arrival process must be continuous or the load the server
sees is no longer the Poisson process that was scheduled — and are discarded when the
outcomes are classified. That also keeps ``classify_outcome`` a pure function, testable
without a clock.

**Disclosure — what this layer does NOT decide.** Drops here mean exactly one thing:
backpressure. Whether a *completed* request met its TTFT/ITL SLO is a smooth-goodput
question decided one layer up, so ``WorkerResult.num_dropped_slo`` is ``None`` rather
than ``0``; reporting zero would claim an SLO evaluation that never happened. The two
drop causes are never summed into a single "dropped" number in any output.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from ..adapter.base import EngineAdapter, GenerationConfig, RequestResult
from ..metrics import Metrics, compute_metrics
from ..stats import StatSummary, summarize
from .arrivals import Arrival, run_arrivals
from .timing import ArrivalTimingStats

logger = logging.getLogger(__name__)

# Outcome classes. Deliberately five distinct labels rather than success/failure:
# collapsing any two of these is how a load generator flatters itself.
SUCCESS = "success"
FAILED = "failed"
DROPPED_BACKPRESSURE = "dropped_backpressure"
INCOMPLETE_AT_WINDOW_END = "incomplete_at_window_end"
OUTSIDE_WINDOW = "outside_window"


@dataclass
class MeasurementWindow:
    """Measurement bounds as offsets (seconds) from the schedule origin.

    Offsets, not timestamps: see invariant 2 in the module docstring.

    ``start_offset_s`` is the seam the warmup-until-stable gate will fill — that gate
    decides *when* steady state began; this dataclass only carries the answer.
    ``end_offset_s`` of None means "no upper bound", which is only correct for a run
    whose schedule ends before anything is expected to be trimmed.
    """

    start_offset_s: float = 0.0
    end_offset_s: float | None = None


@dataclass
class WorkerConfig:
    concurrency_cap: int  # asyncio.Semaphore permits held by this worker
    prompt: str
    generation: GenerationConfig
    window: MeasurementWindow = field(default_factory=MeasurementWindow)
    # None: an arrival waits for a permit as long as it takes. Honest about offered load
    # (nothing is ever thrown away) but the wait queue is unbounded, and a request issued
    # 30s after its scheduled arrival measures queue delay rather than the server. A
    # finite value converts that tail into an explicit, counted drop instead.
    max_queue_wait_s: float | None = None


@dataclass
class RequestOutcome:
    """Raw record of one arrival, before any window rule is applied.

    Kept separate from the classification so a run can be re-classified under a different
    window without being re-run — the same reason ``run_calibration`` records the whole
    sweep rather than only the knee.
    """

    index: int  # global index in the unsharded schedule
    queue_start_s: float  # perf: fire callback entered, permit not yet held
    queue_wait_s: float
    start_s: float | None  # perf: request actually issued. None iff dropped
    end_s: float | None  # perf: adapter returned. None iff dropped
    result: RequestResult | None
    error: str | None


@dataclass
class WorkerResult:
    config: WorkerConfig
    t0_perf: float
    t0_wall: float  # paired with t0_perf; the only wall-clock read of the run
    dispatch_wall_s: float
    total_wall_s: float
    achieved_qps: float  # offered rate, measured over dispatch only
    num_dispatched: int

    # Window-filtered counts. These sum to num_dispatched, less any arrival whose fire
    # callback raised before it could record an outcome (num_fire_exceptions).
    num_success: int
    num_failed: int
    num_dropped_backpressure: int
    # Started inside the window, completed after it. NOT a drop and NOT a success — the
    # server did nothing wrong, the observation is simply censored by the window edge.
    num_incomplete_at_window_end: int
    num_outside_window: int
    # Always None at this layer: no SLO is evaluated here. See the module docstring.
    num_dropped_slo: int | None

    outcomes: list[RequestOutcome]
    labels: list[str]  # positionally parallel to outcomes
    per_request_metrics: list[Metrics]  # successes only, in dispatch order
    token_count_warnings: int
    num_fire_exceptions: int
    queue_wait: StatSummary | None  # over issued requests; None if none were issued
    timing: ArrivalTimingStats | None  # generator self-accuracy, None if no arrivals

    def to_json(self) -> str:
        def _metrics_dict(m: Metrics) -> dict:
            return {
                "ttft_ms": m.ttft * 1000,
                "tpot_ms": m.tpot * 1000 if m.tpot is not None else None,
                "e2e_ms": m.e2e * 1000,
                "itl_ms": [v * 1000 for v in m.itl],
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "token_count_warning": m.token_count_warning,
            }

        def _summary(s: StatSummary | None) -> dict | None:
            if s is None:
                return None
            return {
                "mean_ms": s.mean_ms,
                "p50_ms": s.p50_ms,
                "p90_ms": s.p90_ms,
                "p95_ms": s.p95_ms,
                "p99_ms": s.p99_ms,
                "min_ms": s.min_ms,
                "max_ms": s.max_ms,
            }

        doc = {
            "component": "tier_b_worker",
            "config": {
                "concurrency_cap": self.config.concurrency_cap,
                "max_queue_wait_ms": (
                    self.config.max_queue_wait_s * 1000
                    if self.config.max_queue_wait_s is not None
                    else None
                ),
                "window_start_offset_s": self.config.window.start_offset_s,
                "window_end_offset_s": self.config.window.end_offset_s,
                "max_tokens": self.config.generation.max_tokens,
            },
            # Perf-domain values are worker-local; this pair is what makes them mappable
            # onto a common epoch by a multi-process aggregator.
            "epoch": {"t0_wall": self.t0_wall, "t0_perf": self.t0_perf},
            "summary": {
                "num_dispatched": self.num_dispatched,
                "num_success": self.num_success,
                "num_failed": self.num_failed,
                "num_incomplete_at_window_end": self.num_incomplete_at_window_end,
                "num_outside_window": self.num_outside_window,
                "num_fire_exceptions": self.num_fire_exceptions,
                "token_count_warnings": self.token_count_warnings,
                "dispatch_wall_s": self.dispatch_wall_s,
                "total_wall_s": self.total_wall_s,
                "achieved_qps": self.achieved_qps,
            },
            # Two causes, two fields, never a sum. `slo` is null because this layer does
            # not evaluate SLOs — that is not the same statement as zero SLO drops.
            "dropped": {
                "backpressure": self.num_dropped_backpressure,
                "slo": self.num_dropped_slo,
                "note": (
                    "Backpressure drops are arrivals that never reached the server. SLO "
                    "drops are decided by the goodput layer and are not evaluated here."
                ),
            },
            "queue_wait": _summary(self.queue_wait),
            "per_request": [_metrics_dict(m) for m in self.per_request_metrics],
        }
        return json.dumps(doc, indent=2)


def classify_outcome(outcome: RequestOutcome, t0_perf: float, window: MeasurementWindow) -> str:
    """Apply the hard AND window rule to one outcome. Pure function of its arguments.

    The rule: a request counts only if it BOTH starts and completes inside the window.
    Anything that straddles an edge is censored data, not evidence — counting a request
    that started before the window would import warmup latency into steady-state numbers,
    and counting one that finished after the window would require knowing a completion
    time the run did not observe.

    A dropped arrival has no start time, so it is placed by the moment it entered the
    wait queue; a drop during warmup belongs to warmup, not to the measurement window.
    """
    win_start = t0_perf + window.start_offset_s
    win_end = (
        t0_perf + window.end_offset_s if window.end_offset_s is not None else float("inf")
    )

    ref_start = outcome.start_s if outcome.start_s is not None else outcome.queue_start_s
    if ref_start < win_start or ref_start > win_end:
        return OUTSIDE_WINDOW
    if outcome.start_s is None:
        return DROPPED_BACKPRESSURE
    assert outcome.end_s is not None  # set together with start_s
    if outcome.end_s > win_end:
        return INCOMPLETE_AT_WINDOW_END
    if outcome.result is None or not outcome.result.success:
        return FAILED
    return SUCCESS


def _validate(config: WorkerConfig) -> None:
    if config.concurrency_cap < 1:
        raise ValueError(f"concurrency_cap must be >= 1, got {config.concurrency_cap!r}")
    if config.max_queue_wait_s is not None and config.max_queue_wait_s <= 0:
        raise ValueError(
            f"max_queue_wait_s must be > 0 or None, got {config.max_queue_wait_s!r}"
        )
    if config.window.start_offset_s < 0:
        raise ValueError(
            f"window.start_offset_s must be >= 0, got {config.window.start_offset_s!r}"
        )
    end = config.window.end_offset_s
    if end is not None and end <= config.window.start_offset_s:
        raise ValueError(
            f"window.end_offset_s ({end!r}) must be greater than start_offset_s "
            f"({config.window.start_offset_s!r})"
        )


async def _acquire(semaphore: asyncio.Semaphore, timeout_s: float | None) -> bool:
    """Acquire a permit, returning False if the wait exceeded ``timeout_s``.

    ``asyncio.wait_for`` around ``Semaphore.acquire`` is safe on the supported Python
    (>=3.11): ``Semaphore.acquire`` restores the permit and wakes the next waiter when it
    is cancelled after the permit was already handed to it. On older interpreters this
    pattern leaked permits, which would silently shrink the concurrency cap over a run.
    """
    if timeout_s is None:
        await semaphore.acquire()
        return True
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout_s)
    except TimeoutError:
        return False
    return True


async def run_worker(
    offsets: list[tuple[int, float]],
    adapter: EngineAdapter,
    config: WorkerConfig,
) -> WorkerResult:
    """Drive one shard of an arrival schedule against ``adapter``.

    ``offsets`` is the output of ``shard_schedule``; the single-worker case is
    ``shard_schedule(schedule, 0, 1)``, so there is one code path.

    The adapter is injected rather than constructed here so tests run against a stub with
    no network, and so a multi-process launcher can own adapter (and tokenizer) lifetime.

    Backpressure is an ``asyncio.Semaphore(concurrency_cap)``: the fire callback acquires
    before issuing and releases on completion, so the number of requests in flight from
    this worker can never exceed the cap it was calibrated for. When every permit is held,
    further arrivals wait on the semaphore instead of opening more connections — the
    scheduler's dispatch loop keeps its own timing (the wait happens inside the task, not
    in the loop), which is what keeps the offered process open-loop.
    """
    _validate(config)

    semaphore = asyncio.Semaphore(config.concurrency_cap)
    outcomes_by_index: dict[int, RequestOutcome] = {}

    async def fire(arrival: Arrival) -> None:
        queue_start_s = time.perf_counter()
        got_permit = await _acquire(semaphore, config.max_queue_wait_s)
        queue_wait_s = time.perf_counter() - queue_start_s

        if not got_permit:
            outcomes_by_index[arrival.index] = RequestOutcome(
                index=arrival.index,
                queue_start_s=queue_start_s,
                queue_wait_s=queue_wait_s,
                start_s=None,
                end_s=None,
                result=None,
                error=f"dropped: no permit within {config.max_queue_wait_s}s",
            )
            return

        start_s = time.perf_counter()
        result: RequestResult | None = None
        error: str | None = None
        try:
            result = await adapter.request(config.prompt, config.generation)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the run
            # Recorded as a failure rather than re-raised: run_arrivals would report it as
            # a fire exception with no timing attached, and one adapter-level error would
            # then be invisible in the drop/failure accounting.
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("Request %d raised: %s", arrival.index, error)
        finally:
            semaphore.release()

        outcomes_by_index[arrival.index] = RequestOutcome(
            index=arrival.index,
            queue_start_s=queue_start_s,
            queue_wait_s=queue_wait_s,
            start_s=start_s,
            end_s=time.perf_counter(),
            result=result,
            error=error if error is not None else (result.error if result else None),
        )

    run = await run_arrivals(offsets, fire)

    # Dispatch order, so `outcomes` and `labels` line up with `run.arrivals`. A fire
    # callback that raised before recording leaves no outcome; those are counted only as
    # fire exceptions and excluded here rather than being invented as failures.
    outcomes = [outcomes_by_index[a.index] for a in run.arrivals if a.index in outcomes_by_index]
    labels = [classify_outcome(o, run.t0_perf, config.window) for o in outcomes]

    counts = {
        SUCCESS: 0,
        FAILED: 0,
        DROPPED_BACKPRESSURE: 0,
        INCOMPLETE_AT_WINDOW_END: 0,
        OUTSIDE_WINDOW: 0,
    }
    per_request_metrics: list[Metrics] = []
    token_count_warnings = 0

    for outcome, label in zip(outcomes, labels):
        if label == SUCCESS:
            assert outcome.result is not None
            try:
                metrics = compute_metrics(outcome.result)
            except ValueError as exc:
                # A "successful" response with no chunks is not a measurement. Demoted to
                # a failure rather than dropped silently or counted as a zero-latency win.
                logger.warning("Request %d succeeded but yielded no chunks: %s", outcome.index, exc)
                counts[FAILED] += 1
                continue
            per_request_metrics.append(metrics)
            if metrics.token_count_warning:
                token_count_warnings += 1
        counts[label] += 1

    queue_waits = [o.queue_wait_s for o in outcomes if o.start_s is not None]

    if counts[DROPPED_BACKPRESSURE]:
        logger.warning(
            "%d of %d in-window arrivals were dropped waiting for a permit (cap=%d). The "
            "offered rate exceeded what this worker was allowed to have in flight; the "
            "measured latencies describe only the requests that got through.",
            counts[DROPPED_BACKPRESSURE],
            len(outcomes),
            config.concurrency_cap,
        )
    if counts[INCOMPLETE_AT_WINDOW_END]:
        logger.info(
            "%d requests started inside the window and finished after it; discarded "
            "(neither drops nor successes).",
            counts[INCOMPLETE_AT_WINDOW_END],
        )

    return WorkerResult(
        config=config,
        t0_perf=run.t0_perf,
        t0_wall=run.t0_wall,
        dispatch_wall_s=run.dispatch_wall_s,
        total_wall_s=run.total_wall_s,
        achieved_qps=run.achieved_qps,
        num_dispatched=len(run.arrivals),
        num_success=counts[SUCCESS],
        num_failed=counts[FAILED],
        num_dropped_backpressure=counts[DROPPED_BACKPRESSURE],
        num_incomplete_at_window_end=counts[INCOMPLETE_AT_WINDOW_END],
        num_outside_window=counts[OUTSIDE_WINDOW],
        num_dropped_slo=None,
        outcomes=outcomes,
        labels=labels,
        per_request_metrics=per_request_metrics,
        token_count_warnings=token_count_warnings,
        num_fire_exceptions=run.num_fire_exceptions,
        queue_wait=summarize(queue_waits) if queue_waits else None,
        timing=run.timing,
    )
