"""Per-worker concurrency ceiling calibration (the B2 experiment).

Tier B is multi-process because a single asyncio event loop saturates and starts
injecting its own latency into the offered load. This module measures *where* that
happens for one worker, so the worker count N stops being a guessed default:

    N = ceil(target_concurrency / per_worker_ceiling)

Method: for each target concurrency C, drive ``run_arrivals`` with a synthetic fire
callback and no server at all. The measured quantity is the generator's own
inter-arrival error (``ArrivalTimingStats.interarrival_abs_error``) — how far actual
dispatch gaps drifted from the schedule. That is a property of the load generator, not
of any engine, which is exactly why the sweep needs no GPU to be meaningful.

Offered rate is derived from Little's Law: to hold C requests in flight when each takes
``request_duration_s``, arrivals must come at ``C / request_duration_s`` QPS. The
callback also samples in-flight depth so the result records whether C was actually
reached rather than assuming the law held.

**Disclosure — this is an optimistic bound.** The synthetic callback models a streaming
response as ``chunks_per_request`` sleeps of ``itl_s``, which reproduces the event-loop
wakeup pattern of an SSE stream but not its socket reads, HTTP parsing, or tokenizer
work. A real worker will hit its ceiling at or below the number measured here, never
above it. Run on the target host (RunPod, not a laptop) — socket and timer behavior
differ enough between platforms that a local number does not transfer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, field

from .arrivals import Arrival
from .arrivals import run_arrivals
from .schedule import ArrivalScheduleConfig, build_schedule, shard_schedule
from .timing import ArrivalTimingStats

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = [10, 25, 50, 100, 200, 350, 500]

# Below this many arrivals a p99 is being read off too few samples to trust.
_MIN_ARRIVALS_FOR_P99 = 200


@dataclass
class CalibrationConfig:
    concurrency_levels: list[int] = field(default_factory=lambda: list(DEFAULT_LEVELS))
    duration_s: float = 20.0  # per level
    chunks_per_request: int = 64
    itl_s: float = 0.02
    # The knee criterion. p99 rather than mean: a generator that is accurate on average
    # and 50ms late in the tail has already corrupted the tail latencies Tier B exists
    # to measure.
    p99_error_threshold_ms: float = 2.0
    seed: int = 0

    @property
    def request_duration_s(self) -> float:
        return self.chunks_per_request * self.itl_s

    def qps_for(self, concurrency: int) -> float:
        return concurrency / self.request_duration_s


@dataclass
class CalibrationPoint:
    target_concurrency: int
    offered_qps: float
    num_arrivals: int
    achieved_qps: float
    peak_in_flight: int
    mean_in_flight: float
    timing: ArrivalTimingStats
    passed: bool
    low_sample_warning: bool


@dataclass
class CalibrationResult:
    config: CalibrationConfig
    points: list[CalibrationPoint]
    # Highest swept level that passed before the first failure. None if even the lowest
    # level failed.
    per_worker_ceiling: int | None
    # True when no level failed: the knee is somewhere above the swept range, so the
    # ceiling is a floor on the truth, not the truth.
    ceiling_is_lower_bound: bool

    def workers_for(self, target_concurrency: int) -> int:
        """N = ceil(target_concurrency / ceiling), the number this experiment exists for."""
        if self.per_worker_ceiling is None:
            raise ValueError(
                "No concurrency level met the error threshold; cannot derive a worker "
                "count. Re-run with lower concurrency_levels or a looser threshold."
            )
        return max(1, math.ceil(target_concurrency / self.per_worker_ceiling))

    def to_json(self) -> str:
        def _summary(s) -> dict:
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
            "experiment": "tier_b_worker_calibration",
            "config": {
                "concurrency_levels": self.config.concurrency_levels,
                "duration_s": self.config.duration_s,
                "chunks_per_request": self.config.chunks_per_request,
                "itl_ms": self.config.itl_s * 1000,
                "request_duration_s": self.config.request_duration_s,
                "p99_error_threshold_ms": self.config.p99_error_threshold_ms,
                "seed": self.config.seed,
            },
            "disclosure": (
                "Synthetic fire callback: sleeps only, no HTTP or tokenizer work. "
                "The measured ceiling is an optimistic upper bound on a real worker's."
            ),
            "per_worker_ceiling": self.per_worker_ceiling,
            "ceiling_is_lower_bound": self.ceiling_is_lower_bound,
            "points": [
                {
                    "target_concurrency": p.target_concurrency,
                    "offered_qps": p.offered_qps,
                    "num_arrivals": p.num_arrivals,
                    "achieved_qps": p.achieved_qps,
                    "peak_in_flight": p.peak_in_flight,
                    "mean_in_flight": p.mean_in_flight,
                    "passed": p.passed,
                    "low_sample_warning": p.low_sample_warning,
                    "lateness": _summary(p.timing.lateness),
                    "interarrival_abs_error": _summary(p.timing.interarrival_abs_error),
                    "interarrival_signed_error_mean_ms": p.timing.interarrival_signed_error_mean_ms,
                }
                for p in self.points
            ],
        }
        return json.dumps(doc, indent=2)


class _InFlightTracker:
    """Counts concurrent synthetic requests and samples depth at each arrival.

    Sampling on arrival rather than on a timer keeps the tracker off the event loop as a
    separate task — an extra polling task would itself consume the loop capacity the
    sweep is trying to measure.
    """

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.samples: list[int] = []

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)
        self.samples.append(self.current)

    def exit(self) -> None:
        self.current -= 1

    @property
    def mean(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0


async def _run_level(config: CalibrationConfig, concurrency: int) -> CalibrationPoint:
    qps = config.qps_for(concurrency)
    schedule = build_schedule(
        ArrivalScheduleConfig(qps=qps, duration_s=config.duration_s, seed=config.seed)
    )
    tracker = _InFlightTracker()

    async def fire(arrival: Arrival) -> None:
        tracker.enter()
        try:
            # One sleep per chunk, not a single sleep of request_duration_s: the loop cost
            # of a streaming response is dominated by its per-chunk wakeups, and collapsing
            # them into one sleep would measure a workload that does not exist.
            for _ in range(config.chunks_per_request):
                await asyncio.sleep(config.itl_s)
        finally:
            tracker.exit()

    result = await run_arrivals(shard_schedule(schedule, 0, 1), fire)

    if result.timing is None:
        raise RuntimeError(
            f"Concurrency level {concurrency} dispatched zero arrivals "
            f"({qps:.1f} QPS x {config.duration_s}s) — increase duration_s."
        )

    low_sample = len(result.arrivals) < _MIN_ARRIVALS_FOR_P99
    if low_sample:
        logger.warning(
            "Level %d produced only %d arrivals; p99 inter-arrival error is unreliable "
            "below %d samples. Increase duration_s.",
            concurrency,
            len(result.arrivals),
            _MIN_ARRIVALS_FOR_P99,
        )
    if tracker.peak < concurrency * 0.8:
        logger.warning(
            "Level %d only reached %d in flight (target %d). Little's Law assumed each "
            "request occupies a slot for %.2fs; if the loop is already saturated the "
            "offered concurrency was never achieved.",
            concurrency,
            tracker.peak,
            concurrency,
            config.request_duration_s,
        )

    return CalibrationPoint(
        target_concurrency=concurrency,
        offered_qps=qps,
        num_arrivals=len(result.arrivals),
        achieved_qps=result.achieved_qps,
        peak_in_flight=tracker.peak,
        mean_in_flight=tracker.mean,
        timing=result.timing,
        passed=result.timing.interarrival_abs_error.p99_ms <= config.p99_error_threshold_ms,
        low_sample_warning=low_sample,
    )


def select_ceiling(points: list[CalibrationPoint]) -> tuple[int | None, bool]:
    """Locate the knee: (per_worker_ceiling, ceiling_is_lower_bound).

    Only the FIRST failure defines the ceiling. A level that passes after an earlier one
    failed is noise, not recovery — taking the maximum passing level instead would report
    a ceiling above a concurrency the generator has already been shown to mistime.

    Returns ``(None, False)`` when even the lowest level failed. The bool is True only
    when nothing failed, meaning the knee lies above the swept range and the returned
    ceiling is a floor on the truth rather than the truth.

    Kept separate from ``run_calibration`` so the knee rule is testable without spending
    a multi-minute sweep on it.
    """
    ceiling: int | None = None
    for point in points:
        if not point.passed:
            return ceiling, False
        ceiling = point.target_concurrency
    return ceiling, True


def run_calibration(config: CalibrationConfig) -> CalibrationResult:
    """Sweep concurrency levels and locate the per-worker knee.

    The sweep does not stop at the first failure — the full curve is recorded so the
    knee can be re-derived from the data under a different threshold without re-running
    the experiment.
    """
    if not config.concurrency_levels:
        raise ValueError("concurrency_levels must not be empty")
    if any(c <= 0 for c in config.concurrency_levels):
        raise ValueError(f"concurrency_levels must all be > 0, got {config.concurrency_levels!r}")
    if config.itl_s <= 0 or config.chunks_per_request <= 0:
        raise ValueError("chunks_per_request and itl_s must both be > 0")

    levels = sorted(config.concurrency_levels)
    points: list[CalibrationPoint] = []
    for concurrency in levels:
        logger.info(
            "Calibrating concurrency=%d (%.1f QPS for %.0fs)",
            concurrency,
            config.qps_for(concurrency),
            config.duration_s,
        )
        points.append(asyncio.run(_run_level(config, concurrency)))

    ceiling, ceiling_is_lower_bound = select_ceiling(points)

    if ceiling is None:
        logger.warning(
            "Even the lowest level (%d) exceeded the %.1fms p99 error threshold. The "
            "knee is below the swept range, or the host is too noisy to calibrate on.",
            levels[0],
            config.p99_error_threshold_ms,
        )
    elif ceiling_is_lower_bound:
        logger.warning(
            "No level failed; the ceiling is at least %d but the knee was not found. "
            "Extend concurrency_levels upward before deriving a worker count.",
            ceiling,
        )

    return CalibrationResult(
        config=config,
        points=points,
        per_worker_ceiling=ceiling,
        ceiling_is_lower_bound=ceiling_is_lower_bound,
    )
