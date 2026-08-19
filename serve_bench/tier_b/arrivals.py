"""Open-loop arrival dispatch.

Two timing invariants hold throughout this module:

1. All measurement timestamps come from ``time.perf_counter()``, matching the adapters
   (see ``VLLMAdapter.request``). That is what makes ``chunk_timestamps[0] -
   arrival.scheduled_time_s`` a valid subtraction. ``time.time()`` never appears in the
   timing path — it can step under NTP mid-run and is not comparable with perf_counter.

2. ``perf_counter`` is NOT comparable across processes. ``t0_wall`` is read exactly once,
   at the same instant as ``t0_perf``, purely so a multi-process runner can map each
   worker's perf-domain values onto a common epoch: ``wall = t0_wall + (perf - t0_perf)``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .timing import ArrivalTimingStats, summarize_timing

logger = logging.getLogger(__name__)

FireCallback = Callable[["Arrival"], Awaitable[Any]]


@dataclass
class Arrival:
    index: int  # global index in the unsharded schedule
    scheduled_offset_s: float
    scheduled_time_s: float  # t0_perf + scheduled_offset_s
    actual_time_s: float  # perf_counter() read immediately before invoking fire
    lateness_s: float  # actual_time_s - scheduled_time_s


@dataclass
class ArrivalRunResult:
    arrivals: list[Arrival]
    fire_results: list[Any]  # positionally parallel to arrivals; exceptions become None
    t0_perf: float
    t0_wall: float
    dispatch_wall_s: float  # end of dispatch loop - t0_perf
    total_wall_s: float  # after draining in-flight tasks
    achieved_qps: float
    num_fire_exceptions: int
    timing: ArrivalTimingStats | None  # None when no arrivals were dispatched


async def run_arrivals(offsets: list[tuple[int, float]], fire: FireCallback) -> ArrivalRunResult:
    """Dispatch arrivals on a fixed schedule without ever waiting for a response.

    ``offsets`` is the output of ``shard_schedule`` — [(global_index, offset), ...]. The
    unsharded case is ``shard_schedule(s, 0, 1)``, so there is one code path.

    ``fire`` is injected rather than the scheduler owning an EngineAdapter: this module
    has no opinion about HTTP, so tests need no server, and backpressure / warmup gating /
    window filtering all become wrappers around the callback rather than edits here.
    """
    arrivals: list[Arrival] = []
    tasks: list[asyncio.Task] = []

    t0_perf = time.perf_counter()
    t0_wall = time.time()
    dispatch_wall_s = 0.0

    try:
        for index, offset in offsets:
            # Deadline is recomputed from the immutable anchor with a fresh clock read, so
            # overshoot on one iteration is absorbed by a shorter sleep on the next.
            # `await asyncio.sleep(gap)` would instead accumulate error linearly.
            target = t0_perf + offset
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                # Already late: dispatch immediately, never skip and never re-anchor t0.
                # Skipping deletes exactly the samples with the worst latency (coordinated
                # omission); re-anchoring silently throttles the offered rate. The
                # resulting catch-up burst is correct — those arrivals were owed.
                # sleep(0) yields one loop turn so tasks created during a long burst can
                # reach their first await instead of piling up unstarted.
                await asyncio.sleep(0)

            now = time.perf_counter()
            arrival = Arrival(
                index=index,
                scheduled_offset_s=offset,
                scheduled_time_s=target,
                actual_time_s=now,
                lateness_s=now - target,
            )
            arrivals.append(arrival)
            # create_task, never await: awaiting here is a closed loop in disguise and
            # would make the offered rate a function of the server's response time.
            tasks.append(asyncio.create_task(fire(arrival)))

        dispatch_wall_s = time.perf_counter() - t0_perf
    finally:
        # gather(return_exceptions=True), NOT asyncio.TaskGroup: TaskGroup cancels all
        # siblings on the first exception, so one failed request would abort the run and
        # take down every other in-flight request with it.
        # The finally block ensures a cancellation still drains rather than leaving
        # orphaned tasks pending at interpreter shutdown.
        raw = await asyncio.gather(*tasks, return_exceptions=True)

    total_wall_s = time.perf_counter() - t0_perf

    fire_results: list[Any] = []
    num_fire_exceptions = 0
    for arrival, value in zip(arrivals, raw):
        if isinstance(value, BaseException):
            num_fire_exceptions += 1
            fire_results.append(None)
            logger.warning("Fire callback for arrival %d raised: %s", arrival.index, value)
        else:
            fire_results.append(value)

    return ArrivalRunResult(
        arrivals=arrivals,
        fire_results=fire_results,
        t0_perf=t0_perf,
        t0_wall=t0_wall,
        dispatch_wall_s=dispatch_wall_s,
        total_wall_s=total_wall_s,
        # Measured over the dispatch span, not the drain span — using total_wall_s would
        # understate the offered rate by however long the slowest request took.
        achieved_qps=len(arrivals) / dispatch_wall_s if dispatch_wall_s > 0 else 0.0,
        num_fire_exceptions=num_fire_exceptions,
        timing=summarize_timing(arrivals) if arrivals else None,
    )
