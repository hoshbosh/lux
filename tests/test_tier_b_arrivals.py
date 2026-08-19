import asyncio
import logging
import time

import pytest

from serve_bench.tier_b.arrivals import run_arrivals
from serve_bench.tier_b.schedule import ArrivalScheduleConfig, build_schedule, shard_schedule

# One-sided slack for timing assertions. Every timing assert is an upper bound on
# lateness, never an equality, so these tests do not flake on a loaded CI box.
SLACK_S = 0.05


class RecordingFire:
    """Records dispatch times; optionally stalls or raises on chosen indices.

    sleep_map: index -> seconds of async sleep (yields the event loop)
    block_map: index -> seconds of BLOCKING sleep (stalls the event loop itself)
    raise_on: set of indices that raise RuntimeError
    """

    def __init__(self, sleep_map=None, block_map=None, raise_on=None):
        self.sleep_map = sleep_map or {}
        self.block_map = block_map or {}
        self.raise_on = raise_on or set()
        self.dispatch_times: list[float] = []
        self.dispatched_indices: list[int] = []
        self.completed = 0

    async def __call__(self, arrival):
        self.dispatch_times.append(time.perf_counter())
        self.dispatched_indices.append(arrival.index)
        if arrival.index in self.block_map:
            time.sleep(self.block_map[arrival.index])
        if arrival.index in self.sleep_map:
            await asyncio.sleep(self.sleep_map[arrival.index])
        if arrival.index in self.raise_on:
            raise RuntimeError(f"boom {arrival.index}")
        self.completed += 1
        return arrival.index


async def test_arrivals_fire_at_scheduled_offsets():
    # Likely cause of failure: sleeping the relative gap instead of computing an
    # absolute deadline from the anchor, which accumulates drift across arrivals.
    offsets = [(0, 0.05), (1, 0.10), (2, 0.15), (3, 0.20)]
    fire = RecordingFire()
    result = await run_arrivals(offsets, fire)

    assert len(result.arrivals) == 4
    for arrival, (index, offset) in zip(result.arrivals, offsets):
        assert arrival.index == index
        assert arrival.scheduled_time_s == result.t0_perf + offset
        assert 0 <= arrival.lateness_s < SLACK_S


async def test_slow_request_does_not_delay_later_arrivals():
    # THE defining open-loop property. Request 0 takes 1.0s; arrivals 1-4 are scheduled
    # 20-100ms in. In a closed loop the 5th arrival could not occur before ~1.08s.
    # Likely cause of failure: `await fire(arrival)` instead of
    # `asyncio.create_task(fire(arrival))` — the closed-loop regression this entire
    # tier exists to avoid.
    offsets = [(0, 0.02), (1, 0.04), (2, 0.06), (3, 0.08), (4, 0.10)]
    fire = RecordingFire(sleep_map={0: 1.0})
    result = await run_arrivals(offsets, fire)

    assert len(fire.dispatch_times) == 5
    assert result.arrivals[4].actual_time_s - result.t0_perf < 0.30
    assert max(a.lateness_s for a in result.arrivals[1:]) < SLACK_S
    # The slow task was still awaited, not abandoned.
    assert result.total_wall_s >= 1.0


async def test_catch_up_does_not_skip_or_reanchor():
    # Request 0 blocks the event loop itself for 150ms, so arrivals 1-4 (scheduled at
    # 20-50ms) all come due while the loop is stalled.
    # Likely cause of failure: skipping the missed arrivals (textbook coordinated
    # omission) or re-anchoring t0 (silently throttles the offered rate).
    offsets = [(0, 0.01), (1, 0.02), (2, 0.03), (3, 0.04), (4, 0.05)]
    fire = RecordingFire(block_map={0: 0.15})
    result = await run_arrivals(offsets, fire)

    # Nothing was skipped.
    assert [a.index for a in result.arrivals] == [0, 1, 2, 3, 4]
    # The lateness was recorded honestly, not hidden.
    assert all(a.lateness_s > 0.10 for a in result.arrivals[1:])
    # Arrivals 1-4 were released as one catch-up burst — they were owed.
    burst = [a.actual_time_s for a in result.arrivals[1:]]
    assert max(burst) - min(burst) < SLACK_S
    # The deadlines themselves were never rewritten.
    for arrival, (_, offset) in zip(result.arrivals, offsets):
        assert arrival.scheduled_time_s == result.t0_perf + offset


async def test_fire_exception_recorded_not_swallowed(caplog):
    # Likely cause of failure: a bare create_task whose exception surfaces only as
    # "Task exception was never retrieved" at GC time — by which point the run has
    # already been reported with a wrong denominator.
    offsets = [(0, 0.01), (1, 0.02), (2, 0.03)]
    fire = RecordingFire(raise_on={1})

    with caplog.at_level(logging.WARNING):
        result = await run_arrivals(offsets, fire)

    assert result.num_fire_exceptions == 1
    assert result.fire_results[1] is None
    assert result.fire_results[0] == 0
    assert result.fire_results[2] == 2
    assert "arrival 1" in caplog.text


async def test_all_tasks_complete_before_return():
    # Likely cause of failure: returning without draining, leaking pending tasks.
    offsets = [(0, 0.01), (1, 0.02), (2, 0.03)]
    fire = RecordingFire(sleep_map={2: 0.1})
    result = await run_arrivals(offsets, fire)

    assert fire.completed == 3
    assert len(result.fire_results) == 3


async def test_empty_schedule_returns_empty_result():
    result = await run_arrivals([], RecordingFire())

    assert result.arrivals == []
    assert result.fire_results == []
    assert result.timing is None
    assert result.num_fire_exceptions == 0
    assert result.achieved_qps == 0.0


async def test_achieved_qps_is_count_over_dispatch_span():
    # Identity check, not a statistical one, so it cannot flake.
    # Likely cause of failure: dividing by total_wall_s, which includes the drain window
    # and would understate the offered rate by however long the slowest request took.
    offsets = [(0, 0.01), (1, 0.02), (2, 0.03)]
    fire = RecordingFire(sleep_map={2: 0.1})
    result = await run_arrivals(offsets, fire)

    assert result.achieved_qps == pytest.approx(
        len(result.arrivals) / result.dispatch_wall_s, rel=1e-9
    )
    # Requests were still in flight when dispatching finished — the visible signature
    # of open-loop behavior.
    assert result.dispatch_wall_s < result.total_wall_s


async def test_sharded_run_dispatches_only_its_shard():
    schedule = build_schedule(
        ArrivalScheduleConfig(qps=200.0, duration_s=10.0, seed=9, max_requests=100_000)
    )
    schedule.offsets = schedule.offsets[:20]
    shard = shard_schedule(schedule, 1, 4)

    result = await run_arrivals(shard, RecordingFire())

    assert [a.index for a in result.arrivals] == [1, 5, 9, 13, 17]
