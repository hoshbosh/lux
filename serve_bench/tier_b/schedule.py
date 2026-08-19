from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArrivalScheduleConfig:
    qps: float
    duration_s: float
    seed: int
    max_requests: int = 100_000


@dataclass
class ArrivalSchedule:
    config: ArrivalScheduleConfig
    offsets: list[float]  # seconds from origin, strictly increasing, all > 0
    truncated: bool  # max_requests was hit before duration_s elapsed


def build_schedule(config: ArrivalScheduleConfig) -> ArrivalSchedule:
    """Generate a Poisson arrival schedule as absolute offsets from the run origin.

    The schedule is a pure function of (qps, duration_s, seed) and is fully decided
    BEFORE the run starts. That is the coordinated-omission defense: the offered load
    is a fact, never a consequence of how the server responded.
    """
    if config.qps <= 0:
        raise ValueError(f"qps must be > 0, got {config.qps!r}")
    if config.duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {config.duration_s!r}")
    if config.max_requests <= 0:
        raise ValueError(f"max_requests must be > 0, got {config.max_requests!r}")

    # A local Random, never the global `random` module — the global is shared state and
    # would make schedules non-reproducible under concurrent seeding.
    rng = random.Random(config.seed)

    offsets: list[float] = []
    t = 0.0
    truncated = False
    while True:
        # expovariate takes the RATE (lambda) and returns the GAP. Passing 1/qps here
        # would make the mean gap qps^2 times too large.
        t += rng.expovariate(config.qps)
        if t >= config.duration_s:
            break
        if len(offsets) >= config.max_requests:
            truncated = True
            logger.warning(
                "Arrival schedule truncated at max_requests=%d (%.1f QPS x %.1fs would "
                "produce ~%d arrivals). Offered load will be lower than configured.",
                config.max_requests,
                config.qps,
                config.duration_s,
                int(config.qps * config.duration_s),
            )
            break
        offsets.append(t)

    # Note: no arrival is pinned at offset 0.0. A Poisson process does not guarantee an
    # event at the origin; the first gap is a genuine sample.
    return ArrivalSchedule(config=config, offsets=offsets, truncated=truncated)


def shard_schedule(
    schedule: ArrivalSchedule, shard_index: int, num_shards: int
) -> list[tuple[int, float]]:
    """Deal one global schedule across num_shards workers by index.

    Returns [(global_index, offset), ...] where global_index % num_shards == shard_index.

    Dealing a single global schedule — rather than having each worker sample its own
    stream — keeps the aggregate arrival process bit-identical regardless of worker
    count, so an 8-worker run can be reproduced single-process for debugging.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards!r}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index!r}"
        )
    return [
        (i, offset)
        for i, offset in enumerate(schedule.offsets)
        if i % num_shards == shard_index
    ]
