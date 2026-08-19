from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

from ..stats import StatSummary, summarize

if TYPE_CHECKING:  # avoids a runtime import cycle with arrivals.py
    from .arrivals import Arrival

_ZERO = StatSummary(mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, min_ms=0.0, max_ms=0.0)


@dataclass
class ArrivalTimingStats:
    """How well the load generator kept its own promise. Not a property of the engine."""

    count: int
    lateness: StatSummary
    interarrival_abs_error: StatSummary
    interarrival_signed_error_mean_ms: float


def summarize_timing(arrivals: list[Arrival]) -> ArrivalTimingStats:
    """Summarize generator self-accuracy over a list of dispatched arrivals.

    Two different questions are answered here:
      - lateness: how much service debt the generator accrued (the coordinated-omission
        relevant quantity — a request dispatched late was still owed to the user).
      - inter-arrival error: how accurately gaps between consecutive dispatches matched
        the schedule (the generator-accuracy quantity the B2 calibration sweep consumes).

    The signed mean sits alongside the absolute percentiles because a consistently
    positive signed error means a systematic under-rate, whereas symmetric error is
    just jitter.
    """
    if not arrivals:
        raise ValueError("Cannot summarize zero arrivals")

    # Inter-arrival error is the first difference of lateness:
    #   (actual[i] - actual[i-1]) - (scheduled[i] - scheduled[i-1])
    # Computing it against the FIRST arrival instead of pairwise would report cumulative
    # drift rather than per-gap accuracy.
    signed_errors_s = [
        arrivals[i].lateness_s - arrivals[i - 1].lateness_s for i in range(1, len(arrivals))
    ]

    return ArrivalTimingStats(
        count=len(arrivals),
        lateness=summarize([a.lateness_s for a in arrivals]),
        # A single arrival has no pairs, so there is no gap to be wrong about.
        interarrival_abs_error=summarize([abs(e) for e in signed_errors_s]) if signed_errors_s else _ZERO,
        interarrival_signed_error_mean_ms=mean(signed_errors_s) * 1000 if signed_errors_s else 0.0,
    )
