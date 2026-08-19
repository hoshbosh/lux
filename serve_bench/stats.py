from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, quantiles


@dataclass
class StatSummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def summarize(values_s: list[float]) -> StatSummary:
    """Summarize a list of durations in SECONDS as percentiles in MILLISECONDS.

    Seconds are the internal representation everywhere in serve_bench; milliseconds
    appear only at reporting boundaries (this function, and JSON serialization).
    """
    if not values_s:
        raise ValueError("Cannot summarize zero values")

    ms = [v * 1000 for v in values_s]
    if len(ms) < 2:
        # statistics.quantiles requires at least 2 data points.
        sole = ms[0]
        return StatSummary(mean_ms=sole, p50_ms=sole, p95_ms=sole, p99_ms=sole, min_ms=sole, max_ms=sole)
    # method="inclusive", NOT the "exclusive" default. The default treats the sample as
    # drawn from a larger population and extrapolates past both ends, so it can report a
    # p99 ABOVE the largest value actually observed — with 20 samples of range(1, 21) it
    # returns 20.79 against a true max of 20. A latency percentile that exceeds the
    # slowest measured request is indefensible in a benchmark. "inclusive" interpolates
    # strictly between observed data points and is bounded by [min, max].
    qs = quantiles(ms, n=100, method="inclusive")
    return StatSummary(
        mean_ms=mean(ms),
        p50_ms=qs[49],
        p95_ms=qs[94],
        p99_ms=qs[98],
        min_ms=min(ms),
        max_ms=max(ms),
    )
