from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, quantiles

from ..metrics import Metrics


@dataclass
class StatSummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class AggregateStats:
    count: int
    ttft: StatSummary
    tpot: StatSummary | None
    e2e: StatSummary
    total_completion_tokens: int
    throughput_tok_per_s: float
    cost_per_mtok_usd: float | None


def _summarize(values_s: list[float]) -> StatSummary:
    ms = [v * 1000 for v in values_s]
    if len(ms) < 2:
        sole = ms[0]
        return StatSummary(mean_ms=sole, p50_ms=sole, p95_ms=sole, p99_ms=sole, min_ms=sole, max_ms=sole)
    qs = quantiles(ms, n=100)
    return StatSummary(
        mean_ms=mean(ms),
        p50_ms=qs[49],
        p95_ms=qs[94],
        p99_ms=qs[98],
        min_ms=min(ms),
        max_ms=max(ms),
    )


def aggregate(
    metrics: list[Metrics],
    wall_time_s: float,
    gpu_cost_per_hour: float,
) -> AggregateStats:
    if not metrics:
        raise ValueError("Cannot aggregate zero metrics")

    tpot_values = [m.tpot for m in metrics if m.tpot is not None]
    total_tokens = sum(m.completion_tokens for m in metrics)

    cost: float | None = None
    if gpu_cost_per_hour > 0 and total_tokens > 0:
        gpu_cost_per_s = gpu_cost_per_hour / 3600
        total_cost = gpu_cost_per_s * wall_time_s
        cost = (total_cost / total_tokens) * 1_000_000

    return AggregateStats(
        count=len(metrics),
        ttft=_summarize([m.ttft for m in metrics]),
        tpot=_summarize(tpot_values) if tpot_values else None,
        e2e=_summarize([m.e2e for m in metrics]),
        total_completion_tokens=total_tokens,
        throughput_tok_per_s=total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        cost_per_mtok_usd=cost,
    )
