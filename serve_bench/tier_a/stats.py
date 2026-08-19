from __future__ import annotations

from dataclasses import dataclass

from ..metrics import Metrics
from ..stats import StatSummary, summarize

__all__ = ["StatSummary", "AggregateStats", "aggregate"]


@dataclass
class AggregateStats:
    count: int
    ttft: StatSummary
    tpot: StatSummary | None
    e2e: StatSummary
    total_completion_tokens: int
    throughput_tok_per_s: float
    cost_per_mtok_usd: float | None


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
        ttft=summarize([m.ttft for m in metrics]),
        tpot=summarize(tpot_values) if tpot_values else None,
        e2e=summarize([m.e2e for m in metrics]),
        total_completion_tokens=total_tokens,
        throughput_tok_per_s=total_tokens / wall_time_s if wall_time_s > 0 else 0.0,
        cost_per_mtok_usd=cost,
    )
