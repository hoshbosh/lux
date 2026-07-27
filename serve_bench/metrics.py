from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .adapter.base import RequestResult


@dataclass
class Metrics:
    ttft: float
    tpot: float | None  # None when fewer than 3 chunks (need ≥2 inter-chunk intervals)
    e2e: float
    itl: list[float]
    prompt_tokens: int
    completion_tokens: int
    token_count_warning: bool


def compute_metrics(result: RequestResult) -> Metrics:
    ts = result.chunk_timestamps
    if not ts:
        raise ValueError(
            "chunk_timestamps is empty — request may have failed or produced no tokens"
        )

    n = len(ts)
    ttft = ts[0] - result.t_start
    e2e = ts[-1] - result.t_start
    itl = [ts[i] - ts[i - 1] for i in range(1, n)]

    # TPOT = mean of itl[1:], NOT itl[0:].
    # itl[0] is the interval between the first and second token — it includes the
    # tail of prefill scheduling on some engines. Excluding it matches the correct
    # definition and avoids the LLMperf bug of including the first inter-token gap.
    tpot = mean(itl[1:]) if len(itl) >= 2 else None

    return Metrics(
        ttft=ttft,
        tpot=tpot,
        e2e=e2e,
        itl=itl,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        token_count_warning=result.token_count_warning,
    )
