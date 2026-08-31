"""Smooth goodput: the fraction of offered load that was actually served well.

Plain goodput — "requests that met a P99 latency SLO" — is gameable in two directions,
both documented in arXiv:2410.14257, and both of which a serving engine can perform
without doing anything its own metrics would call cheating:

**Token delay.** Buffer the first few tokens and release them together. TTFT looks fine,
mean inter-token latency looks fine, and the user still stares at a blank screen. Any
check that *averages* ITL — mean, or even a p99 across chunks — is defeated by this,
because a few long stalls hide inside many short intervals. So condition (c) below is a
per-chunk maximum: every single interval must clear the threshold, and one stall anywhere
in the stream fails the whole request.

**Request abandonment.** Drop the requests that were going to miss anyway. Success rate
climbs because the denominator shrank. This is defeated by the choice of denominator:
backpressure drops are counted in it (see ``compute_goodput``), so abandoning a request
can only ever lower the fraction.

A request counts toward smooth goodput only if all three hold at once:

    (a) it completed fully,
    (b) TTFT met the TTFT SLO,
    (c) ITL was at or under the threshold at EVERY chunk.

Condition (a) is not re-derived here. ``worker.classify_outcome`` already decides it —
a request is complete iff it was labelled ``SUCCESS``, meaning it both started and
finished inside the measurement window, the adapter reported success, and it produced
chunks. This layer receives only those requests' ``Metrics`` and adds (b) and (c).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..metrics import Metrics
from ..stats import StatSummary, summarize

logger = logging.getLogger(__name__)

REASON_TTFT = "ttft_slo"
REASON_ITL = "itl_slo"
# A response with a single chunk has no interval to measure, so (c) cannot be checked.
# That is not the same as passing it — see `evaluate_request`.
REASON_NO_ITL = "no_itl_evidence"


@dataclass
class SLO:
    ttft_ms: float
    itl_ms: float
    itl_source: str  # how the ITL threshold was arrived at; echoed into the report


@dataclass
class GoodputConfig:
    ttft_slo_ms: float
    # Explicit ITL ceiling. When None the threshold is derived from the Tier A baseline.
    itl_slo_ms: float | None = None
    # Applied to Tier A's pooled p99 ITL. Note the threshold's sensitivity to response
    # length: the check is a maximum over (completion_tokens - 1) intervals, so the same
    # engine faces a harder test as max_tokens grows. Treat this as a product decision
    # ("no token gap longer than X") and set it deliberately rather than inheriting it.
    itl_baseline_multiple: float = 2.0
    # TPOT excludes itl[0] because that interval carries the tail of prefill scheduling
    # and would inflate a steady-state throughput proxy. Smooth goodput asks a different
    # question — whether the user ever experienced a stall — and a user waiting on token
    # two experiences that wait. Excluding it would also hand an engine a free two-token
    # buffer, which is the exact paradox this metric exists to catch. Off by default.
    exclude_first_itl: bool = False


@dataclass
class RequestVerdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)  # empty iff passed; may hold several
    ttft_s: float = 0.0
    worst_itl_s: float | None = None  # None when the request had no intervals at all


@dataclass
class GoodputResult:
    slo: SLO
    num_goodput_ok: int
    num_slo_fail_ttft: int
    num_slo_fail_itl: int
    num_slo_fail_both: int
    num_no_itl_evidence: int
    num_completed: int
    num_failed: int
    # Completion tokens, split by whether the request they belong to counted. The cost
    # contract needs both: dividing spend by all tokens prices the run at max throughput,
    # dividing by well-served tokens prices it at the SLO operating point.
    completion_tokens_total: int
    completion_tokens_ok: int
    num_dropped_backpressure: int
    denominator: int
    span_s: float
    goodput_qps: float
    goodput_fraction: float
    worst_itl: StatSummary | None

    @property
    def num_dropped_slo(self) -> int:
        """Requests that completed but did not count. Fills the hook every lower layer
        has been carrying as ``None``.

        Computed as completed-minus-passed rather than by summing the failure counters:
        those overlap (a request can miss both SLOs) and they exclude the
        no-ITL-evidence case, so summing them would undercount.
        """
        return self.num_completed - self.num_goodput_ok


def resolve_slo(config: GoodputConfig, baseline_itl_ms: float | None) -> SLO:
    """Decide the ITL threshold from an explicit value or the Tier A baseline.

    Raises when neither is available. There is deliberately no fallback default: a
    guessed ITL threshold produces a goodput number that looks entirely plausible and is
    wrong, which is worse than no number at all.
    """
    if config.ttft_slo_ms <= 0:
        raise ValueError(f"ttft_slo_ms must be positive, got {config.ttft_slo_ms}")

    if config.itl_slo_ms is not None:
        if config.itl_slo_ms <= 0:
            raise ValueError(f"itl_slo_ms must be positive, got {config.itl_slo_ms}")
        return SLO(config.ttft_slo_ms, config.itl_slo_ms, "explicit")

    if baseline_itl_ms is None:
        raise ValueError(
            "No ITL SLO available. Either set itl_slo_ms in the config, or pass a Tier A "
            "result to derive the baseline from (its pooled p99 ITL x "
            f"{config.itl_baseline_multiple})."
        )
    if config.itl_baseline_multiple <= 0:
        raise ValueError("itl_baseline_multiple must be positive")

    threshold = baseline_itl_ms * config.itl_baseline_multiple
    return SLO(
        config.ttft_slo_ms,
        threshold,
        f"tier_a_p99_itl_{baseline_itl_ms:.3f}ms_x{config.itl_baseline_multiple}",
    )


def evaluate_request(
    m: Metrics, slo: SLO, exclude_first_itl: bool = False
) -> RequestVerdict:
    """Apply conditions (b) and (c) to one completed request. Pure.

    A single-chunk response fails rather than passes. It offers no evidence about (c),
    and treating absent evidence as a pass would make "emit one chunk per request" a
    winning strategy. The consequence is that smooth goodput is undefined for runs with
    ``max_tokens=1``; such runs report zero goodput and a nonzero
    ``num_no_itl_evidence``, which is the honest reading rather than a silent one.
    """
    intervals = m.itl[1:] if exclude_first_itl else m.itl

    reasons: list[str] = []
    if m.ttft * 1000 > slo.ttft_ms:
        reasons.append(REASON_TTFT)

    worst_itl_s = max(intervals) if intervals else None
    if worst_itl_s is None:
        reasons.append(REASON_NO_ITL)
    elif worst_itl_s * 1000 > slo.itl_ms:
        reasons.append(REASON_ITL)

    return RequestVerdict(
        passed=not reasons, reasons=reasons, ttft_s=m.ttft, worst_itl_s=worst_itl_s
    )


def compute_goodput(
    metrics: list[Metrics],
    num_failed: int,
    num_dropped_backpressure: int,
    span_s: float,
    config: GoodputConfig,
    baseline_itl_ms: float | None = None,
) -> GoodputResult:
    """Score a measurement window. ``metrics`` are the completed (SUCCESS) requests only.

    **The denominator is the anti-gaming mechanism.** It is every in-window arrival that
    was accounted for — completed, failed, or dropped for backpressure — not just the
    ones that came back. An engine (or a harness) that sheds borderline load therefore
    shrinks the numerator while the denominator holds, so abandonment can only lower the
    fraction. ``denominator`` is emitted as an integer field so the ratio can be audited
    rather than taken on trust.

    ``goodput_qps`` uses the measurement span, making it directly comparable with
    ``TierBLoadResult.achieved_qps_in_window``: the gap between the two is exactly the
    throughput that was served badly.
    """
    slo = resolve_slo(config, baseline_itl_ms)
    verdicts = [evaluate_request(m, slo, config.exclude_first_itl) for m in metrics]

    num_ok = sum(1 for v in verdicts if v.passed)
    fail_ttft = sum(1 for v in verdicts if REASON_TTFT in v.reasons)
    fail_itl = sum(1 for v in verdicts if REASON_ITL in v.reasons)
    fail_both = sum(
        1 for v in verdicts if REASON_TTFT in v.reasons and REASON_ITL in v.reasons
    )
    no_itl = sum(1 for v in verdicts if REASON_NO_ITL in v.reasons)

    denominator = len(metrics) + num_failed + num_dropped_backpressure
    worst = [v.worst_itl_s for v in verdicts if v.worst_itl_s is not None]
    tokens_total = sum(m.completion_tokens for m in metrics)
    tokens_ok = sum(m.completion_tokens for m, v in zip(metrics, verdicts) if v.passed)

    if no_itl:
        logger.warning(
            "%d of %d completed requests produced a single chunk, so their inter-token "
            "behaviour could not be checked; they are counted as failures. If this run "
            "used max_tokens=1, smooth goodput is not a meaningful metric for it.",
            no_itl, len(metrics),
        )
    if num_dropped_backpressure:
        logger.info(
            "%d backpressure drops are included in the goodput denominator (%d). Shedding "
            "load cannot raise this fraction.",
            num_dropped_backpressure, denominator,
        )

    return GoodputResult(
        slo=slo,
        num_goodput_ok=num_ok,
        num_slo_fail_ttft=fail_ttft,
        num_slo_fail_itl=fail_itl,
        num_slo_fail_both=fail_both,
        num_no_itl_evidence=no_itl,
        num_completed=len(metrics),
        num_failed=num_failed,
        completion_tokens_total=tokens_total,
        completion_tokens_ok=tokens_ok,
        num_dropped_backpressure=num_dropped_backpressure,
        denominator=denominator,
        span_s=span_s,
        goodput_qps=num_ok / span_s if span_s > 0 else 0.0,
        goodput_fraction=num_ok / denominator if denominator > 0 else 0.0,
        worst_itl=summarize(worst) if worst else None,
    )
