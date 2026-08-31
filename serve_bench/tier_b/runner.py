"""Tier B entry point: config in, one scored run out.

Composes layers that are each independently testable — schedule, launcher, warmup,
goodput — and adds only the two things that need the whole picture: the Tier A ITL
baseline lookup, and the cost contract.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..adapter.base import GenerationConfig
from ..metrics import Metrics
from ..stats import StatSummary
from ..tier_a.prompt import build_prompt
from .config import TierBConfig
from .goodput import GoodputConfig, GoodputResult, compute_goodput
from .launcher import AdapterSpec, LoadConfig, TierBLoadResult, cap_for_target, run_load
from .schedule import ArrivalScheduleConfig
from .warmup import WarmupConfig

logger = logging.getLogger(__name__)


def baseline_itl_ms_from_tier_a(path: str | Path) -> float | None:
    """Pull the pooled p99 ITL out of a Tier A result JSON.

    **p99, not the median — this was corrected after the first real vLLM run.** Smooth
    goodput tests the MAXIMUM interval across a whole response, so a 128-token request
    puts 127 draws against the threshold and passes only if every one clears it. Setting
    the bar from the median ignores that structure entirely. On the first real run the
    median-derived threshold (24.8ms x 2 = 49.5ms) sat *below* Tier A's own worst
    observed interval of 55.2ms, so even the unloaded engine failed the gate on 4 of 20
    requests, and the loaded run scored 1 of 392. A tail statistic is the right input for
    a tail test.

    Returns None (with a warning) rather than raising when the field is absent: result
    files written before Tier A grew an ITL summary are still valid Tier A runs, and the
    caller may have supplied an explicit itl_slo_ms that makes the baseline unnecessary.
    Whether a missing baseline is fatal is `resolve_slo`'s decision, not this function's.
    """
    doc = json.loads(Path(path).read_text())
    itl = doc.get("aggregate", {}).get("itl")
    if not itl or itl.get("p99_ms") is None:
        logger.warning(
            "Tier A result %s has no aggregate.itl.p99_ms — it predates the pooled ITL "
            "summary, or no request in it produced two chunks. No baseline derived.", path
        )
        return None
    return float(itl["p99_ms"])


def _metrics_from_result(doc: dict) -> list[Metrics]:
    per_request = doc.get("load", {}).get("per_request")
    if per_request is None:
        raise ValueError(
            "This result has no per_request block, so it cannot be re-scored — it was "
            "written before per-request metrics were persisted. Re-run to get a "
            "re-scorable result."
        )
    return [
        Metrics(
            ttft=r["ttft_ms"] / 1000,
            tpot=r["tpot_ms"] / 1000 if r.get("tpot_ms") is not None else None,
            e2e=r["e2e_ms"] / 1000,
            itl=[v / 1000 for v in r["itl_ms"]],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
            token_count_warning=r["token_count_warning"],
        )
        for r in per_request
    ]


def rescore_goodput(
    path: str | Path, config: GoodputConfig, baseline_itl_ms: float | None = None
) -> GoodputResult:
    """Re-score a saved Tier B run at a different SLO, without touching a GPU.

    The ITL threshold is a tuning knob, and re-running a benchmark to move a knob is the
    expensive way to do it. Everything goodput needs — per-chunk ITL, the failure count,
    the backpressure drops, the measurement span — is already in the result file, so the
    same `compute_goodput` that scored the live run scores the saved one. Identical code
    path, so a swept number and a live number are directly comparable.
    """
    doc = json.loads(Path(path).read_text())
    load = doc.get("load", {})
    summary = load.get("summary", {})
    return compute_goodput(
        metrics=_metrics_from_result(doc),
        num_failed=summary.get("num_failed", 0),
        num_dropped_backpressure=load.get("dropped", {}).get("backpressure", 0),
        span_s=summary.get("in_window_span_s", 0.0),
        config=config,
        baseline_itl_ms=baseline_itl_ms,
    )


@dataclass
class TierBResult:
    config: TierBConfig
    run_at: str
    concurrency_cap: int
    load: TierBLoadResult
    goodput: GoodputResult
    # Both required by the cost contract: a lone max-throughput number is the one
    # everyone games. None when no GPU price was disclosed, never 0.0.
    cost_per_mtok_at_throughput: float | None
    cost_per_mtok_at_slo: float | None

    def to_json(self) -> str:
        def _s(x: StatSummary | None) -> dict | None:
            return dataclasses.asdict(x) if x is not None else None

        g = self.goodput
        doc = {
            "run_at": self.run_at,
            "tier": "B",
            "config": dataclasses.asdict(self.config),
            "derived": {
                "concurrency_cap_per_worker": self.concurrency_cap,
                "note": "cap = cap_for_target(target_concurrency, num_workers), sized at "
                        "the Poisson burst peak rather than the mean.",
            },
            "load": json.loads(self.load.to_json()),
            "goodput": {
                "slo": {
                    "ttft_ms": g.slo.ttft_ms,
                    "itl_ms": g.slo.itl_ms,
                    "itl_source": g.slo.itl_source,
                    "per_chunk": True,
                    "note": "ITL is checked at every chunk, never averaged. Averaging is "
                            "what lets a buffering engine pass.",
                },
                "goodput_qps": g.goodput_qps,
                "goodput_fraction": g.goodput_fraction,
                "denominator": g.denominator,
                "denominator_note": "completed + failed + backpressure-dropped, so "
                                    "shedding load cannot raise the fraction.",
                "num_goodput_ok": g.num_goodput_ok,
                "num_slo_fail_ttft": g.num_slo_fail_ttft,
                "num_slo_fail_itl": g.num_slo_fail_itl,
                "num_slo_fail_both": g.num_slo_fail_both,
                "num_no_itl_evidence": g.num_no_itl_evidence,
                "num_dropped_slo": g.num_dropped_slo,
                "num_dropped_backpressure": g.num_dropped_backpressure,
                "completion_tokens_total": g.completion_tokens_total,
                "completion_tokens_ok": g.completion_tokens_ok,
                "worst_itl": _s(g.worst_itl),
            },
            "cost": {
                "per_mtok_at_throughput": self.cost_per_mtok_at_throughput,
                "per_mtok_at_slo": self.cost_per_mtok_at_slo,
                "gpu_cost_per_hour": self.config.gpu_cost_per_hour,
                "note": "at_throughput divides spend by all completion tokens; at_slo "
                        "divides by tokens from requests that met the SLO. The second is "
                        "always the higher number — badly served tokens still cost money.",
            },
        }
        return json.dumps(doc, indent=2)


def _cost(gpu_cost_per_hour: float, span_s: float, tokens: int) -> float | None:
    """$/MTok, or None when undefined. None distinguishes 'no price disclosed' from a
    genuinely free run, matching tier_a.stats.aggregate."""
    if gpu_cost_per_hour <= 0 or tokens <= 0 or span_s <= 0:
        return None
    return ((gpu_cost_per_hour / 3600) * span_s / tokens) * 1_000_000


def run_tier_b(config: TierBConfig, tier_a_result: str | Path | None = None) -> TierBResult:
    from transformers import AutoTokenizer

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    baseline_itl_ms = (
        baseline_itl_ms_from_tier_a(tier_a_result) if tier_a_result is not None else None
    )

    # The prompt is built once in the parent and shipped to every worker, so all workers
    # offer byte-identical load. Workers each load their own tokenizer for the adapter;
    # this parent-side load is the cost of that guarantee.
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompt = build_prompt(tokenizer, config.target_prompt_tokens)

    cap = cap_for_target(config.target_concurrency, config.num_workers)
    load_config = LoadConfig(
        schedule=ArrivalScheduleConfig(
            qps=config.qps, duration_s=config.duration_s, seed=config.seed
        ),
        prompt=prompt,
        generation=GenerationConfig(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
        ),
        num_workers=config.num_workers,
        concurrency_cap=cap,
        max_queue_wait_s=(
            config.max_queue_wait_ms / 1000 if config.max_queue_wait_ms is not None else None
        ),
        warmup=WarmupConfig(
            window_s=config.warmup_window_s, hard_cutoff_s=config.warmup_hard_cutoff_s
        ),
    )
    spec = AdapterSpec(
        engine=config.engine,
        base_url=config.base_url,
        model=config.model,
        tokenizer_name=config.tokenizer_name,
        uds=config.uds,
        seed=config.seed,
    )

    load = run_load(spec, load_config)
    goodput = compute_goodput(
        metrics=load.metrics,
        num_failed=load.num_failed,
        num_dropped_backpressure=load.num_dropped_backpressure,
        span_s=load.in_window_span_s,
        config=GoodputConfig(
            ttft_slo_ms=config.ttft_slo_ms,
            itl_slo_ms=config.itl_slo_ms,
            itl_baseline_multiple=config.itl_baseline_multiple,
            exclude_first_itl=config.exclude_first_itl,
        ),
        baseline_itl_ms=baseline_itl_ms,
    )

    return TierBResult(
        config=config,
        run_at=run_at,
        concurrency_cap=cap,
        load=load,
        goodput=goodput,
        cost_per_mtok_at_throughput=_cost(
            config.gpu_cost_per_hour, load.in_window_span_s, goodput.completion_tokens_total
        ),
        cost_per_mtok_at_slo=_cost(
            config.gpu_cost_per_hour, load.in_window_span_s, goodput.completion_tokens_ok
        ),
    )
