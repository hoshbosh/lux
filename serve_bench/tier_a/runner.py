from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from transformers import AutoTokenizer

from ..adapter.base import GenerationConfig
from ..adapter.llamacpp import LlamaCppAdapter
from ..adapter.vllm import VLLMAdapter
from ..metrics import Metrics, compute_metrics
from .config import TierAConfig
from .prompt import build_prompt
from .stats import AggregateStats, aggregate

logger = logging.getLogger(__name__)


@dataclass
class TierAResult:
    config: TierAConfig
    run_at: str
    num_success: int
    num_failed: int
    wall_time_s: float
    per_request_metrics: list[Metrics]
    aggregate: AggregateStats
    token_count_warnings: int

    def to_json(self) -> str:
        def _metrics_dict(m: Metrics) -> dict:
            return {
                "ttft_ms": m.ttft * 1000,
                "tpot_ms": m.tpot * 1000 if m.tpot is not None else None,
                "e2e_ms": m.e2e * 1000,
                "itl_ms": [v * 1000 for v in m.itl],
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "token_count_warning": m.token_count_warning,
            }

        doc = {
            "run_at": self.run_at,
            "config": dataclasses.asdict(self.config),
            "summary": {
                "num_success": self.num_success,
                "num_failed": self.num_failed,
                "wall_time_s": self.wall_time_s,
                "token_count_warnings": self.token_count_warnings,
            },
            "aggregate": dataclasses.asdict(self.aggregate),
            "per_request": [_metrics_dict(m) for m in self.per_request_metrics],
        }
        return json.dumps(doc, indent=2)


def _make_adapter(config: TierAConfig, tokenizer):
    if config.engine == "vllm":
        return VLLMAdapter(config.base_url, config.model, tokenizer, uds=config.uds)
    if config.engine == "llamacpp":
        return LlamaCppAdapter(config.base_url, config.model, tokenizer, uds=config.uds)
    raise ValueError(f"Unknown engine: {config.engine!r}. Expected 'vllm' or 'llamacpp'.")


async def _run(config: TierAConfig) -> TierAResult:
    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    prompt = build_prompt(tokenizer, config.target_prompt_tokens)
    adapter = _make_adapter(config, tokenizer)
    gen_config = GenerationConfig(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
    )

    logger.info("Warming up with %d requests (discarded)...", config.num_warmup)
    for i in range(config.num_warmup):
        r = await adapter.request(prompt, gen_config)
        if not r.success:
            logger.warning("Warmup request %d failed: %s", i, r.error)

    logger.info("Running %d measurement requests (serial)...", config.num_requests)
    collected: list[Metrics] = []
    num_failed = 0
    token_count_warnings = 0

    wall_start = time.perf_counter()
    for i in range(config.num_requests):
        r = await adapter.request(prompt, gen_config)
        if not r.success:
            logger.warning("Request %d failed: %s", i, r.error)
            num_failed += 1
            continue
        if r.token_count_warning:
            token_count_warnings += 1
        collected.append(compute_metrics(r))
    wall_time = time.perf_counter() - wall_start

    agg = aggregate(collected, wall_time, config.gpu_cost_per_hour)

    return TierAResult(
        config=config,
        run_at=run_at,
        num_success=len(collected),
        num_failed=num_failed,
        wall_time_s=wall_time,
        per_request_metrics=collected,
        aggregate=agg,
        token_count_warnings=token_count_warnings,
    )


def run_tier_a(config: TierAConfig) -> TierAResult:
    return asyncio.run(_run(config))
