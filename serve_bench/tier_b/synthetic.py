"""An adapter that fabricates a stream instead of calling an engine.

Two jobs, both real rather than test-only scaffolding:

1. **Multi-process launcher tests.** ``spawn`` re-imports the target module in the child,
   so a fake adapter defined inside a test file cannot be reconstructed there. It has to
   live in an importable module, and this is it.
2. **Generator self-calibration.** The B2 sweep measured a load generator whose fire
   callback only slept, which is why its ceiling came back as an unusable lower bound.
   Pointing the *real* worker at this adapter exercises the true dispatch path — the
   semaphore, outcome recording, metric computation — with the engine removed, which is a
   far closer approximation of where a worker actually knees.

It still models a response as sleeps, so it reproduces the event-loop wakeup pattern of
an SSE stream but not its socket reads or HTTP parsing. Numbers from it remain an
optimistic bound on real worker capacity, for the same reason ``calibrate.py`` says so.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from ..adapter.base import EngineAdapter, GenerationConfig, RequestResult


@dataclass
class SyntheticProfile:
    ttft_s: float = 0.050
    itl_s: float = 0.020
    num_chunks: int = 32
    # Fractional jitter applied to both delays, uniform in [-j, +j]. Zero produces a
    # perfectly regular stream, which no real engine does; a nonzero default would make
    # warmup detection tests depend on the seed, so callers opt in.
    jitter_frac: float = 0.0
    failure_rate: float = 0.0
    # Multiplies TTFT on the first `warmup_requests` responses per process, imitating
    # JIT/CUDA-graph capture so the warmup detector can be exercised end to end.
    warmup_requests: int = 0
    warmup_ttft_multiple: float = 1.0


class SyntheticAdapter(EngineAdapter):
    def __init__(self, profile: SyntheticProfile | None = None, seed: int = 0) -> None:
        # base_url/model are inherited but meaningless here; passed so the constructor
        # signature stays substitutable for a real adapter.
        super().__init__(base_url="synthetic://", model="synthetic", tokenizer=None)
        self.profile = profile or SyntheticProfile()
        self._rng = random.Random(seed)
        self._served = 0

    def _jittered(self, base_s: float) -> float:
        j = self.profile.jitter_frac
        if j <= 0:
            return base_s
        return max(0.0, base_s * (1.0 + self._rng.uniform(-j, j)))

    async def request(self, prompt: str, config: GenerationConfig) -> RequestResult:
        p = self.profile
        n = self._served
        self._served += 1

        t_start = time.perf_counter()

        ttft = self._jittered(p.ttft_s)
        if n < p.warmup_requests:
            ttft *= p.warmup_ttft_multiple

        if p.failure_rate > 0 and self._rng.random() < p.failure_rate:
            await asyncio.sleep(ttft)
            return RequestResult(
                t_start=t_start,
                chunk_timestamps=[],
                prompt_tokens=0,
                completion_tokens=0,
                token_count_warning=False,
                success=False,
                error="synthetic failure",
            )

        await asyncio.sleep(ttft)
        chunk_timestamps = [time.perf_counter()]

        chunks = min(p.num_chunks, config.max_tokens)
        for _ in range(max(0, chunks - 1)):
            await asyncio.sleep(self._jittered(p.itl_s))
            chunk_timestamps.append(time.perf_counter())

        return RequestResult(
            t_start=t_start,
            chunk_timestamps=chunk_timestamps,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(chunk_timestamps),
            token_count_warning=False,
            success=True,
        )
