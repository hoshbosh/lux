from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TierBConfig:
    """Tier B run configuration. Flat TOML, mirroring TierAConfig.

    Unknown keys are dropped rather than rejected, matching `TierAConfig.from_toml`.
    Validation lives downstream in the layer that can give a useful message, not here.

    Note what is absent: the per-worker semaphore size. Callers state
    `target_concurrency` and the cap is derived by `launcher.cap_for_target`, which sizes
    for the Poisson burst peak rather than the mean. A hand-set cap is how ordinary
    burstiness gets misreported as backpressure drops.
    """

    engine: str
    base_url: str
    model: str
    tokenizer_name: str
    target_prompt_tokens: int
    max_tokens: int
    qps: float
    duration_s: float
    ttft_slo_ms: float

    uds: str | None = None
    seed: int = 0
    num_workers: int = 4
    target_concurrency: int = 64
    max_queue_wait_ms: float | None = None

    itl_slo_ms: float | None = None
    itl_baseline_multiple: float = 2.0
    exclude_first_itl: bool = False

    warmup_window_s: float = 10.0
    warmup_hard_cutoff_s: float = 120.0

    gpu_cost_per_hour: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0

    @classmethod
    def from_toml(cls, path: str | Path) -> TierBConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
