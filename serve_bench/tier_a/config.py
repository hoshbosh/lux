from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TierAConfig:
    engine: str
    base_url: str
    model: str
    tokenizer_name: str
    target_prompt_tokens: int
    max_tokens: int
    uds: str | None = None
    num_warmup: int = 5
    num_requests: int = 20
    gpu_cost_per_hour: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None

    @classmethod
    def from_toml(cls, path: str | Path) -> TierAConfig:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        valid = {field.name for field in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
