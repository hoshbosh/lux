from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass
class GenerationConfig:
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass
class RequestResult:
    t_start: float
    chunk_timestamps: list[float]
    prompt_tokens: int
    completion_tokens: int
    token_count_warning: bool
    success: bool
    error: str | None = None


class EngineAdapter(abc.ABC):
    def __init__(self, base_url: str, model: str, tokenizer: Any, uds: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.tokenizer = tokenizer
        self.uds = uds

    @abc.abstractmethod
    async def request(self, prompt: str, config: GenerationConfig) -> RequestResult: ...
