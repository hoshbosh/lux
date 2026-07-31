from __future__ import annotations

import json
import time
from typing import Any

import httpx

from .base import EngineAdapter, GenerationConfig, RequestResult


class LlamaCppAdapter(EngineAdapter):
    def __init__(self, base_url: str, model: str, tokenizer: Any, uds: str | None = None) -> None:
        super().__init__(base_url, model, tokenizer, uds=uds)

    async def request(self, prompt: str, config: GenerationConfig) -> RequestResult:
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        if config.seed is not None:
            body["seed"] = config.seed
        if config.stop:
            body["stop"] = config.stop

        prompt_tokens = len(self.tokenizer.encode(prompt))
        chunks: list[str] = []
        chunk_timestamps: list[float] = []
        sse_completion_tokens: int | None = None

        t_start = time.perf_counter()
        try:
            transport = httpx.AsyncHTTPTransport(uds=self.uds) if self.uds else None
            async with httpx.AsyncClient(transport=transport) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=body,
                    timeout=300.0,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        data = json.loads(payload)
                        # llama.cpp sends a usage chunk AFTER finish_reason=stop but
                        # BEFORE [DONE]. We must NOT break on finish_reason=stop —
                        # doing so is a known client bug that causes token_count_warning
                        # to fire on every request because the usage field is never read.
                        usage = data.get("usage")
                        if usage:
                            sse_completion_tokens = usage.get("completion_tokens")
                            continue
                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            chunk_timestamps.append(time.perf_counter())
                            chunks.append(content)
        except Exception as exc:
            return RequestResult(
                t_start=t_start,
                chunk_timestamps=chunk_timestamps,
                prompt_tokens=prompt_tokens,
                completion_tokens=0,
                token_count_warning=False,
                success=False,
                error=str(exc),
            )

        full_text = "".join(chunks)
        retokenized = len(self.tokenizer.encode(full_text))
        token_count_warning = (
            sse_completion_tokens is not None and retokenized != sse_completion_tokens
        )

        return RequestResult(
            t_start=t_start,
            chunk_timestamps=chunk_timestamps,
            prompt_tokens=prompt_tokens,
            completion_tokens=retokenized,
            token_count_warning=token_count_warning,
            success=True,
        )
