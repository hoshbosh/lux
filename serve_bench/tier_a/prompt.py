from __future__ import annotations

_BASE = (
    "The quick brown fox jumps over the lazy dog. "
    "Please analyze the following and provide detailed reasoning. "
)


def build_prompt(tokenizer, target_tokens: int) -> str:
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}")

    base_ids = tokenizer.encode(_BASE)
    reps = (target_tokens // len(base_ids)) + 2
    tiled_ids = (base_ids * reps)[:target_tokens]
    return tokenizer.decode(tiled_ids, skip_special_tokens=True)
