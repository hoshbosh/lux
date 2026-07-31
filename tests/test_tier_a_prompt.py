import pytest

from serve_bench.tier_a.prompt import build_prompt


class CharTokenizer:
    """Character-level tokenizer — each character is one token."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in token_ids)


@pytest.fixture
def tok():
    return CharTokenizer()


def test_prompt_token_count_exact(tok):
    prompt = build_prompt(tok, 100)
    assert len(tok.encode(prompt)) == 100


def test_prompt_token_count_large(tok):
    prompt = build_prompt(tok, 512)
    assert len(tok.encode(prompt)) == 512


def test_prompt_deterministic(tok):
    assert build_prompt(tok, 200) == build_prompt(tok, 200)


def test_prompt_different_targets_differ(tok):
    assert build_prompt(tok, 100) != build_prompt(tok, 200)


def test_prompt_invalid_target(tok):
    with pytest.raises(ValueError):
        build_prompt(tok, 0)

    with pytest.raises(ValueError):
        build_prompt(tok, -1)
