import pytest
from serve_bench.adapter.base import RequestResult
from serve_bench.metrics import compute_metrics


def make_result(t_start: float, chunk_timestamps: list[float], **kwargs) -> RequestResult:
    return RequestResult(
        t_start=t_start,
        chunk_timestamps=chunk_timestamps,
        prompt_tokens=kwargs.get("prompt_tokens", 10),
        completion_tokens=kwargs.get("completion_tokens", len(chunk_timestamps)),
        token_count_warning=kwargs.get("token_count_warning", False),
        success=True,
    )


def test_tpot_excludes_first_inter_chunk_interval():
    # Spec-mandated test: 500ms to first token, then 10ms per subsequent token.
    # TPOT must be ~10ms (mean of itl[1:]), NOT ~167ms (mean of itl[0:]).
    t_start = 0.0
    chunk_timestamps = [0.5, 0.51, 0.52, 0.53, 0.54]
    result = make_result(t_start, chunk_timestamps)
    m = compute_metrics(result)

    assert abs(m.ttft - 0.5) < 1e-9
    assert abs(m.e2e - 0.54) < 1e-9
    assert m.tpot is not None
    assert abs(m.tpot - 0.01) < 1e-9, (
        f"TPOT should be ~10ms (mean of itl[1:]), got {m.tpot * 1000:.3f}ms. "
        "Likely cause: including itl[0] (the LLMperf bug)."
    )


def test_itl_values():
    chunk_timestamps = [0.5, 0.51, 0.52, 0.53, 0.54]
    result = make_result(0.0, chunk_timestamps)
    m = compute_metrics(result)
    assert len(m.itl) == 4
    for interval in m.itl:
        assert abs(interval - 0.01) < 1e-9


def test_ttft_and_e2e():
    result = make_result(1.0, [1.3, 1.4, 1.5])
    m = compute_metrics(result)
    assert abs(m.ttft - 0.3) < 1e-9
    assert abs(m.e2e - 0.5) < 1e-9


def test_single_chunk_tpot_is_none():
    # One chunk → no inter-chunk intervals → TPOT undefined.
    result = make_result(0.0, [0.5])
    m = compute_metrics(result)
    assert m.ttft == 0.5
    assert m.e2e == 0.5
    assert m.itl == []
    assert m.tpot is None


def test_two_chunks_tpot_is_none():
    # Two chunks → one inter-chunk interval → itl[1:] is empty → TPOT undefined.
    result = make_result(0.0, [0.5, 0.51])
    m = compute_metrics(result)
    assert len(m.itl) == 1
    assert m.tpot is None


def test_three_chunks_tpot_defined():
    # Three chunks → two intervals → itl[1:] has one element → TPOT defined.
    result = make_result(0.0, [0.5, 0.51, 0.52])
    m = compute_metrics(result)
    assert m.tpot is not None
    assert abs(m.tpot - 0.01) < 1e-9


def test_empty_chunks_raises():
    result = make_result(0.0, [])
    with pytest.raises(ValueError, match="empty"):
        compute_metrics(result)


def test_token_count_warning_passthrough():
    result = make_result(0.0, [0.1, 0.2], token_count_warning=True)
    m = compute_metrics(result)
    assert m.token_count_warning is True


def test_prompt_and_completion_tokens_passthrough():
    result = make_result(0.0, [0.1, 0.2, 0.3], prompt_tokens=42, completion_tokens=7)
    m = compute_metrics(result)
    assert m.prompt_tokens == 42
    assert m.completion_tokens == 7
