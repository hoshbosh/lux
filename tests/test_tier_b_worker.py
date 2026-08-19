import asyncio
import json
import time

from serve_bench.adapter.base import EngineAdapter, GenerationConfig, RequestResult
from serve_bench.tier_b.worker import (
    FAILED,
    INCOMPLETE_AT_WINDOW_END,
    OUTSIDE_WINDOW,
    MeasurementWindow,
    RequestOutcome,
    WorkerConfig,
    classify_outcome,
    run_worker,
)

GEN = GenerationConfig(max_tokens=8)


class FakeAdapter(EngineAdapter):
    """In-process stand-in for a streaming engine. No sockets, no tokenizer.

    duration_s is spent as `chunks` async sleeps so the event loop behaves the way it
    would under a real SSE stream rather than under one long sleep.
    """

    def __init__(self, duration_s: float = 0.01, chunks: int = 4, fail_every: int = 0,
                 raise_every: int = 0, empty_every: int = 0):
        super().__init__("http://fake", "fake-model", tokenizer=None)
        self.duration_s = duration_s
        self.chunks = chunks
        self.fail_every = fail_every
        self.raise_every = raise_every
        self.empty_every = empty_every
        self.calls = 0
        self.in_flight = 0
        self.peak_in_flight = 0

    async def request(self, prompt: str, config: GenerationConfig) -> RequestResult:
        self.calls += 1
        n = self.calls
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            t_start = time.perf_counter()
            per_chunk = self.duration_s / max(self.chunks, 1)
            timestamps = []
            for _ in range(self.chunks):
                await asyncio.sleep(per_chunk)
                timestamps.append(time.perf_counter())
            if self.raise_every and n % self.raise_every == 0:
                raise ConnectionError("fake transport blew up")
            if self.fail_every and n % self.fail_every == 0:
                return RequestResult(
                    t_start=t_start,
                    chunk_timestamps=timestamps,
                    prompt_tokens=10,
                    completion_tokens=len(timestamps),
                    token_count_warning=False,
                    success=False,
                    error="fake 500",
                )
            if self.empty_every and n % self.empty_every == 0:
                timestamps = []
            return RequestResult(
                t_start=t_start,
                chunk_timestamps=timestamps,
                prompt_tokens=10,
                completion_tokens=len(timestamps),
                token_count_warning=False,
                success=True,
            )
        finally:
            self.in_flight -= 1


def make_config(**kwargs) -> WorkerConfig:
    base = dict(concurrency_cap=4, prompt="hello", generation=GEN)
    base.update(kwargs)
    return WorkerConfig(**base)


def make_outcome(queue_start_s, start_s, end_s, success=True) -> RequestOutcome:
    result = None
    if start_s is not None:
        result = RequestResult(
            t_start=start_s,
            chunk_timestamps=[start_s + 0.001],
            prompt_tokens=1,
            completion_tokens=1,
            token_count_warning=False,
            success=success,
        )
    return RequestOutcome(
        index=0,
        queue_start_s=queue_start_s,
        queue_wait_s=0.0 if start_s is None else start_s - queue_start_s,
        start_s=start_s,
        end_s=end_s,
        result=result,
        error=None,
    )


# --- window rule (pure classification) ---------------------------------------------

WINDOW = MeasurementWindow(start_offset_s=1.0, end_offset_s=5.0)


def test_request_completing_after_window_is_discarded_not_dropped():
    # THE hard AND rule. Started at +2s, finished at +6s with the window closing at +5s.
    # Likely cause of failure: treating "did not finish in the window" as a drop, which
    # would inflate the drop count with requests the server handled correctly, or as a
    # success, which would count a latency the window never observed.
    o = make_outcome(queue_start_s=101.9, start_s=102.0, end_s=106.0)
    assert classify_outcome(o, t0_perf=100.0, window=WINDOW) == INCOMPLETE_AT_WINDOW_END


def test_request_starting_before_window_is_outside_even_if_it_finishes_inside():
    o = make_outcome(queue_start_s=100.4, start_s=100.5, end_s=103.0)
    assert classify_outcome(o, t0_perf=100.0, window=WINDOW) == OUTSIDE_WINDOW


def test_failed_request_inside_window_is_failed_not_success():
    o = make_outcome(queue_start_s=100.9, start_s=101.0, end_s=102.0, success=False)
    assert classify_outcome(o, t0_perf=100.0, window=WINDOW) == FAILED


async def test_in_flight_never_exceeds_concurrency_cap():
    # 20 arrivals 5ms apart against a 100ms adapter: without a semaphore ~15 would be in
    # flight at once. Likely cause of failure: acquiring the permit outside the fire
    # callback, or releasing it before the response completes.
    offsets = [(i, 0.005 * i) for i in range(20)]
    adapter = FakeAdapter(duration_s=0.1, chunks=4)
    result = await run_worker(offsets, adapter, make_config(concurrency_cap=3))

    assert adapter.peak_in_flight <= 3
    assert adapter.peak_in_flight == 3  # the cap was actually reached, so this proves it
    assert result.num_dispatched == 20
    assert result.num_success == 20


async def test_backpressure_drops_are_counted_and_never_issued():
    # cap=1 against a 200ms adapter with a 50ms patience: only the requests that get a
    # permit fast enough are issued, the rest are explicit drops.
    offsets = [(i, 0.005 * i) for i in range(8)]
    adapter = FakeAdapter(duration_s=0.2, chunks=2)
    result = await run_worker(
        offsets, adapter, make_config(concurrency_cap=1, max_queue_wait_s=0.05)
    )

    assert result.num_dropped_backpressure > 0
    # A drop is not a request: the adapter never saw it.
    assert adapter.calls == result.num_success + result.num_failed
    assert result.num_dispatched == 8
    assert (
        result.num_success
        + result.num_failed
        + result.num_dropped_backpressure
        + result.num_incomplete_at_window_end
        + result.num_outside_window
    ) == 8


async def test_window_start_discards_warmup_requests():
    offsets = [(i, 0.02 * i) for i in range(10)]
    adapter = FakeAdapter(duration_s=0.001, chunks=1)
    result = await run_worker(
        offsets,
        adapter,
        make_config(window=MeasurementWindow(start_offset_s=0.1)),
    )
    # Arrivals at 0, 20, 40, 60, 80ms start before the window opens.
    assert result.num_outside_window == 5
    assert result.num_success == 5
    assert result.num_dropped_backpressure == 0


async def test_window_end_discards_requests_still_running():
    # Every request takes 200ms; the window closes at 50ms. Nothing can both start and
    # finish inside it, and none of that is a drop.
    offsets = [(i, 0.005 * i) for i in range(5)]
    adapter = FakeAdapter(duration_s=0.2, chunks=2)
    result = await run_worker(
        offsets,
        adapter,
        make_config(concurrency_cap=8, window=MeasurementWindow(end_offset_s=0.05)),
    )
    assert result.num_incomplete_at_window_end == 5
    assert result.num_success == 0
    assert result.num_dropped_backpressure == 0
    assert result.per_request_metrics == []


# --- failures ------------------------------------------------------------------------


async def test_adapter_exception_is_a_failure_not_a_lost_arrival():
    # An exception inside the adapter must not escape as a fire exception: that would
    # lose the arrival from the accounting entirely.
    offsets = [(i, 0.002 * i) for i in range(4)]
    adapter = FakeAdapter(duration_s=0.001, chunks=1, raise_every=2)
    result = await run_worker(offsets, adapter, make_config())

    assert result.num_fire_exceptions == 0
    assert result.num_failed == 2
    assert result.num_success == 2
    assert any(o.error and "ConnectionError" in o.error for o in result.outcomes)


async def test_exception_still_releases_its_permit():
    # cap=1 and every request raises: without release-in-finally the second arrival would
    # wait forever and the test would hang / drop everything.
    offsets = [(i, 0.002 * i) for i in range(4)]
    adapter = FakeAdapter(duration_s=0.001, chunks=1, raise_every=1)
    result = await run_worker(
        offsets, adapter, make_config(concurrency_cap=1, max_queue_wait_s=1.0)
    )
    assert result.num_failed == 4
    assert result.num_dropped_backpressure == 0


async def test_success_with_no_chunks_is_demoted_to_failed():
    # compute_metrics cannot describe a response with no tokens; counting it as a success
    # would enter a zero-latency request into the percentiles.
    offsets = [(i, 0.002 * i) for i in range(4)]
    adapter = FakeAdapter(duration_s=0.001, chunks=1, empty_every=2)
    result = await run_worker(offsets, adapter, make_config())

    assert result.num_success == 2
    assert result.num_failed == 2
    assert len(result.per_request_metrics) == 2


# --- result surface -------------------------------------------------------------------


async def test_metrics_are_computed_for_successes():
    offsets = [(i, 0.002 * i) for i in range(3)]
    adapter = FakeAdapter(duration_s=0.02, chunks=5)
    result = await run_worker(offsets, adapter, make_config())

    assert len(result.per_request_metrics) == 3
    for m in result.per_request_metrics:
        assert m.ttft > 0
        assert m.e2e >= m.ttft
        assert m.tpot is not None  # 5 chunks is enough for a TPOT
        assert len(m.itl) == 4


async def test_epoch_pair_is_preserved_for_a_future_multiprocess_layer():
    offsets = [(0, 0.001)]
    result = await run_worker(offsets, FakeAdapter(duration_s=0.001), make_config())
    assert result.t0_wall > 1_600_000_000  # a real wall-clock epoch, not a perf value
    assert result.t0_perf != result.t0_wall
    # Mapping a perf-domain value onto the common epoch.
    o = result.outcomes[0]
    wall_start = result.t0_wall + (o.start_s - result.t0_perf)
    assert result.t0_wall <= wall_start <= result.t0_wall + 5


async def test_to_json_reports_the_two_drop_causes_separately():
    offsets = [(i, 0.005 * i) for i in range(6)]
    adapter = FakeAdapter(duration_s=0.2, chunks=2)
    result = await run_worker(
        offsets, adapter, make_config(concurrency_cap=1, max_queue_wait_s=0.05)
    )
    doc = json.loads(result.to_json())

    assert doc["dropped"]["backpressure"] == result.num_dropped_backpressure
    # null, not 0: this layer evaluates no SLO, and claiming zero SLO drops would be a
    # measurement that never happened.
    assert doc["dropped"]["slo"] is None
    assert "dropped" not in doc["summary"]  # never folded into a single number
    assert doc["summary"]["num_success"] == result.num_success
    assert doc["epoch"]["t0_perf"] == result.t0_perf
