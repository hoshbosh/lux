"""Warmup-until-stable: when did this run reach steady state?

vLLM does JIT compilation and CUDA graph capture on its first requests, so the head of
any Tier B run is not the system under test. This module finds the offset where warmup
ended, and Tier B measures only after it.

**This is a post-hoc detector, not a live gate — a deliberate deviation.** The original
design polled a shared cross-process in-flight counter while the run was in progress.
That counter is a coordination hot spot at high QPS: every worker writes it on every
request start and finish, which is contention injected by the measurement apparatus into
the thing being measured. It is also unnecessary. ``RequestOutcome`` already carries
``start_s`` and ``end_s`` for every request, and in-flight depth at any instant is
recoverable from those pairs after the fact. Same signal, zero runtime coordination, and
the detector becomes a pure function that unit tests can drive with synthetic outcomes
instead of processes.

The cost of the trade: a live gate can wait for stability and *then* measure for a fixed
duration, whereas a post-hoc detector spends whatever the schedule gave it. If warmup
eats more of the run than expected, the measurement window is what remains — hence
``WarmupResult.measurement_span_s``, which callers should check before trusting a run.

Two conditions must hold simultaneously over a trailing window, per the resolved design:

1. **In-flight depth has plateaued** — CV (std/mean) of sampled depth below threshold.
2. **TTFT has settled** — CV of TTFT across requests that started in the window.

Both, not either. In-flight count can plateau while TTFT is still falling: the engine
reaches a stable queue depth early, but each request in that queue is still slow because
compilation has not finished. TTFT variance is the gate that catches it.

**CV alone is not sufficient, and this was found by test rather than by design.** The
coefficient of variation measures dispersion, not stationarity. A TTFT sliding steadily
from 900ms to 500ms across a window has a CV near 0.10 — comfortably "stable" — because
the spread within any short window is small next to the mean, even though the series is
still plainly moving. That is precisely the warming-up signature the gate exists to
reject, so each condition additionally requires low *drift*: the gap between the
first-half and second-half means of the window, relative to the window mean. Dispersion
catches noise, drift catches trend, and warmup shows up as either one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import mean, pstdev

from .worker import RequestOutcome

logger = logging.getLogger(__name__)

DEFAULT_HARD_CUTOFF_S = 120.0


@dataclass
class WarmupConfig:
    window_s: float = 10.0  # trailing window both conditions are evaluated over
    sample_interval_s: float = 0.25  # in-flight sampling resolution
    # In-flight dispersion is judged RELATIVE to what a stationary Poisson process would
    # produce, not against a constant. Depth under Poisson arrivals has CV = 1/sqrt(mean)
    # by construction, so a fixed threshold like 0.10 is not a stability criterion at all
    # — it is a hidden demand for mean depth >= 100. A run at 2 concurrent requests has
    # CV ~0.71 no matter how perfectly steady it is, and would never converge.
    in_flight_cv_tolerance: float = 2.0  # multiples of the expected 1/sqrt(mean)
    # TTFT keeps an absolute threshold: it is a continuous latency, not a count, so its
    # dispersion carries no equivalent built-in floor.
    ttft_cv_threshold: float = 0.15
    # Trend gate, applied to both series. |mean(2nd half) - mean(1st half)| / mean, so a
    # monotonic slide fails even when its within-window dispersion is small.
    max_drift_frac: float = 0.10
    # Below this many TTFT samples in a window, a CV is being read off too little data to
    # mean anything, and the window is treated as not-yet-stable rather than stable.
    min_ttft_samples: int = 5
    # Mandatory, not optional: an overloaded engine may never plateau, and without this
    # the detector would consume the entire run looking for a steady state that is not
    # coming.
    hard_cutoff_s: float = DEFAULT_HARD_CUTOFF_S


@dataclass
class StabilitySample:
    """One evaluation point. Retained so a run can show its own convergence, not just
    assert it — the same reason ``run_calibration`` keeps every level it swept."""

    offset_s: float
    in_flight_cv: float | None
    in_flight_cv_expected: float | None  # 1/sqrt(mean depth); the Poisson baseline
    in_flight_drift: float | None
    ttft_cv: float | None
    ttft_drift: float | None
    n_ttft: int
    stable: bool


@dataclass
class WarmupResult:
    config: WarmupConfig
    steady_state_offset_s: float
    converged: bool  # False => the offset below is the cutoff, not a detection
    hit_cutoff: bool
    run_span_s: float
    samples: list[StabilitySample] = field(default_factory=list)

    @property
    def measurement_span_s(self) -> float:
        """Seconds of run left after warmup. A small value means the run was too short,
        not that the engine was fast."""
        return max(0.0, self.run_span_s - self.steady_state_offset_s)


def _in_flight_series(
    outcomes: list[RequestOutcome], t0_perf: float, span_s: float, dt: float
) -> tuple[list[float], list[int]]:
    """Sample in-flight depth on a fixed grid by sweeping start/end events.

    Event sweep rather than counting overlaps per sample point: the naive form is
    O(requests x samples) and a long high-QPS run makes that tens of millions of
    comparisons for a number that is only used to compute a variance.
    """
    events: list[tuple[float, int]] = []
    for o in outcomes:
        if o.start_s is None or o.end_s is None:
            continue  # a dropped arrival was never in flight
        events.append((o.start_s - t0_perf, 1))
        events.append((o.end_s - t0_perf, -1))
    events.sort()

    offsets: list[float] = []
    depths: list[int] = []
    depth = 0
    i = 0
    t = 0.0
    while t <= span_s:
        while i < len(events) and events[i][0] <= t:
            depth += events[i][1]
            i += 1
        offsets.append(t)
        depths.append(depth)
        t += dt
    return offsets, depths


def _cv(values: list[float]) -> float | None:
    """Coefficient of variation. None when undefined rather than 0.0 — a mean of zero
    (nothing in flight) is not evidence of stability."""
    if not values:
        return None
    m = mean(values)
    if m <= 0:
        return None
    return pstdev(values) / m


def _drift(values: list[float]) -> float | None:
    """Fractional movement between the window's first and second half.

    None when undefined (too few points, or a non-positive mean) rather than 0.0: an
    undefined trend is not a flat one, and must not be read as evidence of stability.
    """
    if len(values) < 2:
        return None
    m = mean(values)
    if m <= 0:
        return None
    half = len(values) // 2
    return abs(mean(values[half:]) - mean(values[:half])) / m


def _ttft_by_offset(outcomes: list[RequestOutcome], t0_perf: float) -> list[tuple[float, float]]:
    """(start offset, ttft) for every request that actually produced a first token."""
    pairs: list[tuple[float, float]] = []
    for o in outcomes:
        if o.start_s is None or o.result is None or not o.result.success:
            continue
        ts = o.result.chunk_timestamps
        if not ts:
            continue
        pairs.append((o.start_s - t0_perf, ts[0] - o.result.t_start))
    pairs.sort()
    return pairs


def detect_steady_state(
    outcomes: list[RequestOutcome],
    t0_perf: float,
    run_span_s: float,
    config: WarmupConfig | None = None,
) -> WarmupResult:
    """Find the offset at which both stability conditions first hold together.

    The returned offset is the point at which stability could be *asserted* over the
    preceding window, not the earlier point at which it likely began. That discards a
    window's worth of probably-good data, and it is the conservative direction: it
    matches what a live gate would have done with the same signal, and warmup latency
    leaking into steady-state numbers is the failure this exists to prevent.
    """
    config = config or WarmupConfig()

    if not outcomes:
        raise ValueError("Cannot detect steady state from zero outcomes")

    horizon = min(run_span_s, config.hard_cutoff_s)
    offsets, depths = _in_flight_series(outcomes, t0_perf, run_span_s, config.sample_interval_s)
    ttfts = _ttft_by_offset(outcomes, t0_perf)

    samples: list[StabilitySample] = []
    per_window = max(1, int(config.window_s / config.sample_interval_s))

    for idx, t in enumerate(offsets):
        if t > horizon:
            break
        if t < config.window_s:
            continue  # no full trailing window yet; nothing to be stable over

        depth_window = [float(d) for d in depths[max(0, idx - per_window + 1) : idx + 1]]
        in_flight_cv = _cv(depth_window)
        in_flight_drift = _drift(depth_window)
        mean_depth = mean(depth_window) if depth_window else 0.0
        in_flight_cv_expected = mean_depth ** -0.5 if mean_depth > 0 else None

        lo = t - config.window_s
        window_ttfts = [v for (off, v) in ttfts if lo <= off <= t]
        enough = len(window_ttfts) >= config.min_ttft_samples
        ttft_cv = _cv(window_ttfts) if enough else None
        ttft_drift = _drift(window_ttfts) if enough else None

        measured = [in_flight_cv, in_flight_cv_expected, in_flight_drift, ttft_cv, ttft_drift]
        stable = all(v is not None for v in measured) and (
            in_flight_cv <= config.in_flight_cv_tolerance * in_flight_cv_expected
            and ttft_cv < config.ttft_cv_threshold
            and in_flight_drift < config.max_drift_frac
            and ttft_drift < config.max_drift_frac
        )
        samples.append(
            StabilitySample(
                offset_s=t,
                in_flight_cv=in_flight_cv,
                in_flight_cv_expected=in_flight_cv_expected,
                in_flight_drift=in_flight_drift,
                ttft_cv=ttft_cv,
                ttft_drift=ttft_drift,
                n_ttft=len(window_ttfts),
                stable=stable,
            )
        )
        if stable:
            logger.info(
                "Steady state at %.2fs (in-flight CV %.3f vs Poisson %.3f, drift %.3f; "
                "TTFT CV %.3f drift %.3f over %d requests)",
                t, in_flight_cv, in_flight_cv_expected, in_flight_drift,
                ttft_cv, ttft_drift, len(window_ttfts),
            )
            return WarmupResult(
                config=config,
                steady_state_offset_s=t,
                converged=True,
                hit_cutoff=False,
                run_span_s=run_span_s,
                samples=samples,
            )

    hit_cutoff = run_span_s >= config.hard_cutoff_s
    logger.warning(
        "WARMUP DID NOT CONVERGE within %.0fs (%s). Measuring from the cutoff anyway; "
        "these results may not reflect steady-state performance and should be disclosed "
        "as such. Most likely the engine is overloaded at this offered rate and in-flight "
        "depth never plateaued.",
        horizon,
        "hard cutoff reached" if hit_cutoff else "run ended first",
    )
    return WarmupResult(
        config=config,
        steady_state_offset_s=horizon,
        converged=False,
        hit_cutoff=hit_cutoff,
        run_span_s=run_span_s,
        samples=samples,
    )
