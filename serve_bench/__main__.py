from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="serve_bench")
    sub = parser.add_subparsers(dest="command")

    ta = sub.add_parser("run-tier-a", help="Run Tier A serial closed-loop benchmark")
    ta.add_argument("config", type=Path, help="Path to TOML config file")
    ta.add_argument("--out", type=Path, default=None, help="Output JSON path (default: next to config)")

    cb = sub.add_parser(
        "calibrate-tier-b",
        help="Sweep a single worker to find its concurrency ceiling (the B2 experiment)",
    )
    cb.add_argument("--levels", type=str, default=None,
                    help="Comma-separated concurrency levels (default: 10,25,50,100,200,350,500)")
    cb.add_argument("--duration", type=float, default=20.0, help="Seconds per level")
    cb.add_argument("--chunks", type=int, default=64, help="Synthetic chunks per request")
    cb.add_argument("--itl-ms", type=float, default=20.0, help="Synthetic inter-chunk latency (ms)")
    cb.add_argument("--threshold-ms", type=float, default=2.0,
                    help="p99 inter-arrival error above which a level fails")
    cb.add_argument("--seed", type=int, default=0)
    cb.add_argument("--out", type=Path, default=None, help="Output JSON path")

    args = parser.parse_args()
    if args.command == "run-tier-a":
        _cmd_run_tier_a(args)
    elif args.command == "calibrate-tier-b":
        _cmd_calibrate_tier_b(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_run_tier_a(args: argparse.Namespace) -> None:
    from .tier_a import TierAConfig, run_tier_a

    config = TierAConfig.from_toml(args.config)
    result = run_tier_a(config)

    out_path: Path = args.out or args.config.parent / f"{result.run_at}_tier_a.json"
    out_path.write_text(result.to_json())
    print(f"\nResults written to {out_path}")
    _print_summary(result)


def _cmd_calibrate_tier_b(args: argparse.Namespace) -> None:
    from .tier_b import DEFAULT_LEVELS, CalibrationConfig, run_calibration

    levels = (
        [int(x) for x in args.levels.split(",") if x.strip()]
        if args.levels
        else list(DEFAULT_LEVELS)
    )
    config = CalibrationConfig(
        concurrency_levels=levels,
        duration_s=args.duration,
        chunks_per_request=args.chunks,
        itl_s=args.itl_ms / 1000,
        p99_error_threshold_ms=args.threshold_ms,
        seed=args.seed,
    )
    result = run_calibration(config)

    out_path: Path = args.out or Path("tier_b_calibration.json")
    out_path.write_text(result.to_json())
    _print_calibration(result)
    print(f"\nResults written to {out_path}")


def _print_calibration(result) -> None:
    config = result.config
    print(f"\n{'=' * 72}")
    print("Tier B worker calibration  |  synthetic load, no engine")
    print(f"{'=' * 72}")
    print(f"Request model: {config.chunks_per_request} chunks x {config.itl_s * 1000:.0f}ms "
          f"= {config.request_duration_s:.2f}s per request")
    print(f"Threshold    : p99 inter-arrival error <= {config.p99_error_threshold_ms:.1f}ms")
    print()
    print(f"{'Conc':>6} {'QPS':>8} {'Arrivals':>9} {'err p50':>9} {'err p99':>9} "
          f"{'peak':>6} {'':>6}")
    print(f"{'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*6}")
    for p in result.points:
        flag = "ok" if p.passed else "FAIL"
        if p.low_sample_warning:
            flag += "?"
        print(f"{p.target_concurrency:>6} {p.offered_qps:>8.1f} {p.num_arrivals:>9} "
              f"{p.timing.interarrival_abs_error.p50_ms:>9.3f} "
              f"{p.timing.interarrival_abs_error.p99_ms:>9.3f} "
              f"{p.peak_in_flight:>6} {flag:>6}")
    print()

    if result.per_worker_ceiling is None:
        print("Per-worker ceiling: NONE FOUND — every level exceeded the threshold.")
        print("The knee is below the swept range, or this host is too noisy to calibrate on.")
        return

    qualifier = " (lower bound — no level failed, extend the sweep)" if result.ceiling_is_lower_bound else ""
    print(f"Per-worker ceiling: {result.per_worker_ceiling}{qualifier}")
    print()
    print("Derived worker counts  N = ceil(target_concurrency / ceiling):")
    for target in (64, 128, 256, 512, 1024):
        print(f"  target {target:>5} concurrent -> N = {result.workers_for(target)}")
    print()
    print("NOTE: synthetic sleeps only — no sockets, no HTTP parsing, no tokenizer.")
    print("This ceiling is an OPTIMISTIC bound; a real worker knees at or below it.")


def _print_summary(result) -> None:
    agg = result.aggregate
    print(f"\n{'=' * 60}")
    print(f"Tier A  |  {result.config.engine}  |  {result.config.model}")
    print(f"{'=' * 60}")
    print(f"Requests : {result.num_success} ok, {result.num_failed} failed"
          + (f"  [{result.token_count_warnings} token count warnings]" if result.token_count_warnings else ""))
    print(f"Wall time: {result.wall_time_s:.1f}s")
    print()
    print(f"{'Metric':<12} {'P50':>8} {'P95':>8} {'P99':>8} {'Mean':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, s in [("TTFT (ms)", agg.ttft), ("E2E (ms)", agg.e2e)]:
        print(f"{label:<12} {s.p50_ms:>8.1f} {s.p95_ms:>8.1f} {s.p99_ms:>8.1f} {s.mean_ms:>8.1f}")
    if agg.tpot:
        s = agg.tpot
        print(f"{'TPOT (ms)':<12} {s.p50_ms:>8.1f} {s.p95_ms:>8.1f} {s.p99_ms:>8.1f} {s.mean_ms:>8.1f}")
    print()
    print(f"Throughput: {agg.throughput_tok_per_s:.1f} tok/s")
    if agg.cost_per_mtok_usd is not None:
        print(f"Cost:       ${agg.cost_per_mtok_usd:.4f}/MTok")


if __name__ == "__main__":
    main()
