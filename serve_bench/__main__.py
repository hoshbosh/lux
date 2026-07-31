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

    args = parser.parse_args()
    if args.command == "run-tier-a":
        _cmd_run_tier_a(args)
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
