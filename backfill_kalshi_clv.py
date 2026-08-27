#!/usr/bin/env python3
"""
Backfill CLV from Kalshi into the shadow log.

Kalshi retains settled markets with full 1-minute candlestick history well past
our first F5 pick, so this recovers CLV we could never get from the Odds API —
whose historical endpoint does not serve *_1st_5_innings at any price.

ADDITIVE AND IDEMPOTENT: writes only kalshi_* fields, skips entries that
already have them, and never touches `clv` / `market_prob_at_close`, which the
calibration engine and CLV governor read. Safe to interrupt and re-run.

    python3 backfill_kalshi_clv.py --since 2026-08-04              # F5 era
    python3 backfill_kalshi_clv.py --since 2026-07-05 --sports MLB
    python3 backfill_kalshi_clv.py --since 2026-08-01 --max 200    # small bite
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill Kalshi CLV")
    ap.add_argument("--since", default="2026-08-04",
                    help="earliest pick date to backfill (default: first F5 day)")
    ap.add_argument("--max", type=int, default=1500,
                    help="max entries this run (re-run to continue)")
    ap.add_argument("--shadow-only", action="store_true",
                    help="skip the decision-log (candidate) pass")
    ap.add_argument("--recompute", action="store_true",
                    help="re-measure rows that already have kalshi_clv (use after a "
                         "resolution or methodology change)")
    ap.add_argument("--sports", default=None,
                    help="comma-separated sports, e.g. MLB,WNBA (default: all supported)")
    args = ap.parse_args()

    from src.data.kalshi_clv import (update_shadow_log_kalshi_clv,
                                     update_decision_log_kalshi_clv)

    sports = [s.strip().upper() for s in args.sports.split(",")] if args.sports else None
    summary = update_shadow_log_kalshi_clv(since=args.since, max_entries=args.max,
                                           sports=sports, recompute=args.recompute)
    print(f"\nShadow-log result: {summary}")

    # The candidate archive needs CLV too — that is where REJECTED picks live,
    # and measuring those is the reason the decision log exists.
    if not args.shadow_only:
        dsum = update_decision_log_kalshi_clv(since=args.since, max_entries=args.max,
                                              recompute=args.recompute)
        print(f"Decision-log result: {dsum}")
        for k in summary:
            summary[k] = summary.get(k, 0) + dsum.get(k, 0)
    if summary["stamped"] == 0 and summary["unmatched"] == 0 and summary["no_candles"] == 0:
        print("Nothing left to backfill for this window.")
    else:
        print("Re-run to continue if entries remain (idempotent).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
