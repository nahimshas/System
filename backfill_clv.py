"""
One-time (re-runnable) CLV backfill — fetches historical closing lines for
every shadow log entry that predates the nightly CLV capture.

Throttled by a per-run credit budget so it never starves daily operations:
run it repeatedly (manually via the CLV Backfill workflow, or just let the
nightly self-heal chip away at it) and it processes the most recent ungraded
waves first, stopping cleanly at the budget. Idempotent — entries already
stamped are skipped, so re-running is always safe.

Usage:
    python backfill_clv.py [--max-credits 900] [--since 2026-04-25]
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill closing-line values (CLV)")
    parser.add_argument("--max-credits", type=int, default=900,
                        help="Odds API credit budget for this run (default 900)")
    parser.add_argument("--since", type=str, default="2026-04-25",
                        help="Earliest pick date to backfill (YYYY-MM-DD)")
    args = parser.parse_args()

    from src.data.closing_lines import (
        repair_missing_commence_times, update_shadow_log_clv,
    )

    # Step 1: repair entries missing commence_time (1 credit per sport-date) —
    # without a start time we can't know which snapshot holds the closing line.
    repaired = repair_missing_commence_times(
        max_credits=min(100, args.max_credits // 4), since=args.since
    )
    logger.info(f"Commence-time repair: {repaired} entries filled")

    # Step 2: fetch closing lines, most recent waves first, within budget.
    summary = update_shadow_log_clv(max_credits=args.max_credits, since=args.since)
    logger.info(
        f"Backfill run complete: {summary['stamped']} stamped, "
        f"{summary['unmatched']} unmatched, ~{summary['credits_spent']} credits, "
        f"{summary['waves']} snapshot(s)"
    )
    if summary["credits_spent"] == 0 and summary["stamped"] == 0:
        logger.info("Nothing left to backfill — all reachable entries are stamped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
