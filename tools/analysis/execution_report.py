"""
Execution report — does buying at the BID fix MLB, or does adverse selection
eat the saving?

Read-only. Run any time:  python3 -m tools.analysis.execution_report

Reads state/execution_log/*.json (see src/state/execution_log.py). Three things
matter, in order:

  1. FILL RATE — how often a resting bid would have been reached. This is an
     optimistic UPPER BOUND (queue position is ignored), so treat it as a
     ceiling, not an expectation.
  2. SAVING — clv_if_bid minus clv_if_ask, i.e. what the spread is worth.
  3. ADVERSE SELECTION — clv_if_bid on the picks that WOULD have filled versus
     on all picks. Resting orders fill preferentially when the price is about
     to move against you, so if fills are adversely selected the first number
     is worse than the second, and the difference is the real cost of waiting.

A positive average clv_if_bid on filled picks is the bar MLB has to clear.
"""
from __future__ import annotations

import sys
from collections import defaultdict

from src.state.execution_log import FEES, load_entries


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, pct=True):
    if x is None:
        return "   n/a"
    return f"{x * 100:+6.2f}pp" if pct else f"{x:6.2f}"


def main(since: str = "0000-00-00") -> int:
    rows = [e for e in load_entries(since) if e.get("close_price") is not None]
    if not rows:
        print("No settled execution rows yet — the log fills in after games start.")
        print("(Snapshots are taken with the morning card; settlement needs the")
        print(" game to have begun so Kalshi has candlesticks to replay.)")
        return 0

    print(f"=== EXECUTION REPORT — {len(rows)} settled picks (since {since}) ===\n")

    filled = [e for e in rows if e.get("bid_would_fill")]
    fill_rate = len(filled) / len(rows)
    print(f"[1] Fill rate at the bid : {fill_rate*100:.1f}%  ({len(filled)}/{len(rows)})")
    print( "    (upper bound — ignores queue position)\n")

    clv_bid_all = _avg([e.get("clv_if_bid") for e in rows])
    clv_ask_all = _avg([e.get("clv_if_ask") for e in rows])
    print(f"[2] CLV if we always took the ask : {_fmt(clv_ask_all)}")
    print(f"    CLV if we always got the bid  : {_fmt(clv_bid_all)}")
    if clv_bid_all is not None and clv_ask_all is not None:
        print(f"    Spread is worth               : {_fmt(clv_bid_all - clv_ask_all)}\n")

    clv_bid_filled = _avg([e.get("clv_if_bid") for e in filled])
    print(f"[3] CLV at bid, ALL picks         : {_fmt(clv_bid_all)}")
    print(f"    CLV at bid, picks that FILLED : {_fmt(clv_bid_filled)}")
    if clv_bid_all is not None and clv_bid_filled is not None:
        adv = clv_bid_filled - clv_bid_all
        print(f"    Adverse selection cost        : {_fmt(adv)}")
        print(f"    -> {'fills are adversely selected' if adv < 0 else 'no adverse selection detected'}\n")

    verdict = clv_bid_filled
    print("=== VERDICT ===")
    if verdict is None:
        print("  Not enough filled picks yet.")
    elif verdict > 0:
        print(f"  Buying at the bid is +EV on fills ({_fmt(verdict)} after ${FEES:.2f} fees).")
        print("  Requires a time-split confirmation before it means anything.")
    else:
        print(f"  Even at the bid we are underwater ({_fmt(verdict)} after ${FEES:.2f} fees).")
        print("  Execution alone does not make MLB profitable.")

    by_type = defaultdict(list)
    for e in rows:
        by_type[e.get("bet_type", "?")].append(e)
    print("\n=== BY MARKET ===")
    print(f"{'market':<16}{'n':>4}{'fill%':>8}{'clv@bid':>10}{'clv@ask':>10}{'med OI':>10}")
    for k, v in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        fr = sum(1 for e in v if e.get("bid_would_fill")) / len(v)
        ois = sorted(e.get("open_interest_at_pick") or 0 for e in v)
        print(f"{k[:15]:<16}{len(v):>4}{fr*100:>7.0f}%"
              f"{_fmt(_avg([e.get('clv_if_bid') for e in v])):>10}"
              f"{_fmt(_avg([e.get('clv_if_ask') for e in v])):>10}"
              f"{ois[len(ois)//2]:>10,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "0000-00-00"))
