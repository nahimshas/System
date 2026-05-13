"""
Model performance analysis — reads settled history and produces calibration metrics.

Usage:
  python3 backtest.py              # full report
  python3 backtest.py --sport MLB  # filter by sport
  python3 backtest.py --since 2026-05-01  # from a date
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HISTORY_FILE    = Path("state/history.json")
PROP_HISTORY    = Path("state/prop_history.json")


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(n, d):
    return f"{n/d*100:.1f}%" if d else "n/a"

def _roi(pnl, cost):
    return f"{pnl/cost*100:+.1f}%" if cost else "n/a"

def _bar(win_rate, width=20):
    filled = round(win_rate * width)
    return "█" * filled + "░" * (width - filled)

def _section(title):
    print()
    print(f"{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── loaders ──────────────────────────────────────────────────────────────────

def load_singles(sport=None, since=None):
    records = json.loads(HISTORY_FILE.read_text())
    if sport:
        records = [r for r in records if r.get("sport","").upper() == sport.upper()]
    if since:
        records = [r for r in records if r.get("date","") >= since]
    # Separate true singles from parlays
    singles = [r for r in records if r.get("type") != "parlay" and r.get("bet_type") != "Parlay"]
    parlays = [r for r in records if r.get("type") == "parlay" or r.get("bet_type") == "Parlay"]
    return singles, parlays


def load_props(sport=None, since=None):
    records = json.loads(PROP_HISTORY.read_text())
    if sport:
        records = [r for r in records if r.get("sport","").upper() == sport.upper()]
    if since:
        records = [r for r in records if r.get("date","") >= since]
    return records


# ── singles analysis ─────────────────────────────────────────────────────────

def summary(singles, parlays):
    _section("OVERALL SUMMARY")
    for label, recs in [("Singles", singles), ("Parlays", parlays)]:
        if not recs:
            continue
        wins   = sum(1 for r in recs if r.get("result") == "WON")
        losses = sum(1 for r in recs if r.get("result") == "LOST")
        total  = wins + losses
        pnl    = sum(r.get("actual_pnl", 0) for r in recs)
        cost   = sum(r.get("cost", 0) for r in recs if r.get("result") in ("WON","LOST"))
        print(f"\n  {label}: {total} settled | {wins}W {losses}L | "
              f"Win rate {_pct(wins,total)} | ROI {_roi(pnl,cost)} | P&L ${pnl:+.2f}")


def by_confidence(singles):
    _section("BY CONFIDENCE TIER")
    print(f"  {'Tier':<10} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>7} {'ROI':>8} {'P&L':>9}")
    print(f"  {'-'*55}")
    for conf in ["HIGH", "MEDIUM"]:
        recs  = [r for r in singles if r.get("confidence") == conf
                 and r.get("result") in ("WON","LOST")]
        if not recs:
            continue
        wins  = sum(1 for r in recs if r.get("result") == "WON")
        losses = len(recs) - wins
        pnl   = sum(r.get("actual_pnl", 0) for r in recs)
        cost  = sum(r.get("cost", 0) for r in recs)
        print(f"  {conf:<10} {len(recs):>5} {wins:>4} {losses:>4} "
              f"{_pct(wins,len(recs)):>7} {_roi(pnl,cost):>8} ${pnl:>+8.2f}")


def by_sport(singles):
    _section("BY SPORT")
    print(f"  {'Sport':<8} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>7} {'ROI':>8} {'P&L':>9}")
    print(f"  {'-'*55}")
    sports = sorted(set(r.get("sport","?") for r in singles))
    for sport in sports:
        recs  = [r for r in singles if r.get("sport") == sport
                 and r.get("result") in ("WON","LOST")]
        if not recs:
            continue
        wins  = sum(1 for r in recs if r.get("result") == "WON")
        pnl   = sum(r.get("actual_pnl", 0) for r in recs)
        cost  = sum(r.get("cost", 0) for r in recs)
        print(f"  {sport:<8} {len(recs):>5} {wins:>4} {len(recs)-wins:>4} "
              f"{_pct(wins,len(recs)):>7} {_roi(pnl,cost):>8} ${pnl:>+8.2f}")


def by_bet_type(singles):
    _section("BY BET TYPE")
    print(f"  {'Type':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>7} {'ROI':>8} {'P&L':>9}")
    print(f"  {'-'*55}")
    types = sorted(set(r.get("bet_type","?") for r in singles))
    for bt in types:
        recs  = [r for r in singles if r.get("bet_type") == bt
                 and r.get("result") in ("WON","LOST")]
        if not recs:
            continue
        wins  = sum(1 for r in recs if r.get("result") == "WON")
        pnl   = sum(r.get("actual_pnl", 0) for r in recs)
        cost  = sum(r.get("cost", 0) for r in recs)
        print(f"  {bt:<12} {len(recs):>5} {wins:>4} {len(recs)-wins:>4} "
              f"{_pct(wins,len(recs)):>7} {_roi(pnl,cost):>8} ${pnl:>+8.2f}")


def calibration(singles):
    """
    Core calibration check: when the model says X%, does it actually win X%?
    Buckets records by model_prob_pct and compares predicted vs actual win rate.
    A well-calibrated model should land close to the diagonal.
    """
    _section("CALIBRATION — Predicted vs Actual Win Rate")
    print(f"  {'Model prob':>12} {'Bets':>5} {'Actual win%':>12}  Visual")
    print(f"  {'-'*55}")

    buckets = defaultdict(list)
    for r in singles:
        if r.get("result") not in ("WON", "LOST"):
            continue
        prob = r.get("model_prob_pct", 0)
        # Bucket into 10pp bands
        bucket = int(prob // 10) * 10
        bucket = max(40, min(90, bucket))   # clip to [40,90]
        buckets[bucket].append(r)

    for lo in sorted(buckets):
        hi   = lo + 10
        recs = buckets[lo]
        wins = sum(1 for r in recs if r.get("result") == "WON")
        actual = wins / len(recs) if recs else 0
        mid_pred = (lo + 5) / 100   # midpoint of bucket as fraction
        bar  = _bar(actual)
        diff = actual - mid_pred
        flag = "  ⚠ overconfident" if diff < -0.10 else ("  ✓ underconfident+" if diff > 0.10 else "")
        print(f"  {lo}–{hi}%       {len(recs):>5}   {actual*100:>6.1f}%       {bar}{flag}")

    print()
    print("  Interpretation: each row should win close to its model prob range.")
    print("  Consistently below → model is overconfident (edges aren't as large as model thinks).")
    print("  Consistently above → model is underconfident (real edges are larger).")


def by_edge_bucket(singles):
    _section("BY EDGE SIZE")
    print(f"  {'Edge':>10} {'Bets':>5} {'W':>4} {'L':>4} {'Win%':>7} {'ROI':>8} {'P&L':>9}")
    print(f"  {'-'*55}")
    buckets = [(5,10), (10,15), (15,20), (20,30), (30,100)]
    for lo, hi in buckets:
        recs = [r for r in singles
                if lo <= r.get("edge_pct", 0) < hi
                and r.get("result") in ("WON","LOST")]
        if not recs:
            continue
        wins = sum(1 for r in recs if r.get("result") == "WON")
        pnl  = sum(r.get("actual_pnl", 0) for r in recs)
        cost = sum(r.get("cost", 0) for r in recs)
        label = f"{lo}–{hi}%"
        print(f"  {label:>10} {len(recs):>5} {wins:>4} {len(recs)-wins:>4} "
              f"{_pct(wins,len(recs)):>7} {_roi(pnl,cost):>8} ${pnl:>+8.2f}")


def recent_form(singles, n=20):
    """Last N settled bets — quick read on whether things are trending."""
    _section(f"RECENT FORM (last {n} settled bets)")
    settled = [r for r in singles if r.get("result") in ("WON","LOST")]
    recent  = settled[-n:]
    if not recent:
        print("  No settled bets yet.")
        return
    wins  = sum(1 for r in recent if r.get("result") == "WON")
    pnl   = sum(r.get("actual_pnl", 0) for r in recent)
    cost  = sum(r.get("cost", 0) for r in recent)
    streak_char = []
    for r in recent[-10:]:
        streak_char.append("✅" if r.get("result") == "WON" else "❌")
    print(f"  Last {len(recent)} bets: {wins}W {len(recent)-wins}L | "
          f"Win rate {_pct(wins,len(recent))} | ROI {_roi(pnl,cost)} | P&L ${pnl:+.2f}")
    print(f"  Last 10: {''.join(streak_char)}")


# ── props analysis ────────────────────────────────────────────────────────────

def props_summary(props):
    _section("PROPS SUMMARY")
    settled = [r for r in props if "hit" in r]
    if not settled:
        print("  No settled props.")
        return
    hits  = sum(1 for r in settled if r.get("hit"))
    total = len(settled)
    print(f"\n  Total settled: {total} | Hit rate: {_pct(hits,total)}")

    # By confidence
    print(f"\n  {'Tier':<10} {'Props':>6} {'Hits':>6} {'Hit%':>7}")
    print(f"  {'-'*35}")
    for conf in ["HIGH", "MEDIUM"]:
        recs = [r for r in settled if r.get("confidence") == conf]
        if not recs:
            continue
        h = sum(1 for r in recs if r.get("hit"))
        print(f"  {conf:<10} {len(recs):>6} {h:>6} {_pct(h,len(recs)):>7}")

    # By sport
    print(f"\n  {'Sport':<8} {'Props':>6} {'Hits':>6} {'Hit%':>7}")
    print(f"  {'-'*35}")
    for sport in sorted(set(r.get("sport","?") for r in settled)):
        recs = [r for r in settled if r.get("sport") == sport]
        h = sum(1 for r in recs if r.get("hit"))
        print(f"  {sport:<8} {len(recs):>6} {h:>6} {_pct(h,len(recs)):>7}")

    # Model accuracy: how far off was the projected line from actual
    print(f"\n  {'Prop type':<30} {'Count':>6} {'Avg error':>10} {'Hit%':>7}")
    print(f"  {'-'*58}")
    by_type = defaultdict(list)
    for r in settled:
        pt = r.get("prop_type","?")
        ml = r.get("model_line")
        ac = r.get("actual_stat")
        if ml is not None and ac is not None:
            by_type[pt].append((r, abs(float(ml) - float(ac))))
    for pt in sorted(by_type):
        recs_e = by_type[pt]
        avg_err = sum(e for _, e in recs_e) / len(recs_e)
        h = sum(1 for r, _ in recs_e if r.get("hit"))
        print(f"  {pt:<30} {len(recs_e):>6} {avg_err:>9.2f}  {_pct(h,len(recs_e)):>7}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Model performance analysis")
    parser.add_argument("--sport", help="Filter by sport (MLB, NBA, etc.)")
    parser.add_argument("--since", help="Only include bets from this date onward (YYYY-MM-DD)")
    args = parser.parse_args()

    singles, parlays = load_singles(sport=args.sport, since=args.since)
    props            = load_props(sport=args.sport, since=args.since)

    date_range = ""
    if args.since:
        date_range = f" (since {args.since})"
    sport_filter = f" [{args.sport}]" if args.sport else ""

    print(f"\n{'═'*60}")
    print(f"  MODEL PERFORMANCE REPORT{sport_filter}{date_range}")
    print(f"{'═'*60}")

    summary(singles, parlays)
    by_confidence(singles)
    by_sport(singles)
    by_bet_type(singles)
    calibration(singles)
    by_edge_bucket(singles)
    recent_form(singles)
    props_summary(props)

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
