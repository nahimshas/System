"""
Brute-force hunt for ACTUALLY PROFITABLE patterns — ROI first, CLV ignored.

Premise (user, Aug 4 2026): "a winning bet is still winning even if there is
no edge." Fair — profit is the goal, CLV is only a leading indicator. So this
searches the logged record directly for segments that MADE MONEY.

Two honest guardrails, both learned the hard way:
  1. ROI, not win rate. A 70%-win-rate segment at -300 prices loses money.
     Every segment is scored as flat-stake ROI at realistic 4.5% friction.
  2. TIME SPLIT. Any segment is scored on an early window (discovery) AND a
     later held-out window (validation). `pattern_card` looked like +11.7% on
     its discovery data and was -15.8% out-of-sample — a pattern that only
     wins in the window that produced it is noise, not an edge.

Segments are enumerated from decision-log features (both sides of every market,
made AND rejected), so this searches far more than the picks we actually bet.

Usage: python3 -m tools.analysis.pattern_search [--sport MLB] [--min-n 40]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIG = 0.045


def _load(sport):
    rows = []
    for f in sorted(glob.glob(os.path.join(_ROOT, "state/decision_log/*.json"))):
        d = json.load(open(f))
        e = d.get("entries", d)
        rows.extend(e if isinstance(e, list) else list(e.values()))
    out = []
    for r in rows:
        if r.get("sport") != sport:
            continue
        if r.get("outcome") not in ("win", "loss"):
            continue
        p = r.get("market_prob_at_first_pick")
        if not p or not (0.05 < float(p) < 0.95):
            continue
        out.append(r)
    out.sort(key=lambda r: r["date"])
    return out


def _roi(rows):
    """Flat-stake ROI at realistic friction. Returns (n, win_rate, roi_pct)."""
    if not rows:
        return 0, 0.0, 0.0
    u = 0.0
    w = 0
    for r in rows:
        p = float(r["market_prob_at_first_pick"])
        if r["outcome"] == "win":
            w += 1
            u += (1.0 / p) * (1 - VIG) - 1
        else:
            u -= 1
    n = len(rows)
    return n, w / n * 100, u / n * 100


def _segments(rows):
    """Yield (label, predicate-filtered rows) for every segment worth testing."""
    F = lambda r, k: (r.get("features") or {}).get(k)
    segs = {}

    def add(label, sel):
        sub = [r for r in rows if sel(r)]
        if len(sub) >= 25:
            segs[label] = sub

    # market / side / price buckets
    for mkt in ("Moneyline", "Spread", "Total"):
        add(f"{mkt}", lambda r, m=mkt: r.get("market_type") == m)
        for lo, hi in ((0.05, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.95)):
            add(f"{mkt} price {lo:.2f}-{hi:.2f}",
                lambda r, m=mkt, a=lo, b=hi: r.get("market_type") == m
                and a <= float(r["market_prob_at_first_pick"]) < b)
        # home vs away side
        add(f"{mkt} home side",
            lambda r, m=mkt: r.get("market_type") == m and r.get("side") == r.get("home_team"))
        add(f"{mkt} away side",
            lambda r, m=mkt: r.get("market_type") == m and r.get("side") == r.get("away_team"))

    # totals over/under
    add("Total OVER", lambda r: r.get("market_type") == "Total" and str(r.get("side", "")).lower() == "over")
    add("Total UNDER", lambda r: r.get("market_type") == "Total" and str(r.get("side", "")).lower() == "under")

    # FADE the model: bet the side our model did NOT favor
    add("FADE model (bet the side we rejected, edge<0)",
        lambda r: (r.get("edge") is not None and r["edge"] < -0.03))
    add("FOLLOW model (edge>=5%)", lambda r: (r.get("edge") or 0) >= 0.05)

    # pitcher-driven segments (MLB)
    def sp_adv(r):
        h, a = F(r, "home_sp_score"), F(r, "away_sp_score")
        if h is None or a is None or not r.get("side"):
            return None
        return (h - a) if r.get("side") == r.get("home_team") else (a - h)

    add("side has better starter (gap>0.3)", lambda r: (sp_adv(r) or -9) > 0.3)
    add("side has WORSE starter (gap>0.3)", lambda r: (sp_adv(r) or 9) < -0.3)
    add("better starter AND favorite", lambda r: (sp_adv(r) or -9) > 0.3 and float(r["market_prob_at_first_pick"]) > 0.52)
    add("better starter AND underdog", lambda r: (sp_adv(r) or -9) > 0.3 and float(r["market_prob_at_first_pick"]) < 0.48)
    add("WORSE starter AND underdog", lambda r: (sp_adv(r) or 9) < -0.3 and float(r["market_prob_at_first_pick"]) < 0.48)

    # ERA trap on the opposing starter (fade an overpriced arm)
    def opp_trap(r):
        return F(r, "away_trap_sev") if r.get("side") == r.get("home_team") else F(r, "home_trap_sev")
    add("opposing starter ERA-trap >0.4", lambda r: (opp_trap(r) or 0) > 0.4)

    # park / injuries
    add("high park factor >1.05", lambda r: (F(r, "park_factor") or 1) > 1.05)
    add("low park factor <0.97", lambda r: (F(r, "park_factor") or 1) < 0.97)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="MLB")
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--train-frac", type=float, default=0.6)
    a = ap.parse_args()

    rows = _load(a.sport)
    if not rows:
        print(f"no graded {a.sport} rows")
        return
    cut_i = int(len(rows) * a.train_frac)
    split_date = rows[cut_i]["date"]
    early = [r for r in rows if r["date"] < split_date]
    late = [r for r in rows if r["date"] >= split_date]

    print(f"{a.sport}: {len(rows)} graded candidate rows "
          f"({rows[0]['date']} → {rows[-1]['date']})")
    print(f"DISCOVERY window: {len(early)} rows (< {split_date}) | "
          f"VALIDATION window: {len(late)} rows (>= {split_date})")
    print(f"Flat-stake ROI at {VIG:.1%} friction. A pattern must win in BOTH windows to mean anything.\n")

    segs_e = _segments(early)
    segs_l = _segments(late)

    results = []
    for label, sub_e in segs_e.items():
        n_e, wr_e, roi_e = _roi(sub_e)
        if n_e < a.min_n:
            continue
        sub_l = segs_l.get(label, [])
        n_l, wr_l, roi_l = _roi(sub_l)
        results.append((roi_e, label, n_e, wr_e, roi_e, n_l, wr_l, roi_l))

    results.sort(reverse=True)
    print(f"{'segment':44} {'DISCOVERY':>22}   {'VALIDATION (held out)':>24}")
    print(f"{'':44} {'n':>5} {'wr':>6} {'ROI':>8}   {'n':>5} {'wr':>6} {'ROI':>8}   verdict")
    for _, label, n_e, wr_e, roi_e, n_l, wr_l, roi_l in results:
        if n_l < 15:
            verdict = "too few to validate"
        elif roi_e > 0 and roi_l > 0:
            verdict = "★ HELD UP"
        elif roi_e > 0:
            verdict = "collapsed out-of-sample"
        else:
            verdict = ""
        print(f"{label:44} {n_e:5} {wr_e:5.1f}% {roi_e:+7.1f}%   {n_l:5} {wr_l:5.1f}% {roi_l:+7.1f}%   {verdict}")


if __name__ == "__main__":
    main()
