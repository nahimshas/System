"""Would our picks change if we priced off Kalshi instead of the Odds API?

WHY THIS IS A REPORT AND NOT A NEW LOGGER
The parallel-run data already exists. Every decision-log row carries BOTH
prices for the same moment:

    market_prob_at_first_pick   the Odds API no-vig sportsbook consensus
    kalshi_prob_at_pick         the Kalshi mid, stamped by the CLV pass

9,698 rows had both when this was written — more than any three-week trial
would produce — so building a second collection path would have added a failure
surface to answer a question the archive already answers, and would have thrown
away months of history by starting from zero.

⚠️ WHAT THIS CAN AND CANNOT TELL YOU
`made` depends on far more than the edge floor: budget routing, slot limits,
one-bet-per-game, CLV gates and Kelly sizing all intervene. So this does NOT
replay the full selection. It compares the one thing that is apples-to-apples —
whether a candidate CLEARS THE EDGE FLOOR under each price — computed
identically for both feeds. Read a flip as "this candidate would have changed
side of the threshold", not "the card would have changed by exactly this much".

Usage:
    python3 -m tools.analysis.price_compare
    python3 -m tools.analysis.price_compare --since 2026-09-01 --min-edge 0.05
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rows(since: str):
    out = []
    for path in sorted(glob.glob(os.path.join(_ROOT, "state/decision_log/*.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        rs = data.get("entries", data)
        rs = list(rs.values()) if isinstance(rs, dict) else rs
        for r in rs:
            if not isinstance(r, dict) or r.get("date", "") < since:
                continue
            o, k, m = (r.get("market_prob_at_first_pick"),
                       r.get("kalshi_prob_at_pick"), r.get("model_prob"))
            if o is None or k is None or m is None:
                continue
            out.append(r)
    return out


def analyse(since: str = "0000-00-00", min_edge: float = 0.05):
    rows = _rows(since)
    by = defaultdict(list)
    for r in rows:
        by[(r.get("sport"), r.get("market_type"))].append(r)

    report = {"n": len(rows), "since": since, "min_edge": min_edge, "markets": {}}
    for key, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        d = [r["kalshi_prob_at_pick"] - r["market_prob_at_first_pick"] for r in rs]
        flips_in = flips_out = 0
        for r in rs:
            eo = r["model_prob"] - r["market_prob_at_first_pick"]
            ek = r["model_prob"] - r["kalshi_prob_at_pick"]
            if eo >= min_edge and ek < min_edge:
                flips_out += 1          # we bet it today, Kalshi pricing would not
            elif eo < min_edge and ek >= min_edge:
                flips_in += 1           # Kalshi pricing would bet something we skip
        n = len(rs)
        report["markets"][f"{key[0]} {key[1]}"] = {
            "n": n,
            "mean_delta_pp": round(100 * sum(d) / n, 3),
            "mean_abs_delta_pp": round(100 * sum(abs(x) for x in d) / n, 3),
            "p95_abs_delta_pp": round(100 * sorted(abs(x) for x in d)[int(0.95 * (n - 1))], 3),
            "over_2pp_pct": round(100 * sum(1 for x in d if abs(x) > 0.02) / n, 1),
            "flips_out": flips_out,
            "flips_in": flips_in,
            "flip_pct": round(100 * (flips_in + flips_out) / n, 2),
        }
    if rows:
        allд = [r["kalshi_prob_at_pick"] - r["market_prob_at_first_pick"] for r in rows]
        fo = sum(1 for r in rows
                 if (r["model_prob"] - r["market_prob_at_first_pick"]) >= min_edge
                 and (r["model_prob"] - r["kalshi_prob_at_pick"]) < min_edge)
        fi = sum(1 for r in rows
                 if (r["model_prob"] - r["market_prob_at_first_pick"]) < min_edge
                 and (r["model_prob"] - r["kalshi_prob_at_pick"]) >= min_edge)
        report["overall"] = {
            "mean_delta_pp": round(100 * sum(allд) / len(allд), 3),
            "mean_abs_delta_pp": round(100 * sum(abs(x) for x in allд) / len(allд), 3),
            "sd_pp": round(100 * st.pstdev(allд), 3),
            "flips_out": fo, "flips_in": fi,
            "flip_pct": round(100 * (fo + fi) / len(rows), 2),
        }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="0000-00-00")
    ap.add_argument("--min-edge", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rep = analyse(a.since, a.min_edge)
    if a.json:
        print(json.dumps(rep, indent=2))
        return
    print(f"=== KALSHI vs ODDS API PRICING — {rep['n']} candidate rows "
          f"(since {rep['since']}, floor {rep['min_edge']:.0%}) ===\n")
    print(f"{'market':22s} {'n':>6s} {'mean Δ':>8s} {'mean|Δ|':>8s} "
          f"{'p95|Δ|':>7s} {'>2pp':>6s} {'flips':>7s}")
    for name, m in rep["markets"].items():
        if m["n"] < 20:
            continue
        print(f"{name:22s} {m['n']:6d} {m['mean_delta_pp']:+7.2f}pp "
              f"{m['mean_abs_delta_pp']:7.2f}pp {m['p95_abs_delta_pp']:6.2f}pp "
              f"{m['over_2pp_pct']:5.1f}% {m['flip_pct']:6.2f}%")
    o = rep.get("overall")
    if o:
        print(f"\nOVERALL  mean {o['mean_delta_pp']:+.3f}pp | "
              f"mean|Δ| {o['mean_abs_delta_pp']:.3f}pp | sd {o['sd_pp']:.2f}pp")
        print(f"         edge-floor flips: {o['flips_out']} out, {o['flips_in']} in "
              f"({o['flip_pct']:.2f}% of candidates)")
        print("\nNOTE: flips count candidates crossing the EDGE FLOOR, not full "
              "card replays —\n      budget routing, slot limits, one-per-game "
              "and CLV gates are not simulated.")


if __name__ == "__main__":
    main()
