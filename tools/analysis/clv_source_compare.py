"""
Is Kalshi CLV a safe replacement for Odds API CLV?

Run before switching the nightly CLV job off the Odds API:

    python3 -m tools.analysis.clv_source_compare

The two are measured differently ON PURPOSE — each is self-consistent within
its own venue (sportsbook no-vig open vs close; Kalshi mid at pick vs mid at
close). Mixing ends across venues would fold the venue gap into the signal, so
they are never combined; they are only COMPARED here.

What to look for before switching:
  • correlation >= 0.7 on rows where both exist — they track the same thing
  • mean difference small relative to the CLV values themselves
  • coverage: Kalshi should match at least as many rows as the Odds API,
    and strictly more for F5 (which the Odds API cannot serve at all)

A high correlation does NOT mean the numbers are interchangeable in an
existing series. Switching sources mid-history creates a seam: everything
before is sportsbook-based, everything after is Kalshi-based. Either backfill
Kalshi across the whole window first, or treat the switch date as a boundary.
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict


def _rows(since: str):
    out = []
    for f in sorted(glob.glob("state/shadow_log/*.json")):
        try:
            for e in json.load(open(f)).get("entries", {}).values():
                if (e.get("date") or "") >= since:
                    out.append(e)
        except Exception:
            continue
    return out


def _corr(pairs):
    if len(pairs) < 3:
        return None
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in pairs)
    sx = sum((a - mx) ** 2 for a in xs) ** 0.5
    sy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (sx * sy) if sx and sy else None


def main(since: str = "2026-08-04") -> int:
    rows = _rows(since)
    if not rows:
        print(f"No shadow rows since {since}.")
        return 0

    sb = [r for r in rows if r.get("clv") is not None]
    kx = [r for r in rows if r.get("kalshi_clv") is not None]
    both = [(r["clv"], r["kalshi_clv"]) for r in rows
            if r.get("clv") is not None and r.get("kalshi_clv") is not None]

    print(f"=== CLV SOURCE COMPARISON (since {since}) ===\n")
    print(f"shadow rows              : {len(rows)}")
    print(f"  with Odds API CLV      : {len(sb)}  ({len(sb)/len(rows)*100:.0f}%)")
    print(f"  with Kalshi CLV        : {len(kx)}  ({len(kx)/len(rows)*100:.0f}%)")
    print(f"  with BOTH (comparable) : {len(both)}\n")

    if both:
        a = [x for x, _ in both]
        b = [y for _, y in both]
        c = _corr(both)
        print(f"mean Odds API CLV : {statistics.mean(a)*100:+6.2f}pp")
        print(f"mean Kalshi   CLV : {statistics.mean(b)*100:+6.2f}pp")
        print(f"mean |difference| : {statistics.mean([abs(x-y) for x,y in both])*100:6.2f}pp")
        print(f"correlation       : {c:.3f}" if c is not None else "correlation: n/a")
        print(f"  -> {'TRACKS THE SAME SIGNAL' if (c or 0) >= 0.7 else 'DIVERGENT — do not switch'}\n")

    print("=== COVERAGE BY MARKET ===")
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = f"{r.get('sport','?')} {r.get('market_type','?')}"
        agg[k][0] += 1
        agg[k][1] += 1 if r.get("clv") is not None else 0
        agg[k][2] += 1 if r.get("kalshi_clv") is not None else 0
    print(f"{'market':<22}{'rows':>6}{'oddsAPI':>9}{'kalshi':>8}   note")
    for k, (n, s, x) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        note = ""
        if s == 0 and x > 0:
            note = "<- Kalshi-only (Odds API cannot serve this)"
        elif x < s:
            note = "<- Kalshi coverage BEHIND; backfill before switching"
        print(f"{k[:21]:<22}{n:>6}{s:>9}{x:>8}   {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "2026-08-04"))
