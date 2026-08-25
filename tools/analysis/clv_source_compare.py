"""
Is Kalshi CLV a safe replacement for Odds API CLV?

Run before switching the nightly CLV job off the Odds API:

    python3 -m tools.analysis.clv_source_compare

The two are measured differently ON PURPOSE — each is self-consistent within
its own venue (sportsbook no-vig open vs close; Kalshi mid at pick vs mid at
close). Mixing ends across venues would fold the venue gap into the signal, so
they are never combined; they are only COMPARED here.

⚠️ AGREEMENT IS THE WRONG TEST. The first version of this tool gated the
switch on correlation BETWEEN the two sources, which asks "does Kalshi
reproduce the sportsbook number?" That is not what CLV is for. CLV earns its
keep by PREDICTING RESULTS, so the question is which source better separates
our winners from our losers — measured on the same rows. A source can
correlate poorly with the other and still be the better signal, and Kalshi has
the stronger claim on principle: it is the book we actually trade in.

What to look for before switching:
  • predictive power (section 3): Kalshi >= Odds API on the SAME rows
  • coverage: Kalshi matches or beats Odds API row counts
  • history depth: Kalshi retains only ~2 months of settled markets, so
    switching TRUNCATES the governor's window to what Kalshi can reach

Switching sources mid-history creates a seam: everything before is
sportsbook-based, everything after is Kalshi-based. Backfill as far as Kalshi
retains first, and treat the earliest Kalshi date as a hard boundary.
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
        print("  -> context only. Low agreement does NOT block a switch; the two")
        print("     measure different venues. Section 3 is the deciding test.\n")

    # ── 3. predictive power: which CLV actually forecasts winning? ────────
    settled = [r for r in rows if r.get("outcome") in ("win", "loss")
               and r.get("clv") is not None and r.get("kalshi_clv") is not None]
    print("=== PREDICTIVE POWER (the test that matters) ===")
    if len(settled) < 30:
        print(f"  only {len(settled)} settled paired rows — need ~30+\n")
    else:
        def _pb(key):
            xs = [r[key] for r in settled]
            ys = [1.0 if r["outcome"] == "win" else 0.0 for r in settled]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            sx = sum((a - mx) ** 2 for a in xs) ** 0.5
            sy = sum((b - my) ** 2 for b in ys) ** 0.5
            return cov / (sx * sy) if sx and sy else None

        def _split(key):
            pos = [r for r in settled if r[key] > 0]
            neg = [r for r in settled if r[key] <= 0]
            def wr(g):
                if not g:
                    return "n/a", 0.0
                w = sum(1 for r in g if r["outcome"] == "win")
                return f"{w}-{len(g)-w} ({w/len(g)*100:.1f}%)", w / len(g)
            return wr(pos), wr(neg)

        se = 1.0 / (len(settled) ** 0.5)
        print(f"  n = {len(settled)} settled rows with BOTH CLVs "
              f"(1 s.e. on a correlation is ~{se:.3f})")
        best, best_r = None, None
        for key, label in (("clv", "Odds API"), ("kalshi_clv", "Kalshi  ")):
            r = _pb(key)
            (pw, pr), (nw, nr) = _split(key)
            print(f"  {label}: r={r:+.4f}   CLV>0 {pw:<16} CLV<=0 {nw:<16} "
                  f"gap {(pr-nr)*100:+.1f}pp")
            if r is not None and (best_r is None or r > best_r):
                best, best_r = label.strip(), r
        print(f"  -> stronger predictor: {best}")
        print("  NOTE: both are near zero. Read a difference smaller than one")
        print("  s.e. as a tie, not as a winner.\n")

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
