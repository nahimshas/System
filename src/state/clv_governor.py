"""
CLV governor — automatic budget-pool gating driven by closing-line value.

CLV (market_prob_at_close − market_prob_at_first_pick, stamped by
src/data/closing_lines.py) is the fastest reliable skill signal available:
a market whose picks consistently close WORSE than we bet them is one where
the market disagrees with the model after seeing more information — and the
closing line is almost always right. Win/loss needs 500+ picks to say this;
CLV says it in ~50.

Phase-gated like the calibration engine — deployed dormant, wakes up only
when a market has earned a sample:

  Phase 0  (n < 30)        observe only — gates nothing
  Phase 1  (30 ≤ n < 50)   extreme gate — block budget entry only when
                           avg CLV ≤ −2% (clearly chasing the market)
  Phase 2  (n ≥ 50)        active gate — block budget entry when
                           avg CLV ≤ −1%

Scope and safety:
  • The governor only gates entry into the BUDGET pool (real money). Display
    pools, watchlist tiles, and the shadow log are untouched — gated markets
    keep logging picks, so a market that improves un-gates itself.
  • Decisions are per (sport, market_type), recomputed from the shadow log on
    every run — no ratchet, no persistence of the decision itself.
  • Every call site is exception-safe: any failure means "allow", so a broken
    state file can never block the report or silently empty the card.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CLV_STATE_PATH = Path("state/clv_state.json")

# Phase thresholds (sample sizes) and gate levels (average CLV in prob points)
PHASE1_MIN_N = 30
PHASE2_MIN_N = 50
PHASE1_GATE  = -0.02   # extreme gate during the small-sample phase
PHASE2_GATE  = -0.01   # standard gate once the sample is trustworthy

# Process-level cache — stats are recomputed once per run, not once per pick
_stats_cache: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None


def _market_type_for_rec(rec: Any) -> str:
    bt = getattr(rec, "bet_type", "") or ""
    return bt if bt in ("Moneyline", "Spread", "Total", "Draw") else (bt or "Unknown")


# Prefer Kalshi's closing line — it is the book we actually trade in, it is
# free, and it is the ONLY source for F5 (the Odds API historical endpoint
# serves no *_1st_5_innings markets at any price).
#
# NOTHING IS ERASED. Kalshi retains settled markets only back to 2026-06-18
# while the shadow log starts in April, and that retention window moves
# FORWARD, so earlier history is permanently unreachable. Rather than truncate
# the governor's window, entries fall back to the sportsbook `clv` they already
# carry. Every historical value stays exactly where it was.
#
# ⚠️ This makes the series MIXED-VENUE at the 2026-06-18 boundary: Kalshi after,
# sportsbook before. That is a deliberate trade — measured Aug 25, the two are
# statistically indistinguishable as predictors of our own results (Odds API
# r=+0.0513 vs Kalshi r=+0.0347 at n=1597; 1 s.e. ~0.025, and the ranking flips
# with sample size), so the seam costs less than throwing away four months of
# history would. Set CLV_PREFER_KALSHI=False to revert to sportsbook-only.
CLV_PREFER_KALSHI = True

# How recent the secondary CLV series must be to keep a vote on gating.
ALT_FRESH_DAYS = 45


def effective_clv(entry: Dict[str, Any]) -> Optional[float]:
    """The CLV value the governor should gate on for one shadow-log entry."""
    if CLV_PREFER_KALSHI:
        k = entry.get("kalshi_clv")
        if k is not None:
            return float(k)
    v = entry.get("clv")
    return float(v) if v is not None else None


def clv_source(entry: Dict[str, Any]) -> str:
    """Which feed supplied this entry's CLV — for panels and audits."""
    if CLV_PREFER_KALSHI and entry.get("kalshi_clv") is not None:
        return "kalshi"
    return "odds_api" if entry.get("clv") is not None else "none"


def compute_clv_stats(force: bool = False) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate CLV per (sport, market_type) from the shadow log.

    Returns {(SPORT, market_type): {"n": int, "avg_clv": float}}.
    Cached per process. Never raises — returns {} on failure.
    """
    global _stats_cache
    if _stats_cache is not None and not force:
        return _stats_cache
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        from src.state.shadow_log import SHADOW_LOG_DIR, _load_shard
        sums: Dict[Tuple[str, str], list] = {}
        alt_sums: Dict[Tuple[str, str], list] = {}
        alt_latest: Dict[Tuple[str, str], str] = {}
        if SHADOW_LOG_DIR.exists():
            for shard_path in sorted(SHADOW_LOG_DIR.glob("*.json")):
                shard = _load_shard(shard_path)
                for entry in shard.get("entries", {}).values():
                    clv = effective_clv(entry)
                    if clv is None:
                        continue
                    key = (
                        (entry.get("sport") or "").upper(),
                        entry.get("market_type") or "Unknown",
                    )
                    sums.setdefault(key, []).append(float(clv))
                    # The OTHER source, tracked in parallel purely for gating.
                    alt = (entry.get("clv") if clv_source(entry) == "kalshi"
                           else entry.get("kalshi_clv"))
                    if alt is not None:
                        alt_sums.setdefault(key, []).append(float(alt))
                        d = entry.get("date") or ""
                        if d > alt_latest.get(key, ""):
                            alt_latest[key] = d
        for key, vals in sums.items():
            stats[key] = {"n": len(vals), "avg_clv": sum(vals) / len(vals)}
        # Per-source averages, so a gate can never be LOOSENED just because the
        # feed changed. See _phase_and_gate_multi below.
        for key, vals in alt_sums.items():
            if key in stats:
                stats[key]["alt_n"] = len(vals)
                stats[key]["alt_avg_clv"] = sum(vals) / len(vals)
                stats[key]["alt_latest"] = alt_latest.get(key, "")
    except Exception as e:
        logger.warning(f"CLV stats computation failed (non-fatal): {e}")
    _stats_cache = stats
    return stats


def _phase_and_gate_multi(st: Dict[str, Any]) -> Tuple[int, bool]:
    """Phase from the primary series; GATE IF EITHER SOURCE WOULD GATE.

    A safety rail must never come off just because the measurement feed
    changed. Aug 25 2026: preferring Kalshi would have UN-GATED MLB Total —
    gated since Jun 11 on -2.03% over 593 picks, but Kalshi reads the same
    market at -0.88%. The two sources are statistically indistinguishable as
    predictors overall, so a 1.15pp disagreement on one market is not evidence
    the gate was wrong; it is evidence we do not know. When we do not know, the
    protective setting stays on.

    This is deliberately ASYMMETRIC: either source can turn a gate ON, and both
    must agree to turn one OFF.
    """
    phase, gated = _phase_and_gate(st["n"], st["avg_clv"])
    alt_n, alt_avg = st.get("alt_n"), st.get("alt_avg_clv")
    if alt_n and alt_avg is not None and _alt_is_fresh(st.get("alt_latest", "")):
        _, alt_gated = _phase_and_gate(alt_n, alt_avg)
        gated = gated or alt_gated
    return phase, gated


def _alt_is_fresh(latest_date: str) -> bool:
    """Does the secondary series still have recent data?

    Once the Odds API CLV job is switched off, its series FREEZES. Without a
    sunset the asymmetric rule above would hold a market gated forever on
    evidence that stopped updating — which breaks the governor's core promise
    that gated markets RECOVER automatically as fresh data arrives (display and
    watchlist keep logging even while a market is gated out of the budget).
    So a stale secondary loses its vote after ALT_FRESH_DAYS and the primary
    series governs alone.
    """
    if not latest_date:
        return False
    try:
        from datetime import date as _d, timedelta as _td
        return _d.fromisoformat(latest_date) >= _d.today() - _td(days=ALT_FRESH_DAYS)
    except Exception:
        return False


def _phase_and_gate(n: int, avg_clv: float) -> Tuple[int, bool]:
    """Return (phase, gated) for a market's sample size and average CLV."""
    if n < PHASE1_MIN_N:
        return 0, False
    if n < PHASE2_MIN_N:
        return 1, avg_clv <= PHASE1_GATE
    return 2, avg_clv <= PHASE2_GATE


def clv_gate(rec: Any) -> Tuple[bool, str]:
    """Decide whether a pick may enter the budget pool.

    Returns (allowed, reason). Default is always allow — only a market with
    an earned negative-CLV track record is blocked. Never raises.
    """
    try:
        key = ((getattr(rec, "sport", "") or "").upper(), _market_type_for_rec(rec))
        st = compute_clv_stats().get(key)
        if not st:
            return True, ""
        phase, gated = _phase_and_gate_multi(st)
        if gated:
            return False, (
                f"CLV governor: {key[0]} {key[1]} avg CLV "
                f"{st['avg_clv'] * 100:+.1f}% over {st['n']} picks (phase {phase})"
            )
        return True, ""
    except Exception as e:
        logger.warning(f"CLV gate failed open (non-fatal): {e}")
        return True, ""


def persist_state() -> None:
    """Write the per-market CLV snapshot to state/clv_state.json for the
    report panel. Never raises."""
    try:
        stats = compute_clv_stats()
        rows = []
        for (sport, market_type), st in sorted(stats.items()):
            phase, gated = _phase_and_gate_multi(st)
            rows.append({
                "sport":       sport,
                "market_type": market_type,
                "n":           st["n"],
                "avg_clv":     round(st["avg_clv"], 4),
                "phase":       phase,
                "gated":       gated,
            })
        CLV_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CLV_STATE_PATH, "w") as f:
            json.dump({
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "thresholds": {
                    "phase1_min_n": PHASE1_MIN_N, "phase2_min_n": PHASE2_MIN_N,
                    "phase1_gate":  PHASE1_GATE,  "phase2_gate":  PHASE2_GATE,
                },
                "markets": rows,
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"CLV state persist failed (non-fatal): {e}")
