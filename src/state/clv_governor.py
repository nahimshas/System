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
        if SHADOW_LOG_DIR.exists():
            for shard_path in sorted(SHADOW_LOG_DIR.glob("*.json")):
                shard = _load_shard(shard_path)
                for entry in shard.get("entries", {}).values():
                    clv = entry.get("clv")
                    if clv is None:
                        continue
                    key = (
                        (entry.get("sport") or "").upper(),
                        entry.get("market_type") or "Unknown",
                    )
                    sums.setdefault(key, []).append(float(clv))
        for key, vals in sums.items():
            stats[key] = {"n": len(vals), "avg_clv": sum(vals) / len(vals)}
    except Exception as e:
        logger.warning(f"CLV stats computation failed (non-fatal): {e}")
    _stats_cache = stats
    return stats


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
        phase, gated = _phase_and_gate(st["n"], st["avg_clv"])
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
            phase, gated = _phase_and_gate(st["n"], st["avg_clv"])
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
