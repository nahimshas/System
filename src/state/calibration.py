"""
Calibration engine — set-and-forget probability calibration.

Reads settled outcomes from the shadow log and computes per-sport, per-market
adjustment ratios that the slot-ranking logic applies to live picks. Auto-
promotes between phases as data accumulates:

  Phase 0 (n < 100)          no adjustment, raw edges used unchanged
  Phase A (100 ≤ n < 400)    single ratio per sport × market
  Phase B (n ≥ 400, ≥30/bucket)  per-bucket ratios (50-60%, 60-70%, 70-80%, 80%+)
  Phase C (n ≥ 800, future)  isotonic regression (smooth curve)

The engine is exception-safe — any failure falls back to identity adjustment
(ratio = 1.0) so the report continues to work even with corrupted calibration
data.

Design contract:
  - One public function `effective_edge(rec) → float` used at the ranking
    chokepoint in main.py. Falls back to raw edge if anything goes wrong.
  - Calibration state cached per-process run (recomputed on each report
    generation, not per-pick).
  - State also serialised to state/calibration_state.json for the panel layer.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Source: shadow log shards
SHADOW_LOG_DIR = Path("state/shadow_log")
# Persisted state for the panel and external inspection
CALIBRATION_STATE_PATH = Path("state/calibration_state.json")

# Phase thresholds (sample-size gates)
PHASE_A_MIN = 100   # below this, no adjustment
PHASE_B_MIN = 400   # below this, single ratio per sport × market
PHASE_C_MIN = 800   # below this, bucket-based; above, isotonic (future)

# Bucket edges for Phase B (model_prob ranges)
BUCKET_EDGES = [0.50, 0.60, 0.70, 0.80, 1.01]   # 4 buckets: 50-60, 60-70, 70-80, 80+
BUCKET_MIN_N = 30                                # per-bucket minimum to use bucket ratio

# Auto-relaxation bounds for the cap counterfactual layer
CAP_MIN_TRIGGERS = 50            # need 50+ cap firings to consider adjustment
CAP_MAX_DRIFT = 0.30             # never widen cap beyond ±30% from market
CAP_MAX_MONTHLY_DELTA = 0.025    # max ±2.5% adjustment per month


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BucketStat:
    """Realised vs predicted within one model_prob bucket."""
    lo: float
    hi: float
    n: int = 0
    n_settled: int = 0
    sum_predicted: float = 0.0
    sum_realized: float = 0.0   # 1.0 per win, 0.5 per push, 0.0 per loss

    @property
    def predicted_rate(self) -> float:
        return self.sum_predicted / self.n_settled if self.n_settled else 0.0

    @property
    def realized_rate(self) -> float:
        return self.sum_realized / self.n_settled if self.n_settled else 0.0

    @property
    def ratio(self) -> float:
        """realized / predicted — 1.0 = perfectly calibrated."""
        p = self.predicted_rate
        return (self.realized_rate / p) if p > 0 else 1.0


@dataclass
class MarketCalibration:
    """Calibration state for one (sport, market_type) pair."""
    sport: str
    market_type: str
    phase: str = "0"                    # "0" / "A" / "B" / "C"
    n_total: int = 0
    n_settled: int = 0

    # Phase A — single ratio
    single_ratio: float = 1.0
    single_predicted_rate: float = 0.0
    single_realized_rate: float = 0.0

    # Phase B — bucket ratios
    buckets: List[BucketStat] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading shadow log shards
# ---------------------------------------------------------------------------

def _load_all_settled_entries() -> List[Dict[str, Any]]:
    """Read every shadow log shard and return all SETTLED entries (outcome != null)."""
    if not SHADOW_LOG_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(SHADOW_LOG_DIR.glob("*.json")):
        try:
            with open(path) as f:
                shard = json.load(f)
        except Exception as e:
            logger.warning(f"Calibration: failed to read {path}: {e}")
            continue
        for entry in shard.get("entries", {}).values():
            if entry.get("outcome") is not None:
                out.append(entry)
    return out


def _outcome_value(outcome: str) -> Optional[float]:
    """Convert canonical outcome label to a realised value for calibration math."""
    o = (outcome or "").lower()
    if o == "win":
        return 1.0
    if o == "loss":
        return 0.0
    if o == "push":
        return 0.5            # neutral
    # cancel / unknown — exclude from calibration entirely
    return None


def _bucket_for(model_prob: float) -> Optional[int]:
    """Return the index of the bucket containing model_prob, or None if out of range."""
    if model_prob is None or model_prob < BUCKET_EDGES[0] or model_prob >= BUCKET_EDGES[-1]:
        return None
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= model_prob < BUCKET_EDGES[i + 1]:
            return i
    return None


# ---------------------------------------------------------------------------
# Compute per-(sport, market_type) calibration
# ---------------------------------------------------------------------------

def _empty_buckets() -> List[BucketStat]:
    return [
        BucketStat(lo=BUCKET_EDGES[i], hi=BUCKET_EDGES[i + 1])
        for i in range(len(BUCKET_EDGES) - 1)
    ]


def compute_calibration_state() -> Dict[Tuple[str, str], MarketCalibration]:
    """
    Read all shadow log shards, compute per-(sport, market_type) calibration.

    Returns a dict mapping (sport, market_type) → MarketCalibration.
    """
    out: Dict[Tuple[str, str], MarketCalibration] = {}

    try:
        entries = _load_all_settled_entries()
    except Exception as e:
        logger.error(f"Calibration: failed to load shadow log entries: {e}")
        return out

    # Group entries by (sport, market_type) and aggregate
    for entry in entries:
        sport       = entry.get("sport", "")
        market_type = entry.get("market_type", "")
        if not sport or not market_type:
            continue

        outcome_val = _outcome_value(entry.get("outcome", ""))
        if outcome_val is None:
            continue   # cancelled / unknown — exclude

        model_prob = entry.get("model_prob")
        if not isinstance(model_prob, (int, float)):
            continue

        key = (sport, market_type)
        mc = out.get(key)
        if mc is None:
            mc = MarketCalibration(sport=sport, market_type=market_type, buckets=_empty_buckets())
            out[key] = mc

        mc.n_settled += 1
        mc.single_predicted_rate += model_prob
        mc.single_realized_rate  += outcome_val

        bi = _bucket_for(model_prob)
        if bi is not None:
            b = mc.buckets[bi]
            b.n += 1
            b.n_settled += 1
            b.sum_predicted += model_prob
            b.sum_realized  += outcome_val

    # Finalise: compute rates, ratios, and determine phase
    for mc in out.values():
        if mc.n_settled > 0:
            mean_pred = mc.single_predicted_rate / mc.n_settled
            mean_real = mc.single_realized_rate  / mc.n_settled
            mc.single_predicted_rate = mean_pred
            mc.single_realized_rate  = mean_real
            mc.single_ratio = (mean_real / mean_pred) if mean_pred > 0 else 1.0

        # Phase determination
        mc.phase = _determine_phase(mc)

    # Also include n_total from full shadow log scan (settled + unsettled)
    try:
        _populate_n_totals(out)
    except Exception as e:
        logger.debug(f"Calibration: n_total scan failed (non-fatal): {e}")

    return out


def _populate_n_totals(state: Dict[Tuple[str, str], MarketCalibration]) -> None:
    """Add unsettled entry counts to MarketCalibration.n_total."""
    if not SHADOW_LOG_DIR.exists():
        return
    counts: Dict[Tuple[str, str], int] = {}
    for path in sorted(SHADOW_LOG_DIR.glob("*.json")):
        try:
            with open(path) as f:
                shard = json.load(f)
        except Exception:
            continue
        for entry in shard.get("entries", {}).values():
            key = (entry.get("sport", ""), entry.get("market_type", ""))
            if key in state or key[0]:   # count all keys, not just ones with settled data
                counts[key] = counts.get(key, 0) + 1
    for key, n in counts.items():
        mc = state.get(key)
        if mc:
            mc.n_total = n
        else:
            state[key] = MarketCalibration(
                sport=key[0], market_type=key[1],
                n_total=n, n_settled=0,
                buckets=_empty_buckets(),
            )


def _determine_phase(mc: MarketCalibration) -> str:
    """Decide which calibration phase this (sport, market_type) qualifies for."""
    if mc.n_settled < PHASE_A_MIN:
        return "0"
    if mc.n_settled < PHASE_B_MIN:
        return "A"
    # Bucket phase requires minimum samples per bucket
    if all(b.n_settled >= BUCKET_MIN_N for b in mc.buckets):
        if mc.n_settled >= PHASE_C_MIN:
            return "C"
        return "B"
    # Have ≥ PHASE_B_MIN total but some buckets are sparse — keep Phase A
    return "A"


# ---------------------------------------------------------------------------
# Adjustment lookup — applied at the ranking chokepoint
# ---------------------------------------------------------------------------

# Process-cached state — recomputed once per main.py run
_STATE_CACHE: Optional[Dict[Tuple[str, str], MarketCalibration]] = None


def get_state(force_refresh: bool = False) -> Dict[Tuple[str, str], MarketCalibration]:
    """Get the calibration state, computing it once per run."""
    global _STATE_CACHE
    if _STATE_CACHE is None or force_refresh:
        try:
            _STATE_CACHE = compute_calibration_state()
        except Exception as e:
            logger.error(f"Calibration state compute failed (non-fatal): {e}")
            _STATE_CACHE = {}
    return _STATE_CACHE


def adjustment_for(sport: str, market_type: str, model_prob: float) -> float:
    """
    Return the multiplicative edge adjustment for a (sport, market, model_prob).

    1.0  = no adjustment (Phase 0, insufficient data)
    < 1  = model historically overconfident, shrink edge
    > 1  = model historically underconfident, amplify edge

    Falls back to 1.0 on any error — calibration never breaks live picks.
    """
    try:
        state = get_state()
        mc = state.get((sport, market_type))
        if mc is None:
            return 1.0

        if mc.phase == "0":
            return 1.0
        if mc.phase == "A":
            return mc.single_ratio

        if mc.phase in ("B", "C"):
            bi = _bucket_for(model_prob)
            if bi is None:
                # Outside our bucket range — fall back to single ratio
                return mc.single_ratio
            b = mc.buckets[bi]
            if b.n_settled < BUCKET_MIN_N:
                return mc.single_ratio
            return b.ratio

        return 1.0
    except Exception as e:
        logger.debug(f"Calibration adjustment lookup failed (non-fatal): {e}")
        return 1.0


def effective_edge(rec: Any) -> float:
    """
    Compute the calibration-adjusted edge for a BetRecommendation.

    Phase 0:  returns raw edge unchanged
    Phase A+: returns raw_edge × calibration_ratio (single or bucket)

    This is the chokepoint used by the slot-ranking logic. The raw edge
    remains available on the recommendation for display purposes — only
    the ranking decision uses the effective edge.
    """
    try:
        raw_edge   = float(getattr(rec, "edge", 0.0))
        sport      = getattr(rec, "sport", "")
        bet_type   = getattr(rec, "bet_type", "")
        model_prob = float(getattr(rec, "model_prob", 0.0))
        # market_type uses the same labels as bet_type (Moneyline/Spread/Total/Draw)
        return raw_edge * adjustment_for(sport, bet_type, model_prob)
    except Exception:
        return float(getattr(rec, "edge", 0.0) or 0.0)


# ---------------------------------------------------------------------------
# State serialisation for the panel layer
# ---------------------------------------------------------------------------

def _save_state_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def persist_state() -> None:
    """
    Serialise the current calibration state to state/calibration_state.json
    so the report panel and external inspection tools can read it.

    Non-fatal — failure is logged but does not block anything.
    """
    try:
        state = get_state()
        payload = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "phase_thresholds": {
                "phase_a_min": PHASE_A_MIN,
                "phase_b_min": PHASE_B_MIN,
                "phase_c_min": PHASE_C_MIN,
                "bucket_min_n": BUCKET_MIN_N,
                "bucket_edges": BUCKET_EDGES,
            },
            "markets": [],
        }
        for (sport, market_type), mc in sorted(state.items()):
            payload["markets"].append({
                "sport":          sport,
                "market_type":    market_type,
                "phase":          mc.phase,
                "n_total":        mc.n_total,
                "n_settled":      mc.n_settled,
                "single_ratio":   round(mc.single_ratio, 4),
                "single_predicted_rate": round(mc.single_predicted_rate, 4),
                "single_realized_rate":  round(mc.single_realized_rate, 4),
                "buckets": [
                    {
                        "lo":             b.lo,
                        "hi":             b.hi,
                        "n_settled":      b.n_settled,
                        "predicted_rate": round(b.predicted_rate, 4),
                        "realized_rate":  round(b.realized_rate, 4),
                        "ratio":          round(b.ratio, 4),
                    }
                    for b in mc.buckets
                ],
            })
        _save_state_atomic(CALIBRATION_STATE_PATH, payload)
        logger.info(f"Calibration state persisted → {CALIBRATION_STATE_PATH} ({len(state)} markets)")
    except Exception as e:
        logger.error(f"Calibration persist failed (non-fatal): {e}")
