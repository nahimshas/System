"""
Cap auto-relaxation — set-and-forget cap value tuning.

Reads cap-fired entries from the shadow log and runs a counterfactual analysis:
when the credibility cap fired, was the raw model prediction more accurate
than the capped one, or was the cap correctly suppressing overconfidence?

Adjusts each cap's value (within hard safety bounds) based on the evidence:
  • Capped prediction was meaningfully WORSE than raw → widen cap (+2.5%)
  • Capped prediction was meaningfully BETTER than raw → tighten cap (-2.5%)
  • No meaningful difference → keep current value

Throttled to once-per-cap-per-month so caps can't oscillate. Failures are
logged but never propagate.

State persisted to state/cap_state.json. The edge finder reads current values
via `get_current_cap()` with a constant fallback (so the system continues to
function identically if the state file is missing).
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CAP_STATE_PATH = Path("state/cap_state.json")
SCHEMA_VERSION = 1

# Hard safety bounds — caps can never exceed these regardless of counterfactual.
CAP_MIN = 0.05     # never tighter than ±5%
CAP_MAX = 0.30     # never wider than ±30%

# Counterfactual thresholds — VALUE adjustment (Step 7)
MIN_TRIGGERS_FOR_ADJUST = 50      # need 50+ cap firings to consider value adjustment
ADJUST_DELTA_PER_TICK   = 0.025   # max ±2.5% adjustment per evaluation
ADJUST_THROTTLE_DAYS    = 30      # don't adjust same cap more than once per month
MAE_GAP_THRESHOLD       = 0.03    # raw vs capped MAE gap to trigger adjustment

# MODE promotion thresholds (Step 6 — Option B upgrade paths).
# Mode switching is more disruptive than value tuning, so it requires both
# more data and stronger evidence before triggering.
#
# Modes:
#   0 = hard clip (default; sharp boundary at ±drift)
#   1 = tanh saturation (smooth pull, no full clipping)
#   2 = logistic blend (reserved; not yet implemented)
MIN_TRIGGERS_FOR_MODE_PROMOTION = 200    # 4× value-adjust threshold
MODE_PROMOTION_MAE_GAP          = 0.015  # alt mode must be ≥ 1.5% better
MODE_PROMOTION_THROTTLE_DAYS    = 60     # 2× value-adjust throttle

# Initial default cap values per sport. Must match the constants in
# src/models/edge_finder.py — these are the values the cap defaults to before
# any data-driven adjustment has occurred. Kept duplicated (rather than
# imported) to avoid circular dependency with edge_finder.
_INITIAL_DEFAULT_CAPS: Dict[str, float] = {
    # NBA
    "nba.credibility_moneyline": 0.15,
    "nba.credibility_spread":    0.15,
    "nba.credibility_total":     0.15,
    "nba.credibility_prop":      0.15,
    # MLB
    "mlb.credibility_moneyline": 0.15,
    "mlb.credibility_spread":    0.15,
    "mlb.credibility_total":     0.15,
    "mlb.credibility_prop":      0.15,
    # NFL
    "nfl.credibility_moneyline": 0.15,
    "nfl.credibility_spread":    0.15,
    "nfl.credibility_total":     0.15,
    # NHL
    "nhl.credibility_moneyline": 0.15,
    "nhl.credibility_spread":    0.15,
    "nhl.credibility_total":     0.15,
    # MLS
    "mls.credibility_moneyline": 0.10,
    "mls.credibility_draw":      0.10,
    "mls.credibility_total":     0.10,
    "mls.credibility_spread":    0.10,

    "ligamx.credibility_moneyline": 0.10,
    "ligamx.credibility_draw":      0.10,
    # WNBA / IPL (moneyline only)
    "wnba.credibility_moneyline": 0.15,
    "ipl.credibility_moneyline":  0.15,
}

_DISPLAY_LABELS: Dict[str, str] = {
    "nba.credibility_moneyline":  "NBA Moneylines",
    "nba.credibility_spread":     "NBA Spreads",
    "nba.credibility_total":      "NBA Over/Unders",
    "nba.credibility_prop":       "NBA Props",
    "mlb.credibility_moneyline":  "MLB Moneylines",
    "mlb.credibility_spread":     "MLB Spreads",
    "mlb.credibility_total":      "MLB Over/Unders",
    "mlb.credibility_prop":       "MLB Props",
    "nfl.credibility_moneyline":  "NFL Moneylines",
    "nfl.credibility_spread":     "NFL Spreads",
    "nfl.credibility_total":      "NFL Over/Unders",
    "nhl.credibility_moneyline":  "NHL Moneylines",
    "nhl.credibility_spread":     "NHL Spreads",
    "nhl.credibility_total":      "NHL Over/Unders",
    "mls.credibility_moneyline":  "MLS Moneylines",
    "mls.credibility_draw":       "MLS Draws",
    "mls.credibility_total":      "MLS Over/Unders",
    "mls.credibility_spread":     "MLS Spreads",
    "ligamx.credibility_moneyline": "Liga MX Moneylines",
    "ligamx.credibility_draw":      "Liga MX Draws",
    "wnba.credibility_moneyline": "WNBA Moneylines",
    "ipl.credibility_moneyline":  "IPL Moneylines",
}

_SPORT_ORDER: Dict[str, int] = {
    "nba": 0, "mlb": 1, "nfl": 2, "nhl": 3, "mls": 4, "wnba": 5, "ipl": 6
}
_BET_TYPE_ORDER: Dict[str, int] = {
    "credibility_moneyline": 0,
    "credibility_spread":    1,
    "credibility_total":     2,
    "credibility_prop":      3,
    "credibility_draw":      4,
}

# Maps cap key suffix → set of shadow log bet_type values covered by that cap.
# Used in evaluate_and_adjust_caps to filter entries to only those bet types.
_CAP_BET_TYPE_FILTER: Dict[str, set] = {
    "credibility_moneyline": {"Moneyline"},
    "credibility_spread":    {"Spread"},
    "credibility_total":     {"Total"},
    "credibility_prop":      {"Prop", ""},   # "" covers legacy shadow log entries before PropPick got bet_type
    "credibility_draw":      {"Draw"},
}

# Process cache — invalidated on save
_CAP_CACHE: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load_state() -> Dict[str, Any]:
    if not CAP_STATE_PATH.exists():
        return {"schema_version": SCHEMA_VERSION, "caps": {}, "computed_at": None}
    try:
        with open(CAP_STATE_PATH) as f:
            data = json.load(f)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("caps", {})
        return data
    except Exception as e:
        logger.warning(f"Cap state load failed (using defaults): {e}")
        return {"schema_version": SCHEMA_VERSION, "caps": {}, "computed_at": None}


def _save_state_atomic(data: Dict[str, Any]) -> None:
    CAP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=CAP_STATE_PATH.name + ".", suffix=".tmp", dir=str(CAP_STATE_PATH.parent)
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, CAP_STATE_PATH)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API — read current cap (called from edge_finder per-pick)
# ---------------------------------------------------------------------------

def get_current_cap(sport: str, cap_type: str = "credibility") -> float:
    """
    Return the current effective cap value for sport/cap_type.

    Falls back to the initial default if state file missing or cap not
    listed. Exception-safe — never raises.
    """
    global _CAP_CACHE
    key = f"{sport.lower()}.{cap_type}"
    try:
        if _CAP_CACHE is None:
            _CAP_CACHE = _load_state().get("caps", {})
        entry = _CAP_CACHE.get(key)
        if entry and "current_value" in entry:
            return float(entry["current_value"])
        return _INITIAL_DEFAULT_CAPS.get(key, 0.15)
    except Exception:
        return _INITIAL_DEFAULT_CAPS.get(key, 0.15)


def get_current_cap_mode(sport: str, cap_type: str = "credibility") -> int:
    """
    Return the current cap MODE for sport/cap_type:
      0 = hard clip (default)
      1 = tanh saturation
      2 = logistic blend (reserved, not yet implemented)

    Falls back to Mode 0 if state missing or cap not listed.
    Exception-safe — never raises (edge_finder dispatch depends on this).
    """
    global _CAP_CACHE
    key = f"{sport.lower()}.{cap_type}"
    try:
        if _CAP_CACHE is None:
            _CAP_CACHE = _load_state().get("caps", {})
        entry = _CAP_CACHE.get(key) or {}
        return int(entry.get("current_mode", 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Counterfactual analysis — was the cap helping or hurting?
# ---------------------------------------------------------------------------

def _tanh_saturation_predict(raw: float, market: float, drift: float) -> float:
    """Pure function — what Mode 1 (tanh saturation) would predict given raw/market/drift."""
    if drift <= 0:
        return raw
    delta  = raw - market
    smooth = drift * math.tanh(delta / drift)
    return max(0.0, min(1.0, market + smooth))


def _compute_mode_counterfactual(
    entries: List[Dict[str, Any]],
    current_drift: float,
) -> Dict[str, Any]:
    """
    For cap-fired settled entries, compute what Mode 1 (tanh saturation) WOULD
    have predicted using the same raw model probability and current drift —
    and compare its MAE against the actual Mode 0 (hard clip) prediction's MAE.

    Returns a dict including a `recommendation`:
      - "promote"      — tanh meaningfully better than hard clip → switch to Mode 1
      - "keep"         — no meaningful improvement → stay on current mode
      - "insufficient" — n < MIN_TRIGGERS_FOR_MODE_PROMOTION
    """
    fired = [
        e for e in entries
        if e.get("credibility_cap_fired") is True
        and e.get("outcome") in ("win", "loss")
        and e.get("model_prob_raw") is not None
        and e.get("model_prob") is not None
    ]
    n = len(fired)
    if n < MIN_TRIGGERS_FOR_MODE_PROMOTION:
        return {"recommendation": "insufficient", "trigger_count": n}

    market_inferred_count = 0
    hard_err = 0.0
    tanh_err = 0.0

    for e in fired:
        outcome = 1.0 if e["outcome"] == "win" else 0.0
        raw     = float(e["model_prob_raw"])
        capped  = float(e["model_prob"])
        # Infer market_prob from the cap-firing relationship:
        # Mode 0 clipped raw to capped at distance ±current_drift from market,
        # so market is approximately: market = capped - sign(raw-capped)*drift
        # (Within ±0.005 — the cap clipping is exact at ±drift.)
        if raw > capped:
            market_approx = capped - current_drift
        else:
            market_approx = capped + current_drift
        market_approx = max(0.0, min(1.0, market_approx))
        market_inferred_count += 1

        tanh_pred = _tanh_saturation_predict(raw, market_approx, current_drift)
        hard_err += abs(capped    - outcome)
        tanh_err += abs(tanh_pred - outcome)

    hard_mae = hard_err / n
    tanh_mae = tanh_err / n
    gap = hard_mae - tanh_mae    # positive = tanh is better → promote

    if gap > MODE_PROMOTION_MAE_GAP:
        recommendation = "promote"
    else:
        recommendation = "keep"

    return {
        "recommendation":   recommendation,
        "trigger_count":    n,
        "hard_clip_mae":    round(hard_mae, 4),
        "tanh_satur_mae":   round(tanh_mae, 4),
        "mae_gap":          round(gap, 4),
        "inferred_markets": market_inferred_count,
    }


def _compute_counterfactual(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For a list of shadow log entries where the credibility cap fired, compute
    the mean absolute error (MAE) of raw vs capped predictions against the
    actual outcomes. The MAE gap drives the recommendation.

    Returns a dict with the analysis and a recommendation:
      - "widen"        — capped meaningfully WORSE than raw, cap is suppressing real signal
      - "tighten"      — capped meaningfully BETTER than raw, cap is correctly suppressing overconfidence
      - "keep"         — no meaningful difference; current cap is well-calibrated
      - "insufficient" — n < MIN_TRIGGERS_FOR_ADJUST
    """
    fired = [
        e for e in entries
        if e.get("credibility_cap_fired") is True
        and e.get("outcome") in ("win", "loss")
        and e.get("model_prob_raw") is not None
        and e.get("model_prob") is not None
    ]
    n = len(fired)

    if n < MIN_TRIGGERS_FOR_ADJUST:
        return {"recommendation": "insufficient", "trigger_count": n}

    raw_err_sum    = 0.0
    capped_err_sum = 0.0
    raw_sum        = 0.0
    capped_sum     = 0.0
    realized_sum   = 0.0

    for e in fired:
        outcome  = 1.0 if e["outcome"] == "win" else 0.0
        raw      = float(e["model_prob_raw"])
        capped   = float(e["model_prob"])
        raw_err_sum    += abs(raw - outcome)
        capped_err_sum += abs(capped - outcome)
        raw_sum        += raw
        capped_sum     += capped
        realized_sum   += outcome

    raw_mae    = raw_err_sum    / n
    capped_mae = capped_err_sum / n
    mae_gap    = capped_mae - raw_mae    # positive = capped is worse (widen)

    if mae_gap > MAE_GAP_THRESHOLD:
        recommendation = "widen"
    elif mae_gap < -MAE_GAP_THRESHOLD:
        recommendation = "tighten"
    else:
        recommendation = "keep"

    return {
        "recommendation":         recommendation,
        "trigger_count":          n,
        "raw_mae":                round(raw_mae, 4),
        "capped_mae":             round(capped_mae, 4),
        "mae_gap":                round(mae_gap, 4),
        "raw_predicted_rate":     round(raw_sum / n, 4),
        "capped_predicted_rate":  round(capped_sum / n, 4),
        "realized_rate":          round(realized_sum / n, 4),
    }


# ---------------------------------------------------------------------------
# Public API — run once per day to evaluate and adjust caps
# ---------------------------------------------------------------------------

def evaluate_and_adjust_caps() -> int:
    """
    Read every settled shadow log entry, compute the counterfactual per cap,
    and adjust each cap's value if the evidence warrants (and the throttle
    permits).

    Adjustments are bounded by:
      • Hard floor:   CAP_MIN  (0.05) — can never tighten below
      • Hard ceiling: CAP_MAX  (0.30) — can never widen above
      • Tick size:    ADJUST_DELTA_PER_TICK (0.025) per evaluation
      • Throttle:     ADJUST_THROTTLE_DAYS (30) per cap

    Returns the number of caps adjusted in this call (may be 0 if no data
    crossed the MIN_TRIGGERS_FOR_ADJUST threshold, or throttle in effect).
    Failures are logged but never propagate.
    """
    try:
        from src.state.calibration import _load_all_settled_entries
        all_entries = _load_all_settled_entries()
    except Exception as e:
        logger.warning(f"Cap evaluator: failed to load shadow log: {e}")
        return 0

    if not all_entries:
        return 0

    state = _load_state()
    caps_dict: Dict[str, Any] = state.setdefault("caps", {})
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    adjusted_count = 0

    for key, default in _INITIAL_DEFAULT_CAPS.items():
        sport, _, _ = key.partition(".")
        # Initialise entry if missing
        entry = caps_dict.setdefault(key, {
            "current_value":              default,
            "default_value":              default,
            "min_value":                  CAP_MIN,
            "max_value":                  CAP_MAX,
            "current_mode":               0,        # 0 = hard clip, 1 = tanh, 2 = reserved
            "last_adjusted_at":           None,
            "last_mode_change_at":        None,
            "adjustment_history":         [],
            "mode_change_history":        [],
            "current_counterfactual":     None,
            "current_mode_counterfactual": None,
        })
        # Back-fill mode fields on older state files
        entry.setdefault("current_mode", 0)
        entry.setdefault("last_mode_change_at", None)
        entry.setdefault("mode_change_history", [])
        entry.setdefault("current_mode_counterfactual", None)

        # Filter shadow log to this sport
        sport_entries = [
            e for e in all_entries
            if e.get("sport", "").lower() == sport
        ]

        # Filter by bet type for this specific cap
        cap_type_suffix = key.split(".", 1)[1]
        _bt_filter = _CAP_BET_TYPE_FILTER.get(cap_type_suffix)
        if _bt_filter is not None:
            sport_entries = [e for e in sport_entries if e.get("bet_type", "") in _bt_filter]

        # ── VALUE adjustment (Step 7) ─────────────────────────────────────
        cf = _compute_counterfactual(sport_entries)
        cf["computed_at"] = now_iso
        entry["current_counterfactual"] = cf

        value_changed = False
        if cf["recommendation"] not in ("insufficient", "keep"):
            # Throttle check
            last_adj = entry.get("last_adjusted_at")
            throttled = False
            if last_adj:
                try:
                    last_dt = datetime.fromisoformat(last_adj.replace("Z", "+00:00"))
                    if (now - last_dt) < timedelta(days=ADJUST_THROTTLE_DAYS):
                        throttled = True
                except Exception:
                    pass

            if not throttled:
                current = float(entry["current_value"])
                if cf["recommendation"] == "widen":
                    new_value = min(current + ADJUST_DELTA_PER_TICK, CAP_MAX)
                else:  # "tighten"
                    new_value = max(current - ADJUST_DELTA_PER_TICK, CAP_MIN)

                if abs(new_value - current) >= 0.0005:
                    entry["adjustment_history"].append({
                        "at":             now_iso,
                        "from":           round(current,   4),
                        "to":             round(new_value, 4),
                        "trigger_count":  cf["trigger_count"],
                        "raw_mae":        cf["raw_mae"],
                        "capped_mae":     cf["capped_mae"],
                        "recommendation": cf["recommendation"],
                    })
                    entry["current_value"]    = round(new_value, 4)
                    entry["last_adjusted_at"] = now_iso
                    adjusted_count += 1
                    value_changed = True
                    logger.info(
                        f"Cap auto-adjust: {key}: {current:.3f} → {new_value:.3f} "
                        f"(raw MAE {cf['raw_mae']:.3f} vs capped MAE {cf['capped_mae']:.3f}, n={cf['trigger_count']})"
                    )

        # ── MODE promotion (Step 6) ───────────────────────────────────────
        # Skip if value just changed this run — let value stabilise before
        # considering a structural change.
        if not value_changed:
            current_drift = float(entry["current_value"])
            mcf = _compute_mode_counterfactual(sport_entries, current_drift)
            mcf["computed_at"] = now_iso
            entry["current_mode_counterfactual"] = mcf

            if mcf["recommendation"] == "promote" and entry.get("current_mode", 0) == 0:
                # Throttle mode changes more aggressively
                last_mode = entry.get("last_mode_change_at")
                throttled_mode = False
                if last_mode:
                    try:
                        last_mode_dt = datetime.fromisoformat(last_mode.replace("Z", "+00:00"))
                        if (now - last_mode_dt) < timedelta(days=MODE_PROMOTION_THROTTLE_DAYS):
                            throttled_mode = True
                    except Exception:
                        pass

                if not throttled_mode:
                    entry["current_mode"] = 1
                    entry["last_mode_change_at"] = now_iso
                    entry["mode_change_history"].append({
                        "at":              now_iso,
                        "from_mode":       0,
                        "to_mode":         1,
                        "trigger_count":   mcf["trigger_count"],
                        "hard_clip_mae":   mcf["hard_clip_mae"],
                        "tanh_satur_mae":  mcf["tanh_satur_mae"],
                    })
                    adjusted_count += 1
                    logger.info(
                        f"Cap mode promote: {key}: Mode 0 (hard clip) → Mode 1 (tanh saturation) "
                        f"(hard MAE {mcf['hard_clip_mae']:.3f} vs tanh MAE {mcf['tanh_satur_mae']:.3f}, n={mcf['trigger_count']})"
                    )

    state["schema_version"] = SCHEMA_VERSION
    state["computed_at"]    = now_iso
    try:
        _save_state_atomic(state)
    except Exception as e:
        logger.error(f"Cap state save failed: {e}")
        return 0

    # Invalidate cache so subsequent get_current_cap() picks up new values
    global _CAP_CACHE
    _CAP_CACHE = None

    return adjusted_count


# ---------------------------------------------------------------------------
# Panel helper — load current state in a display-friendly shape
# ---------------------------------------------------------------------------

def load_panel_data() -> Dict[str, Any]:
    """
    Return a display-friendly view of the cap state for the report panel.
    """
    out = {
        "has_data":    False,
        "computed_at": None,
        "caps":        [],
        "bounds":      {"min": CAP_MIN, "max": CAP_MAX,
                        "tick": ADJUST_DELTA_PER_TICK,
                        "throttle_days": ADJUST_THROTTLE_DAYS,
                        "min_triggers": MIN_TRIGGERS_FOR_ADJUST},
    }
    state = _load_state()
    out["computed_at"] = state.get("computed_at")
    caps = state.get("caps", {}) or {}

    # Mode label lookup
    _MODE_LABELS = {0: "hard clip", 1: "tanh saturation", 2: "logistic blend"}

    # Build a row per known cap (include defaults even if not in state yet)
    for key, default in _INITIAL_DEFAULT_CAPS.items():
        sport, _, cap_type = key.partition(".")
        entry = caps.get(key, {})
        cf  = entry.get("current_counterfactual") or {}
        mcf = entry.get("current_mode_counterfactual") or {}
        history = entry.get("adjustment_history", []) or []
        mode_history = entry.get("mode_change_history", []) or []
        current = float(entry.get("current_value", default))
        current_mode = int(entry.get("current_mode", 0))
        delta_from_default = current - default

        # Status label
        if cf.get("recommendation") == "insufficient":
            status_label = f"Building ({cf.get('trigger_count', 0)} / {MIN_TRIGGERS_FOR_ADJUST} cap firings)"
        elif cf.get("recommendation") == "widen":
            status_label = "Cap suppressing signal — recommend widen"
        elif cf.get("recommendation") == "tighten":
            status_label = "Cap working — recommend tighten"
        elif cf.get("recommendation") == "keep":
            status_label = "Calibrated"
        else:
            status_label = "No data yet"

        # Mode promotion label (Step 6) — sensitive to whether we've already promoted
        if mcf.get("recommendation") == "insufficient":
            n_have = mcf.get("trigger_count", 0)
            mode_status = f"Building Mode 1 evaluation ({n_have} / {MIN_TRIGGERS_FOR_MODE_PROMOTION} firings)"
        elif mcf.get("recommendation") == "promote":
            if current_mode == 0:
                mode_status = "Mode 1 (tanh) outperforming hard clip — promotion queued"
            else:
                mode_status = f"Mode {current_mode} ({_MODE_LABELS.get(current_mode, '?')}) active — still outperforming alternatives"
        elif mcf.get("recommendation") == "keep":
            mode_status = f"Mode {current_mode} ({_MODE_LABELS.get(current_mode, '?')}) — best-fitting form"
        else:
            mode_status = f"Mode {current_mode} ({_MODE_LABELS.get(current_mode, '?')})"

        out["caps"].append({
            "sport":              sport.upper(),
            "cap_type":           cap_type,
            "display_label":      _DISPLAY_LABELS.get(key, f"{sport.upper()} {cap_type}"),
            "current":            round(current, 4),
            "default":            round(default, 4),
            "delta_from_default": round(delta_from_default, 4),
            "delta_display":      f"{delta_from_default:+.3f}" if abs(delta_from_default) >= 0.0005 else "—",
            "current_mode":       current_mode,
            "current_mode_label": _MODE_LABELS.get(current_mode, "?"),
            "last_adjusted_at":   entry.get("last_adjusted_at"),
            "last_mode_change_at": entry.get("last_mode_change_at"),
            "adjustment_count":   len(history),
            "mode_change_count":  len(mode_history),
            "counterfactual":     cf,
            "mode_counterfactual": mcf,
            "status_label":       status_label,
            "mode_status_label":  mode_status,
            "recent_history":     history[-3:],
            "recent_mode_history": mode_history[-3:],
        })

    out["caps"].sort(key=lambda c: (_SPORT_ORDER.get(c["sport"].lower(), 99), _BET_TYPE_ORDER.get(c["cap_type"], 99)))
    out["has_data"] = any(c["counterfactual"] or c["mode_counterfactual"] for c in out["caps"])
    return out
