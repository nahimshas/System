"""
Rule-based recap generator for yesterday's settled bets.

Reads from state/history.json (actual outcomes + scores) and yesterday's
state file (signals/research/projected scores).  No external API calls.
Included in the morning email as "Yesterday's Results" so you can see
what happened and why at a glance.
"""
import re
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal / projection helpers
# ---------------------------------------------------------------------------

def _parse_projected_scores(signals: List[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts (proj_home, proj_away) from a signal like:
      'Model projected score: Chicago White Sox 3.7 — Los Angeles Angels 4.8'
    Returns (None, None) if not found.
    """
    for sig in signals:
        if "projected score" in sig.lower():
            nums = re.findall(r"\d+\.?\d*", sig)
            if len(nums) >= 2:
                return float(nums[-2]), float(nums[-1])
    return None, None


def _parse_nba_projected_total(signals: List[str]) -> Optional[float]:
    """Extracts expected total from 'Model expected total: X.X vs market line Y'."""
    for sig in signals:
        if "expected total" in sig.lower():
            nums = re.findall(r"\d+\.?\d*", sig)
            if nums:
                return float(nums[0])
    return None


def _pick_key_signals(signals: List[str]) -> List[str]:
    """
    Returns up to 2 signals most relevant to understanding the outcome.
    Skips projected-score lines (those appear in the narrative itself).
    Prioritises signals mentioning meaningful context: pitcher quality,
    umpire, weather, injuries, park, pace/rest.
    """
    skip = ("model projected", "model expected", "home court advantage",
            "playoffs:", "rating edge")
    priority_keywords = [
        "FIP", "ERA", "umpire", "weather", "wind", "rain", "precipitation",
        "injury", "park factor", "bullpen", "B2B", "back-to-back",
        "schedule load", "net rating", "OPS",
    ]
    keep = [
        s for s in signals
        if not any(p in s.lower() for p in skip) and len(s) > 10
    ]
    scored = sorted(
        keep,
        key=lambda s: sum(1 for kw in priority_keywords if kw.lower() in s.lower()),
        reverse=True,
    )
    return scored[:2]


# ---------------------------------------------------------------------------
# Narrative builder per bet type / sport
# ---------------------------------------------------------------------------

def _narrative_mlb(rec: Dict, pick_str: str, bet_type: str,
                   home: str, away: str, signals: List[str],
                   actual_home, actual_away) -> str:
    parts = []

    # Score line
    if actual_home is not None and actual_away is not None:
        parts.append(f"Final: {away} {int(actual_away)} @ {home} {int(actual_home)}.")

    proj_home, proj_away = _parse_projected_scores(signals)

    if proj_home is not None:
        parts.append(
            f"Model projected: {home} {proj_home:.1f}, {away} {proj_away:.1f}."
        )

    result = rec["result"]

    if actual_home is not None and actual_away is not None and proj_home is not None:
        proj_margin   = proj_home - proj_away
        actual_margin = actual_home - actual_away

        if bet_type in ("Moneyline", "Spread"):
            correct_side = (proj_margin > 0) == (actual_margin > 0)
            if result == "WON":
                if correct_side:
                    spread_m = re.search(r"[+-]\d+\.?\d*$", pick_str)
                    if spread_m:
                        parts.append(
                            f"Covered {spread_m.group()} — actual margin "
                            f"{'+' if actual_margin > 0 else ''}{actual_margin:.1f}."
                        )
                    else:
                        parts.append("Model's directional call was correct.")
                else:
                    parts.append("Won despite model margin miss — market overpriced the favourite.")
            else:
                if not correct_side:
                    parts.append("Actual margin reversed model's projection.")
                else:
                    spread_m = re.search(r"[+-]\d+\.?\d*$", pick_str)
                    if spread_m:
                        parts.append(
                            f"Correct side but didn't cover {spread_m.group()} "
                            f"(actual margin {'+' if actual_margin > 0 else ''}{actual_margin:.1f})."
                        )

        elif bet_type == "Total":
            actual_total = actual_home + actual_away
            proj_total   = proj_home   + proj_away
            line_m = re.search(r"[\d.]+", pick_str.split()[-1])
            if line_m:
                line = float(line_m.group())
                direction = "Over" if pick_str.lower().startswith("over") else "Under"
                diff = actual_total - line
                parts.append(
                    f"Actual total {int(actual_total)} vs line {line:.1f} "
                    f"({'over' if diff > 0 else 'under'} by {abs(diff):.1f}). "
                    f"Model projected {proj_total:.1f}."
                )

    # Contextual flags for losses
    if result == "LOST":
        rain_sig = next(
            (s for s in signals if "rain" in s.lower() or "precipitation" in s.lower()), None
        )
        if rain_sig:
            parts.append("⚠ Weather/rain likely a factor.")

    return " ".join(parts)


def _narrative_nba(rec: Dict, pick_str: str, bet_type: str,
                   home: str, away: str, signals: List[str],
                   actual_home, actual_away) -> str:
    parts = []
    result = rec["result"]

    if actual_home is not None and actual_away is not None:
        parts.append(f"Final: {away} {int(actual_away)} @ {home} {int(actual_home)}.")

    if bet_type == "Total":
        proj_total = _parse_nba_projected_total(signals)
        if actual_home is not None and actual_away is not None:
            actual_total = actual_home + actual_away
            line_m = re.search(r"[\d.]+", pick_str.split()[-1])
            if line_m:
                line = float(line_m.group())
                diff  = actual_total - line
                parts.append(
                    f"Actual total {int(actual_total)} vs line {line:.1f} "
                    f"({'over' if diff > 0 else 'under'} by {abs(diff):.1f})."
                    + (f" Model projected {proj_total:.1f}." if proj_total else "")
                )
    else:
        model_pct = rec.get("model_prob_pct", 0)
        if model_pct:
            parts.append(f"Model win probability: {model_pct:.1f}%.")
        if actual_home is not None and actual_away is not None:
            actual_margin = actual_home - actual_away
            if bet_type == "Spread":
                spread_m = re.search(r"[+-]\d+\.?\d*$", pick_str)
                if spread_m:
                    parts.append(
                        f"Actual margin {'+' if actual_margin > 0 else ''}{int(actual_margin)} "
                        f"(needed {spread_m.group()})."
                    )

    return " ".join(parts)


def _narrative_parlay(rec: Dict) -> str:
    """Summarise each parlay leg result."""
    leg_parts = []
    for leg in rec.get("legs", []):
        icon = "✅" if leg.get("result") == "WON" else "❌" if leg.get("result") == "LOST" else "➡"
        leg_parts.append(f"{icon} {leg.get('pick', '')} ({leg.get('game', '')})")
    return " · ".join(leg_parts)


# ---------------------------------------------------------------------------
# Item builder
# ---------------------------------------------------------------------------

def _build_item(rec: Dict, bet: Dict) -> Dict:
    sport    = rec.get("sport", "")
    bet_type = rec.get("bet_type", "")
    pick_str = rec.get("pick", "")
    home     = rec.get("home_team") or bet.get("home_team", "")
    away     = rec.get("away_team") or bet.get("away_team", "")
    signals  = bet.get("signals", [])
    ah       = rec.get("actual_home_score")
    aa       = rec.get("actual_away_score")

    if bet_type == "Parlay":
        narrative = _narrative_parlay(rec)
    elif sport == "MLB":
        narrative = _narrative_mlb(rec, pick_str, bet_type, home, away, signals, ah, aa)
    elif sport == "NBA":
        narrative = _narrative_nba(rec, pick_str, bet_type, home, away, signals, ah, aa)
    else:
        narrative = ""

    return {
        "result":        rec["result"],
        "sport":         sport,
        "bet_type":      bet_type,
        "pick":          pick_str,
        "game":          rec.get("game", ""),
        "edge_pct":      rec.get("edge_pct", 0),
        "model_prob_pct": rec.get("model_prob_pct", 0),
        "cost":          rec.get("cost", 0),
        "profit_if_win": rec.get("profit_if_win", 0),
        "actual_pnl":    rec.get("actual_pnl", 0),
        "home_team":     home,
        "away_team":     away,
        "actual_home":   ah,
        "actual_away":   aa,
        "narrative":     narrative,
        "key_signals":   _pick_key_signals(signals),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_recap(yesterday: date) -> Dict:
    """
    Returns recap data for yesterday's settled bets.
    Always safe to call — returns {"has_recap": False} on any failure
    or when there are no settled bets.
    """
    try:
        from src.data.outcome_checker import _load_history
        from src.state.manager import load_state
    except Exception as e:
        logger.warning(f"Recap import error: {e}")
        return {"has_recap": False}

    try:
        all_records = _load_history()
    except Exception as e:
        logger.warning(f"Recap: history load failed: {e}")
        return {"has_recap": False}

    yest_str     = yesterday.isoformat()
    yest_records = [
        r for r in all_records
        if r.get("date") == yest_str and r.get("result") in ("WON", "LOST", "PUSH")
    ]

    if not yest_records:
        return {"has_recap": False}

    # Load yesterday's state file for signals / research context
    try:
        state = load_state(yesterday) or {}
    except Exception:
        state = {}

    # Build lookup: (game, bet_type, pick) → full bet dict from state
    bet_lookup: Dict[tuple, Dict] = {}
    for pick in state.get("singles", []):
        key = (pick.get("game", ""), pick.get("bet_type", ""), pick.get("pick", ""))
        bet_lookup[key] = pick
    for par in state.get("parlays", []):
        lbl = par.get("label", "")
        bet_lookup[(lbl, "Parlay", lbl)] = par

    items    = []
    won = lost = push = 0
    total_pnl = 0.0

    for rec in yest_records:
        result = rec["result"]
        if result == "WON":    won  += 1
        elif result == "LOST": lost += 1
        else:                  push += 1
        total_pnl += rec.get("actual_pnl", 0.0)

        key = (rec.get("game", ""), rec.get("bet_type", ""), rec.get("pick", ""))
        bet = bet_lookup.get(key, {})
        try:
            items.append(_build_item(rec, bet))
        except Exception as e:
            logger.warning(f"Recap item build failed ({rec.get('pick')}): {e}")

    record_str = f"{won}W–{lost}L" + (f"–{push}P" if push else "")
    pnl_sign   = "+" if total_pnl >= 0 else ""

    return {
        "has_recap":  True,
        "date":       yesterday.strftime("%A, %B %d"),
        "won":        won,
        "lost":       lost,
        "push":       push,
        "total_pnl":  round(total_pnl, 2),
        "pnl_sign":   pnl_sign,
        "record_str": record_str,
        "results":      items,
    }
