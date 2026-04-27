"""
Phase 1 Outcome Settlement — checks ESPN for yesterday's final scores
and settles each pick as WON / LOST / PUSH, appending to state/history.json.

Called once per daily run (before today's analysis) to close out yesterday's picks.
Idempotent: already-settled picks are skipped via (date, game, bet_type, pick) dedup.
"""
import json
import logging
import re
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import HISTORY_FILE

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
ESPN_SPORT_PATHS = {
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
}

HISTORY_PATH = Path(HISTORY_FILE)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_and_settle(today: date) -> int:
    """
    Settle yesterday's picks against ESPN final scores.
    Appends new settled records to state/history.json.
    Returns the number of picks settled this run.
    """
    from datetime import timedelta
    yesterday = today - timedelta(days=1)

    # Avoid circular import — manager imports config, not outcome_checker
    from src.state.manager import load_state
    state = load_state(yesterday)
    if not state:
        logger.info(f"No state for {yesterday} — nothing to settle")
        return 0

    # Load already-settled records to avoid double counting
    existing = _load_history()
    settled_keys = {
        (r["date"], r["game"], r["bet_type"], r["pick"])
        for r in existing
    }

    # Fetch ESPN final scores for each sport we need
    sports_needed = set()
    for pick in state.get("singles", []):
        sports_needed.add(pick.get("sport", "").upper())
    for par in state.get("parlays", []):
        for leg in par.get("legs", []):
            sports_needed.add(leg.get("sport", "").upper())

    score_cache: Dict[str, Dict] = {}
    for sport in sports_needed & ESPN_SPORT_PATHS.keys():
        score_cache[sport] = _fetch_espn_final_scores(sport, yesterday)

    new_records: List[Dict] = []

    # ── Singles ──────────────────────────────────────────────────────────────
    for pick in state.get("singles", []):
        sport = pick.get("sport", "").upper()
        key = (yesterday.isoformat(), pick.get("game", ""), pick.get("bet_type", ""), pick.get("pick", ""))
        if key in settled_keys:
            continue

        score_data = _find_game_score(
            score_cache.get(sport, {}),
            pick.get("home_team", ""),
            pick.get("away_team", ""),
        )
        if not score_data:
            logger.debug(f"Score not found for {pick.get('game')} on {yesterday}")
            continue

        result = _determine_outcome(
            pick.get("pick", ""),
            pick.get("bet_type", ""),
            pick.get("home_team", ""),
            pick.get("away_team", ""),
            score_data["home_score"],
            score_data["away_score"],
        )
        if result not in ("WON", "LOST", "PUSH"):
            continue

        cost        = pick.get("total_cost", 0) or 0
        profit      = pick.get("profit_if_win", 0) or 0
        actual_pnl  = profit if result == "WON" else (0.0 if result == "PUSH" else -cost)

        new_records.append({
            "date":             yesterday.isoformat(),
            "type":             "single",
            "sport":            sport,
            "game":             pick.get("game", ""),
            "bet_type":         pick.get("bet_type", ""),
            "pick":             pick.get("pick", ""),
            "edge_pct":         pick.get("edge_pct", 0),
            "confidence":       pick.get("confidence", "MEDIUM"),
            "model_prob_pct":   pick.get("model_prob_pct", 0),
            "market_prob_pct":  pick.get("market_prob_pct", 0),
            "cost":             cost,
            "profit_if_win":    profit,
            "result":           result,
            "actual_pnl":       round(actual_pnl, 2),
        })

    # ── Parlays ───────────────────────────────────────────────────────────────
    for par in state.get("parlays", []):
        label = par.get("label", "")
        key = (yesterday.isoformat(), label, "Parlay", label)
        if key in settled_keys:
            continue

        legs = par.get("legs", [])
        if not legs:
            continue

        leg_results: List[str] = []
        for leg in legs:
            sport = leg.get("sport", "").upper()
            score_data = _find_game_score(
                score_cache.get(sport, {}),
                leg.get("home_team", ""),
                leg.get("away_team", ""),
            )
            if not score_data:
                leg_results.append("UNKNOWN")
                continue
            r = _determine_outcome(
                leg.get("pick", ""),
                leg.get("bet_type", ""),
                leg.get("home_team", ""),
                leg.get("away_team", ""),
                score_data["home_score"],
                score_data["away_score"],
            )
            leg_results.append(r)

        if "UNKNOWN" in leg_results:
            continue  # Can't settle until all legs have final scores

        if any(r == "LOST" for r in leg_results):
            result = "LOST"
        elif all(r == "WON" for r in leg_results):
            result = "WON"
        else:
            result = "PUSH"  # one or more legs pushed, no leg lost

        cost        = par.get("total_cost", 0) or 0
        profit      = par.get("profit_if_win", 0) or 0
        actual_pnl  = profit if result == "WON" else (0.0 if result == "PUSH" else -cost)

        new_records.append({
            "date":             yesterday.isoformat(),
            "type":             "parlay",
            "sport":            "PARLAY",
            "game":             label,
            "bet_type":         "Parlay",
            "pick":             label,
            "edge_pct":         par.get("edge_pct", 0),
            "confidence":       par.get("confidence", "MEDIUM"),
            "model_prob_pct":   par.get("combined_prob_pct", 0),
            "market_prob_pct":  0,
            "cost":             cost,
            "profit_if_win":    profit,
            "result":           result,
            "actual_pnl":       round(actual_pnl, 2),
            "legs":             [
                {"game": l.get("game"), "pick": l.get("pick"), "result": leg_results[i]}
                for i, l in enumerate(legs)
            ],
        })

    if new_records:
        _append_to_history(new_records)
        won   = sum(1 for r in new_records if r["result"] == "WON")
        lost  = sum(1 for r in new_records if r["result"] == "LOST")
        push  = sum(1 for r in new_records if r["result"] == "PUSH")
        pnl   = sum(r["actual_pnl"] for r in new_records)
        logger.info(
            f"Settled {len(new_records)} pick(s) from {yesterday}: "
            f"{won}W / {lost}L / {push}P | PnL: ${pnl:+.2f}"
        )
    else:
        logger.info(f"No new picks to settle from {yesterday}")

    return len(new_records)


# ---------------------------------------------------------------------------
# ESPN score fetching
# ---------------------------------------------------------------------------

def _fetch_espn_final_scores(sport: str, game_date: date) -> Dict:
    """
    Returns a dict keyed by frozenset({team_name_1, team_name_2}) → score data.
    Only includes completed games.
    """
    path = ESPN_SPORT_PATHS.get(sport)
    if not path:
        return {}

    date_str = game_date.strftime("%Y%m%d")
    url = f"{ESPN_BASE}/{path}/scoreboard"
    try:
        r = requests.get(url, params={"dates": date_str}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"ESPN fetch failed ({sport}, {game_date}): {e}")
        return {}

    scores: Dict = {}
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        # Only settled games
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        # Build a lookup from homeAway → team name + score
        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_name  = home_comp.get("team", {}).get("displayName", "")
        away_name  = away_comp.get("team", {}).get("displayName", "")
        home_abbr  = home_comp.get("team", {}).get("abbreviation", "")
        away_abbr  = away_comp.get("team", {}).get("abbreviation", "")

        try:
            home_score = float(home_comp.get("score", 0))
            away_score = float(away_comp.get("score", 0))
        except (TypeError, ValueError):
            continue

        entry = {
            "home_name":  home_name,
            "away_name":  away_name,
            "home_abbr":  home_abbr,
            "away_abbr":  away_abbr,
            "home_score": home_score,
            "away_score": away_score,
        }
        # Index by both name variants for flexible lookup
        scores[home_name.lower()] = entry
        scores[away_name.lower()] = entry
        if home_abbr:
            scores[home_abbr.lower()] = entry
        if away_abbr:
            scores[away_abbr.lower()] = entry

    logger.debug(f"ESPN {sport} {game_date}: {len(scores)//4} completed game(s) found")
    return scores


def _find_game_score(
    scores: Dict,
    home_team: str,
    away_team: str,
) -> Optional[Dict]:
    """
    Match a pick's home/away team pair against ESPN score data.
    Returns the score entry or None if not found / not yet final.
    """
    if not scores:
        return None

    # Try direct lookup by home team name (case-insensitive, partial)
    for query in [home_team.lower(), away_team.lower()]:
        entry = scores.get(query)
        if entry:
            return entry
        # Partial match: check if query is a substring of any key
        for key, val in scores.items():
            if query and (query in key or key in query):
                return val

    return None


# ---------------------------------------------------------------------------
# Outcome determination
# ---------------------------------------------------------------------------

def _determine_outcome(
    pick: str,
    bet_type: str,
    home_team: str,
    away_team: str,
    home_score: float,
    away_score: float,
) -> str:
    """
    Returns 'WON', 'LOST', or 'PUSH'.
    Handles Moneyline, Total (Over/Under), and Spread bets.
    """
    bt = bet_type.lower().strip()

    # ── Moneyline ────────────────────────────────────────────────────────────
    if bt in ("moneyline", "h2h"):
        home_won = home_score > away_score
        # pick should be the team name (or contain it)
        pick_lower = pick.lower()
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        if home_score == away_score:
            return "PUSH"

        # Check if pick refers to home team
        if home_lower and (home_lower in pick_lower or pick_lower in home_lower):
            return "WON" if home_won else "LOST"
        # Check if pick refers to away team
        if away_lower and (away_lower in pick_lower or pick_lower in away_lower):
            return "WON" if not home_won else "LOST"

        logger.warning(f"Moneyline: could not match pick '{pick}' to '{home_team}' or '{away_team}'")
        return "UNKNOWN"

    # ── Total (Over / Under) ─────────────────────────────────────────────────
    if bt == "total":
        match = re.search(r"(over|under)\s+([\d.]+)", pick, re.IGNORECASE)
        if not match:
            logger.warning(f"Total: cannot parse line from pick '{pick}'")
            return "UNKNOWN"
        direction = match.group(1).lower()
        line      = float(match.group(2))
        actual    = home_score + away_score

        if actual == line:
            return "PUSH"
        over_hit = actual > line
        return "WON" if (direction == "over") == over_hit else "LOST"

    # ── Spread ───────────────────────────────────────────────────────────────
    if bt in ("spread", "point_spread"):
        # pick format: "Boston Celtics +3.5" or "Los Angeles Lakers -7.0"
        match = re.search(r"([+-][\d.]+)\s*$", pick)
        if not match:
            logger.warning(f"Spread: cannot parse spread from pick '{pick}'")
            return "UNKNOWN"
        spread = float(match.group(1))

        # Determine which team the spread applies to
        pick_team = re.sub(r"[+-][\d.]+\s*$", "", pick).strip().lower()
        home_lower = home_team.lower()
        away_lower = away_team.lower()

        if home_lower and (home_lower in pick_team or pick_team in home_lower):
            adjusted = home_score + spread - away_score
        elif away_lower and (away_lower in pick_team or pick_team in away_lower):
            adjusted = away_score + spread - home_score
        else:
            logger.warning(f"Spread: could not match pick team '{pick_team}' to '{home_team}' or '{away_team}'")
            return "UNKNOWN"

        if adjusted == 0:
            return "PUSH"
        return "WON" if adjusted > 0 else "LOST"

    logger.warning(f"Unrecognised bet type '{bet_type}' — cannot settle")
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# History file I/O
# ---------------------------------------------------------------------------

def _load_history() -> List[Dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []


def _append_to_history(records: List[Dict]) -> None:
    existing = _load_history()
    existing.extend(records)
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"History updated → {HISTORY_PATH} ({len(existing)} total records)")
    except Exception as e:
        logger.error(f"Failed to write history: {e}")


# ---------------------------------------------------------------------------
# Quick summary helper (used by report generator later)
# ---------------------------------------------------------------------------

def load_performance_summary() -> Dict:
    """
    Returns a summary dict for the report:
      total_bets, won, lost, push, win_rate_pct, total_pnl,
      roi_pct (PnL / total_cost * 100), by_confidence, recent_30
    """
    records = _load_history()
    if not records:
        return {}

    def _stats(subset: List[Dict]) -> Dict:
        won   = sum(1 for r in subset if r.get("result") == "WON")
        lost  = sum(1 for r in subset if r.get("result") == "LOST")
        push  = sum(1 for r in subset if r.get("result") == "PUSH")
        total = won + lost + push
        pnl   = sum(r.get("actual_pnl", 0) for r in subset)
        cost  = sum(r.get("cost", 0) for r in subset)
        return {
            "total": total,
            "won":   won,
            "lost":  lost,
            "push":  push,
            "win_rate_pct": round(won / (won + lost) * 100, 1) if (won + lost) > 0 else None,
            "total_pnl":    round(pnl, 2),
            "roi_pct":      round(pnl / cost * 100, 1) if cost > 0 else None,
        }

    settled = [r for r in records if r.get("result") in ("WON", "LOST", "PUSH")]
    summary = _stats(settled)

    # Break down by confidence tier
    by_conf: Dict[str, Dict] = {}
    for conf in ("HIGH", "MEDIUM"):
        subset = [r for r in settled if r.get("confidence") == conf]
        if subset:
            by_conf[conf] = _stats(subset)
    summary["by_confidence"] = by_conf

    # Break down by sport
    by_sport: Dict[str, Dict] = {}
    for sport in ("NBA", "MLB", "PARLAY"):
        subset = [r for r in settled if r.get("sport") == sport]
        if subset:
            by_sport[sport] = _stats(subset)
    summary["by_sport"] = by_sport

    # Last 30 bets (chronological)
    summary["recent_30"] = settled[-30:]

    # Full settled list (used by JS for live chart updates)
    summary["all_records"] = settled

    return summary


# ---------------------------------------------------------------------------
# Phase 2 — Chart data (pre-computed SVG coordinates)
# ---------------------------------------------------------------------------

# SVG canvas constants (viewBox="0 0 460 130")
_SVG_W, _SVG_H   = 460, 130
_PAD_L, _PAD_R   = 42, 12
_PAD_T, _PAD_B   = 10, 28
_PLOT_W = _SVG_W - _PAD_L - _PAD_R   # 406
_PLOT_H = _SVG_H - _PAD_T - _PAD_B   # 92


def _sx(i: int, n: int) -> float:
    """Map index i (0-based) to SVG x coordinate."""
    if n <= 1:
        return _PAD_L + _PLOT_W / 2
    return _PAD_L + i / (n - 1) * _PLOT_W


def _sy(val: float, lo: float, hi: float) -> float:
    """Map value to SVG y coordinate (inverted — SVG y increases downward)."""
    if hi == lo:
        return _PAD_T + _PLOT_H / 2
    frac = (val - lo) / (hi - lo)
    return _PAD_T + _PLOT_H * (1 - frac)


def _points_str(coords: list) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)


def build_chart_data() -> Dict:
    """
    Returns pre-computed chart data for Phase 2 visualizations:
      - Cumulative PnL SVG polyline
      - Rolling win-rate SVG polyline
      - Calibration table rows
    All SVG coordinates are computed here so the Jinja template stays simple.
    """
    records = _load_history()
    settled = [r for r in records if r.get("result") in ("WON", "LOST", "PUSH")]

    if len(settled) < 3:
        return {"has_data": False}

    # ── Cumulative PnL series ────────────────────────────────────────────────
    cum = 0.0
    pnl_series: List[Dict] = []
    for i, r in enumerate(settled):
        cum += r.get("actual_pnl", 0.0)
        pnl_series.append({"i": i, "date": r.get("date", ""), "cum": round(cum, 2)})

    pnl_vals   = [p["cum"] for p in pnl_series]
    pnl_lo     = min(pnl_vals + [0])   # always include 0 in range
    pnl_hi     = max(pnl_vals + [0])
    if pnl_hi == pnl_lo:
        pnl_hi = pnl_lo + 1

    n = len(pnl_series)
    pnl_coords = [(_sx(p["i"], n), _sy(p["cum"], pnl_lo, pnl_hi)) for p in pnl_series]

    # Zero line y-position
    zero_y = round(_sy(0, pnl_lo, pnl_hi), 1)

    # Y-axis labels (top, zero if interior, bottom) — deduplicate when zero == min or max
    pnl_y_labels = [
        {"y": round(_sy(pnl_hi, pnl_lo, pnl_hi), 1), "val": f"${pnl_hi:+.0f}"},
        {"y": round(_sy(pnl_lo, pnl_lo, pnl_hi), 1), "val": f"${pnl_lo:+.0f}"},
    ]
    # Only add zero label if it's meaningfully between min and max (not at either edge)
    if pnl_lo < 0 < pnl_hi:
        pnl_y_labels.insert(1, {"y": round(zero_y, 1), "val": "$0"})

    # Colour segments: green above zero, red below
    # Split polyline into above-zero and below-zero segments
    pnl_above = _points_str([(x, y) for (x, y), p in zip(pnl_coords, pnl_series) if p["cum"] >= 0])
    pnl_below = _points_str([(x, y) for (x, y), p in zip(pnl_coords, pnl_series) if p["cum"] < 0])
    pnl_all   = _points_str(pnl_coords)

    # ── Rolling win-rate series (window = 20, show from bet 3 onwards) ───────
    WINDOW     = 20
    MIN_SHOW   = 3
    wr_series: List[Dict] = []
    for i in range(MIN_SHOW - 1, len(settled)):
        window    = settled[max(0, i - WINDOW + 1): i + 1]
        decisions = [r for r in window if r.get("result") in ("WON", "LOST")]
        if not decisions:
            continue
        rate = sum(1 for r in decisions if r["result"] == "WON") / len(decisions) * 100
        wr_series.append({"i": i, "rate": round(rate, 1)})

    wr_points = ""
    ref50_y   = round(_sy(50, 0, 100), 1)
    if wr_series:
        nw = len(settled)
        wr_coords  = [(_sx(p["i"], nw), _sy(p["rate"], 0, 100)) for p in wr_series]
        wr_points  = _points_str(wr_coords)

    wr_y_labels = [
        {"y": round(_sy(100, 0, 100), 1), "val": "100%"},
        {"y": round(_sy(50,  0, 100), 1), "val":  "50%"},
        {"y": round(_sy(0,   0, 100), 1), "val":   "0%"},
    ]

    # ── Calibration table ────────────────────────────────────────────────────
    BUCKETS = [
        ("Under 15%", 0.00, 0.15),
        ("15–25%",    0.15, 0.25),
        ("25%+",      0.25, 1.00),
    ]
    calibration = []
    for label, lo, hi in BUCKETS:
        bets      = [r for r in settled if lo <= r.get("edge_pct", 0) / 100 < hi
                     or (hi == 1.0 and r.get("edge_pct", 0) / 100 >= lo)]
        decisions = [r for r in bets if r.get("result") in ("WON", "LOST")]
        won       = sum(1 for r in decisions if r["result"] == "WON")
        actual    = round(won / len(decisions) * 100, 1) if decisions else None
        avg_model = round(
            sum(r.get("model_prob_pct", 50) for r in bets) / len(bets), 1
        ) if bets else None
        # Delta: positive means model was pessimistic (we beat expectations)
        delta     = round(actual - avg_model, 1) if (actual is not None and avg_model is not None) else None
        calibration.append({
            "bucket":          label,
            "count":           len(bets),
            "decisions":       len(decisions),
            "actual_win_pct":  actual,
            "avg_model_pct":   avg_model,
            "delta":           delta,
        })

    # X-axis: first and last date labels
    x_labels = []
    if pnl_series:
        x_labels = [
            {"x": round(_sx(0,  n), 1), "val": pnl_series[0]["date"][5:]},   # MM-DD
            {"x": round(_sx(n-1, n), 1), "val": pnl_series[-1]["date"][5:]},
        ]
        if n > 2:
            mid = n // 2
            x_labels.insert(1, {"x": round(_sx(mid, n), 1), "val": pnl_series[mid]["date"][5:]})

    return {
        "has_data":       True,
        "total_bets":     len(settled),
        "svg_w":          _SVG_W,
        "svg_h":          _SVG_H,
        "pad_l":          _PAD_L,
        "pad_t":          _PAD_T,
        "plot_h":         _PLOT_H,
        # PnL chart
        "pnl_all":        pnl_all,
        "pnl_above":      pnl_above,
        "pnl_below":      pnl_below,
        "zero_y":         zero_y,
        "pnl_y_labels":   pnl_y_labels,
        "x_labels":       x_labels,
        "final_pnl":      round(cum, 2),
        # Win-rate chart
        "wr_points":      wr_points,
        "ref50_y":        ref50_y,
        "wr_y_labels":    wr_y_labels,
        "has_wr":         bool(wr_series),
        # Calibration
        "calibration":    calibration,
    }
