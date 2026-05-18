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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import HISTORY_FILE

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"
PROP_HISTORY_PATH = Path("state/prop_history.json")
ESPN_SPORT_PATHS = {
    "NBA": "basketball/nba",
    "MLB": "baseball/mlb",
}

# Watchlist-only leagues (NHL, IPL, WNBA, MLS) — separate history file, no PnL tracking
WATCHLIST_HISTORY_PATH = Path("state/watchlist_history.json")
ESPN_WATCHLIST_PATHS = {
    "NHL": "hockey/nhl",
    "IPL": "cricket/ipl",
    "WNBA": "basketball/wnba",
    "MLS": "soccer/usa.1",
}

# Rolling pending list for leagues whose games finish AFTER the morning run.
# Key = sport string, value = expected game duration in hours (used as settlement gate).
# Sports not listed here (NHL) settle date-based via check_and_settle_watchlist().
# Add future leagues here without touching any other constants:
#   "EPL": 2.5   — European soccer (~2h game + 30 min buffer)
#   "BBL": 4.5   — Big Bash League (same timing profile as IPL)
WATCHLIST_PENDING_PATH = Path("state/watchlist_pending.json")
WATCHLIST_PENDING_SPORTS: Dict[str, float] = {
    "IPL": 4.5,   # ~3.5 hr match + 1 hr buffer for rain delays / super overs
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
            "home_team":        pick.get("home_team", ""),
            "away_team":        pick.get("away_team", ""),
            "edge_pct":         pick.get("edge_pct", 0),
            "confidence":       pick.get("confidence", "MEDIUM"),
            "model_prob_pct":   pick.get("model_prob_pct", 0),
            "market_prob_pct":  pick.get("market_prob_pct", 0),
            "cost":             cost,
            "profit_if_win":    profit,
            "result":           result,
            "actual_pnl":       round(actual_pnl, 2),
            # Calibration: actual final scores for post-mortem analysis
            "actual_home_score": score_data["home_score"],
            "actual_away_score": score_data["away_score"],
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


_TEAM_NAME_SKIP_TOKENS = {
    "fc", "sc", "ac", "cf", "cd", "sk",
    "new", "los", "san", "fort", "real", "club", "the",
}


def _team_name_tokens(s: str) -> list:
    """Tokenise a team name for fallback matching — drops league suffixes and
    generic city prefixes that don't disambiguate teams."""
    if not s:
        return []
    return [
        t for t in s.lower().replace(".", " ").split()
        if len(t) >= 3 and t not in _TEAM_NAME_SKIP_TOKENS
    ]


def _team_token_overlap_match(query: str, key: str) -> bool:
    """
    Token-overlap fallback for team-name matching. Handles names that defeat
    substring matching due to reordering or singular/plural variants.

    Example: ESPN's "Red Bull New York" vs Odds API's "New York Red Bulls" —
    last token "bulls" (plural) doesn't appear in "red bull new york"
    (singular). Token overlap catches it via bull↔bulls prefix match.

    Requires ≥ 2 substantial token overlaps to avoid false positives like
    NYCFC ↔ NY Red Bulls (only "york" overlaps).
    """
    q_tok = _team_name_tokens(query)
    k_tok = _team_name_tokens(key)
    if len(q_tok) < 2 or len(k_tok) < 2:
        return False
    used = set()
    matched = 0
    for a in q_tok:
        for j, b in enumerate(k_tok):
            if j in used:
                continue
            # Exact OR one-is-prefix-of-other (handles bull/bulls, dynamo/dynamos)
            if a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))):
                used.add(j)
                matched += 1
                break
    return matched >= 2


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

    # Token-overlap fallback — handles reordered / singular-vs-plural names
    # (e.g. ESPN "Red Bull New York" ↔ Odds API "New York Red Bulls").
    for query in [home_team, away_team]:
        if not query:
            continue
        for key, val in scores.items():
            if _team_token_overlap_match(query, key):
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


def _determine_mls_outcome(pick: str, bet_type: str, home_team: str, away_team: str,
                            home_score: float, away_score: float) -> str:
    """
    MLS-specific outcome determination.
    Key difference from _determine_outcome():
      - Moneyline: draw (home_score == away_score) -> LOST (not PUSH)
      - Draw bet_type: WON if draw, LOST otherwise
      - Total, Spread: delegate to _determine_outcome()
    """
    bt = bet_type.lower().strip()
    if bt == "draw":
        return "WON" if home_score == away_score else "LOST"
    if bt in ("moneyline", "h2h"):
        if home_score == away_score:
            return "LOST"  # soccer: draw = loss for ML picks
        pick_lower = pick.lower()
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        home_won = home_score > away_score
        if home_lower and (home_lower in pick_lower or pick_lower in home_lower):
            return "WON" if home_won else "LOST"
        if away_lower and (away_lower in pick_lower or pick_lower in away_lower):
            return "WON" if not home_won else "LOST"
        return "UNKNOWN"
    # Total and Spread use existing logic
    return _determine_outcome(pick, bet_type, home_team, away_team, home_score, away_score)


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
        ("3–8%",   0.03, 0.08),
        ("8–15%",  0.08, 0.15),
        ("15–20%", 0.15, 0.20),
        ("20–25%", 0.20, 0.25),
        ("25%+",   0.25, 1.00),
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


# ---------------------------------------------------------------------------
# Prop model accuracy — settlement + summary
# ---------------------------------------------------------------------------

def check_and_settle_props(today: date) -> int:
    """
    Settle prop projections against ESPN box score actuals.

    Settles TWO windows each run:
      1. Yesterday's props  — always attempted (primary settlement)
      2. Today's props      — settled if games are already final in ESPN
                              (handles same-day settlement when the workflow
                              runs after games finish, e.g. a manual evening run)

    Appends new settled records to state/prop_history.json.
    Returns the total number of props newly settled this run.
    """
    from src.state.manager import load_state
    from src.data.prop_outcomes import check_prop_outcomes

    existing      = _load_prop_history()
    existing = [
        {**r, "prop_type": "Hits Over"} if r.get("prop_type") == "Hits Over (1+)" else r
        for r in existing
    ]
    existing_keys = {(r["date"], r["player"], r["prop_type"]) for r in existing}
    all_new: List[Dict] = []

    for settle_date in [today - timedelta(days=3), today - timedelta(days=2),
                        today - timedelta(days=1), today]:
        state = load_state(settle_date)
        if not state:
            continue
        props = state.get("props", [])
        if not props:
            continue

        settled = check_prop_outcomes(props, settle_date)
        if not settled:
            continue

        new = [
            r for r in settled
            if (r["date"], r["player"], r["prop_type"]) not in existing_keys
        ]
        all_new.extend(new)
        # Keep existing_keys current so today's records don't double-count yesterday's
        for r in new:
            existing_keys.add((r["date"], r["player"], r["prop_type"]))

    if all_new:
        _append_to_prop_history(existing + all_new)
        hits   = sum(1 for r in all_new if r["hit"])
        misses = len(all_new) - hits
        logger.info(
            f"Prop settlement: {len(all_new)} prop(s) settled — "
            f"{hits} hit / {misses} miss"
        )

    return len(all_new)


def load_prop_accuracy() -> Dict:
    """
    Returns a summary of prop model accuracy for the report:
      total, hit_rate, by_type, by_conf, recent (last 20)

    Also returns hist_* variants that exclude today's already-settled props.
    These are used as the JS baseline in data-hist-* attributes so that
    updatePropAccuracy() can add today's ESPN-confirmed results without
    double-counting props already written to history by the morning workflow.
    """
    records = _load_prop_history()
    # Normalise legacy "Hits Over (1+)" → "Hits Over" so they share one table row
    records = [
        {**r, "prop_type": "Hits Over"} if r.get("prop_type") == "Hits Over (1+)" else r
        for r in records
    ]
    if not records:
        return {}

    from datetime import datetime, timezone, timedelta as _td
    def _pacific_today() -> str:
        now_utc = datetime.now(timezone.utc)
        offset = -7 if 3 <= now_utc.month <= 10 else -8
        return (now_utc + _td(hours=offset)).date().isoformat()
    today_str     = _pacific_today()
    hist_records  = [r for r in records if r.get("date", "") < today_str]
    today_records = [r for r in records if r.get("date", "") == today_str]

    def _acc(subset: List[Dict]) -> Dict:
        total  = len(subset)
        hits   = sum(1 for r in subset if r.get("hit"))
        projs  = [r.get("model_line", 0) for r in subset]
        actuals = [r.get("actual_stat", 0) for r in subset]
        avg_proj   = round(sum(projs)   / total, 1) if total else None
        avg_actual = round(sum(actuals) / total, 1) if total else None
        avg_err    = round((sum(actuals) - sum(projs)) / total, 1) if total else None
        return {
            "total":      total,
            "hits":       hits,
            "misses":     total - hits,
            "hit_rate":   round(hits / total * 100, 1) if total else None,
            "avg_proj":   avg_proj,
            "avg_actual": avg_actual,
            "avg_err":    avg_err,   # positive = we under-projected (players outperformed)
        }

    # Discover all prop types present in history — no hardcoded list,
    # so any new prop type (Hits, HRs, Steals, Blocks, etc.) appears
    # automatically once it has at least one settled record.
    seen_types: list = []
    for r in records:
        pt = r.get("prop_type", "")
        if pt and pt not in seen_types:
            seen_types.append(pt)

    by_type: Dict[str, Dict] = {}
    hist_by_type: Dict[str, Dict] = {}
    for pt in seen_types:
        subset      = [r for r in records      if r.get("prop_type") == pt]
        hist_subset = [r for r in hist_records if r.get("prop_type") == pt]
        if subset:
            by_type[pt]      = _acc(subset)
            hist_by_type[pt] = _acc(hist_subset)

    by_conf: Dict[str, Dict] = {}
    hist_by_conf: Dict[str, Dict] = {}
    for conf in ("HIGH", "MEDIUM"):
        subset      = [r for r in records      if r.get("confidence") == conf]
        hist_subset = [r for r in hist_records if r.get("confidence") == conf]
        if subset:
            by_conf[conf]      = _acc(subset)
            hist_by_conf[conf] = _acc(hist_subset)

    # Break down by sport across ALL settled records (not just recent-20)
    seen_sports: list = []
    for r in records:
        sp = r.get("sport", "")
        if sp and sp not in seen_sports:
            seen_sports.append(sp)

    by_sport: Dict[str, Dict] = {}
    hist_by_sport: Dict[str, Dict] = {}
    for sp in seen_sports:
        subset      = [r for r in records      if r.get("sport") == sp]
        hist_subset = [r for r in hist_records if r.get("sport") == sp]
        if subset:
            by_sport[sp]      = _acc(subset)
            hist_by_sport[sp] = _acc(hist_subset)

    hist_all = _acc(hist_records)

    return {
        "total":         len(records),
        "all":           _acc(records),
        "by_type":       by_type,
        "by_conf":       by_conf,
        "by_sport":      by_sport,
        "recent":        records[-20:],   # last 20 for display in report
        # Before-today baselines for JS data-hist-* attributes (avoids double-count)
        "hist_total":    len(hist_records),
        "hist_all":      hist_all,
        "hist_by_type":  hist_by_type,
        "hist_by_conf":  hist_by_conf,
        "hist_by_sport": hist_by_sport,
        "today_total":   len(today_records),
    }


def build_prop_chart_data() -> Dict:
    """
    Pre-computes SVG coordinates for the prop hit-rate trend chart.
    Uses the same SVG canvas constants as build_chart_data().

    Returns:
      has_data         — False if fewer than 3 historical props
      wr_points        — SVG polyline points for rolling hit rate
      ref50_y          — y-coordinate of the 50% reference line
      wr_y_labels      — list of {y, val} for y-axis ticks
      x_labels         — list of {x, val} for x-axis date labels
      total            — number of hist records used
      hist_records_json — JSON string of {date, hit} for each hist record
                          (embedded in the page so JS can extend the chart live)
    """
    records = _load_prop_history()

    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    def _pac_today() -> str:
        now_utc = _dt.now(_tz.utc)
        offset = -7 if 3 <= now_utc.month <= 10 else -8
        return (now_utc + _td(hours=offset)).date().isoformat()

    today_str    = _pac_today()
    hist_records = [r for r in records if r.get("date", "") < today_str]

    empty = {"has_data": False, "hist_records_json": "[]"}
    if len(hist_records) < 3:
        return empty

    # ── Rolling hit rate (window=20, show from prop 3 onwards) ───────────────
    WINDOW   = 20
    MIN_SHOW = 3
    wr_series: List[Dict] = []
    for i in range(MIN_SHOW - 1, len(hist_records)):
        window = hist_records[max(0, i - WINDOW + 1): i + 1]
        hits   = sum(1 for r in window if r.get("hit"))
        rate   = hits / len(window) * 100
        wr_series.append({"i": i, "rate": round(rate, 1)})

    if not wr_series:
        return empty

    n          = len(hist_records)
    wr_coords  = [(_sx(p["i"], n), _sy(p["rate"], 0, 100)) for p in wr_series]
    wr_points  = _points_str(wr_coords)
    ref50_y    = round(_sy(50, 0, 100), 1)

    wr_y_labels = [
        {"y": round(_sy(100, 0, 100), 1), "val": "100%"},
        {"y": round(_sy(50,  0, 100), 1), "val":  "50%"},
        {"y": round(_sy(0,   0, 100), 1), "val":   "0%"},
    ]

    # X-axis: first, mid, last date labels
    x_labels = [{"x": round(_sx(0, n), 1), "val": hist_records[0].get("date", "")[5:]}]
    if n > 2:
        mid = n // 2
        x_labels.append({"x": round(_sx(mid, n), 1), "val": hist_records[mid].get("date", "")[5:]})
    x_labels.append({"x": round(_sx(n - 1, n), 1), "val": hist_records[-1].get("date", "")[5:]})

    # Serialize hist records for JS live update baseline (minimal fields)
    import json as _json
    hist_json = _json.dumps([
        {"date": r.get("date", ""), "hit": bool(r.get("hit"))}
        for r in hist_records
    ])

    return {
        "has_data":          True,
        "total":             n,
        "svg_w":             _SVG_W,
        "svg_h":             _SVG_H,
        "pad_l":             _PAD_L,
        "wr_points":         wr_points,
        "ref50_y":           ref50_y,
        "wr_y_labels":       wr_y_labels,
        "x_labels":          x_labels,
        "hist_records_json": hist_json,
    }


def _load_prop_history() -> List[Dict]:
    if not PROP_HISTORY_PATH.exists():
        return []
    try:
        with open(PROP_HISTORY_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load prop history: {e}")
        return []


def _append_to_prop_history(records: List[Dict]) -> None:
    PROP_HISTORY_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(PROP_HISTORY_PATH, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(
            f"Prop history updated → {PROP_HISTORY_PATH} ({len(records)} total records)"
        )
    except Exception as e:
        logger.error(f"Failed to write prop history: {e}")


# ---------------------------------------------------------------------------
# Watchlist settlement (NHL, IPL) — separate history, no PnL
# ---------------------------------------------------------------------------

def _load_watchlist_history() -> List[Dict]:
    if not WATCHLIST_HISTORY_PATH.exists():
        return []
    try:
        with open(WATCHLIST_HISTORY_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load watchlist history: {e}")
        return []


def _save_watchlist_history(records: List[Dict]) -> None:
    WATCHLIST_HISTORY_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(WATCHLIST_HISTORY_PATH, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(f"Watchlist history updated → {WATCHLIST_HISTORY_PATH} ({len(records)} total records)")
    except Exception as e:
        logger.error(f"Failed to write watchlist history: {e}")


def _parse_score_str(score_str) -> float:
    """
    Parse a score that may be an integer, float, or cricket-style 'runs/wickets' string.
    Returns the numeric part before any slash (e.g. '182/6' → 182.0).
    """
    s = str(score_str or "0").split("/")[0].strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def _fetch_watchlist_final_scores(sport: str, game_date: date) -> Dict:
    """
    Fetch completed game scores for watchlist sports (NHL, IPL) from ESPN.
    Identical structure to _fetch_espn_final_scores but uses WATCHLIST sport paths
    and _parse_score_str to handle cricket 'runs/wickets' format.
    """
    path = ESPN_WATCHLIST_PATHS.get(sport)
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
        if not comp.get("status", {}).get("type", {}).get("completed", False):
            continue

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_name  = home_comp.get("team", {}).get("displayName", "")
        away_name  = away_comp.get("team", {}).get("displayName", "")
        home_abbr  = home_comp.get("team", {}).get("abbreviation", "")
        away_abbr  = away_comp.get("team", {}).get("abbreviation", "")
        home_score = _parse_score_str(home_comp.get("score", 0))
        away_score = _parse_score_str(away_comp.get("score", 0))

        entry = {
            "home_name": home_name, "away_name": away_name,
            "home_abbr": home_abbr, "away_abbr": away_abbr,
            "home_score": home_score, "away_score": away_score,
        }
        for key in [home_name, away_name, home_abbr, away_abbr]:
            if key:
                scores[key.lower()] = entry

    logger.debug(f"ESPN {sport} {game_date}: {len(scores)//4 or len(scores)} game(s) found")
    return scores


def check_and_settle_watchlist(today: date) -> int:
    """
    Settle yesterday's NHL, WNBA, and MLS watchlist picks against ESPN final scores.
    NHL, WNBA, and MLS games always finish before the 9am PST morning run, so date-based
    settlement (look at yesterday's state) is correct.

    IPL (and other WATCHLIST_PENDING_SPORTS) are settled by
    settle_watchlist_pending() instead, which uses the rolling pending file.

    Appends new WON/LOST records to state/watchlist_history.json.
    Returns the number of newly settled picks.
    """
    yesterday = today - timedelta(days=1)

    from src.state.manager import load_state
    state = load_state(yesterday)
    if not state:
        logger.info(f"No state for {yesterday} — no NHL watchlist picks to settle")
        return 0

    existing = _load_watchlist_history()
    settled_keys = {(r["date"], r["sport"], r["pick"], r["game"]) for r in existing}
    new_records: List[Dict] = []

    # ── NHL ──────────────────────────────────────────────────────────────────
    nhl_picks = [p for p in (state.get("singles_display") or []) if p.get("sport") == "NHL"]
    nhl_scores = _fetch_watchlist_final_scores("NHL", yesterday) if nhl_picks else {}

    for pick in nhl_picks:
        key = (yesterday.isoformat(), "NHL", pick.get("pick", ""), pick.get("game", ""))
        if key in settled_keys:
            continue
        score_data = _find_game_score(nhl_scores, pick.get("home_team", ""), pick.get("away_team", ""))
        if not score_data:
            logger.debug(f"NHL watchlist: score not found for {pick.get('game')} on {yesterday}")
            continue
        result = _determine_outcome(
            pick.get("pick", ""), pick.get("bet_type", ""),
            pick.get("home_team", ""), pick.get("away_team", ""),
            score_data["home_score"], score_data["away_score"],
        )
        if result not in ("WON", "LOST"):
            continue
        new_records.append({
            "date":            yesterday.isoformat(),
            "sport":           "NHL",
            "game":            pick.get("game", ""),
            "pick":            pick.get("pick", ""),
            "bet_type":        pick.get("bet_type", "Moneyline"),
            "home_team":       pick.get("home_team", ""),
            "away_team":       pick.get("away_team", ""),
            "edge_pct":        pick.get("edge_pct", 0),
            "confidence":      pick.get("confidence", "MEDIUM"),
            "model_prob_pct":  pick.get("model_prob_pct", 0),
            "market_prob_pct": pick.get("market_prob_pct", 0),
            "result":          result,
        })
        logger.info(f"NHL watchlist settled: {pick.get('pick')} → {result}")

    # ── WNBA ─────────────────────────────────────────────────────────────────
    wnba_picks = [p for p in (state.get("wnba_display") or []) if p.get("sport") == "WNBA"]
    wnba_scores = _fetch_watchlist_final_scores("WNBA", yesterday) if wnba_picks else {}

    for pick in wnba_picks:
        key = (yesterday.isoformat(), "WNBA", pick.get("pick", ""), pick.get("game", ""))
        if key in settled_keys:
            continue
        score_data = _find_game_score(wnba_scores, pick.get("home_team", ""), pick.get("away_team", ""))
        if not score_data:
            logger.debug(f"WNBA watchlist: score not found for {pick.get('game')} on {yesterday}")
            continue
        result = _determine_outcome(
            pick.get("pick", ""), pick.get("bet_type", ""),
            pick.get("home_team", ""), pick.get("away_team", ""),
            score_data["home_score"], score_data["away_score"],
        )
        if result not in ("WON", "LOST"):
            continue
        new_records.append({
            "date":            yesterday.isoformat(),
            "sport":           "WNBA",
            "game":            pick.get("game", ""),
            "pick":            pick.get("pick", ""),
            "bet_type":        pick.get("bet_type", "Moneyline"),
            "home_team":       pick.get("home_team", ""),
            "away_team":       pick.get("away_team", ""),
            "edge_pct":        pick.get("edge_pct", 0),
            "confidence":      pick.get("confidence", "MEDIUM"),
            "model_prob_pct":  pick.get("model_prob_pct", 0),
            "market_prob_pct": pick.get("market_prob_pct", 0),
            "result":          result,
        })
        logger.info(f"WNBA watchlist settled: {pick.get('pick')} → {result}")

    # ── MLS ──────────────────────────────────────────────────────────────────
    mls_picks = [p for p in (state.get("mls_display") or []) if p.get("sport") == "MLS"]
    mls_scores = _fetch_watchlist_final_scores("MLS", yesterday) if mls_picks else {}

    for pick in mls_picks:
        key = (yesterday.isoformat(), "MLS", pick.get("pick", ""), pick.get("game", ""))
        if key in settled_keys:
            continue
        score_data = _find_game_score(mls_scores, pick.get("home_team", ""), pick.get("away_team", ""))
        if not score_data:
            logger.debug(f"MLS watchlist: score not found for {pick.get('game')} on {yesterday}")
            continue
        result = _determine_mls_outcome(
            pick.get("pick", ""), pick.get("bet_type", ""),
            pick.get("home_team", ""), pick.get("away_team", ""),
            score_data["home_score"], score_data["away_score"],
        )
        if result not in ("WON", "LOST"):
            continue
        new_records.append({
            "date":            yesterday.isoformat(),
            "sport":           "MLS",
            "game":            pick.get("game", ""),
            "pick":            pick.get("pick", ""),
            "bet_type":        pick.get("bet_type", "Moneyline"),
            "home_team":       pick.get("home_team", ""),
            "away_team":       pick.get("away_team", ""),
            "edge_pct":        pick.get("edge_pct", 0),
            "confidence":      pick.get("confidence", "MEDIUM"),
            "model_prob_pct":  pick.get("model_prob_pct", 0),
            "market_prob_pct": pick.get("market_prob_pct", 0),
            "result":          result,
        })
        logger.info(f"MLS watchlist settled: {pick.get('pick')} → {result}")

    if new_records:
        _save_watchlist_history(existing + new_records)

    return len(new_records)


# ---------------------------------------------------------------------------
# Rolling pending watchlist — for leagues whose games finish after the
# morning run (IPL starts at 7am PST, ends ~11am; run is at 9am PST).
# ---------------------------------------------------------------------------

def _load_watchlist_pending() -> List[Dict]:
    """Load the rolling pending pick list from state/watchlist_pending.json."""
    if not WATCHLIST_PENDING_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PENDING_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Failed to load watchlist pending: {e}")
        return []


def _save_watchlist_pending(records: List[Dict]) -> None:
    """Persist the rolling pending pick list."""
    WATCHLIST_PENDING_PATH.parent.mkdir(exist_ok=True)
    try:
        with open(WATCHLIST_PENDING_PATH, "w") as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(
            f"Watchlist pending saved → {WATCHLIST_PENDING_PATH} "
            f"({len(records)} unsettled pick(s))"
        )
    except Exception as e:
        logger.error(f"Failed to write watchlist pending: {e}")


def load_watchlist_pending() -> List[Dict]:
    """Public accessor — load the rolling pending list (called from main.py)."""
    return _load_watchlist_pending()


def save_watchlist_pending(records: List[Dict]) -> None:
    """Public accessor — persist the rolling pending list (called from main.py)."""
    _save_watchlist_pending(records)


def _settle_ipl_pick(pick: Dict, game_date: date) -> Optional[tuple]:
    """
    Determine WON/LOST for an IPL pending pick via Cricbuzz completed match data.
    Returns (result, match_summary) where result is "WON"/"LOST" and match_summary
    is the raw Cricbuzz status string (e.g. "DC won by 3 wickets").
    Returns None if the match is not yet in the completed list.
    ESPN cricket/ipl always returns 404, so this replaces that path for IPL.
    """
    try:
        from src.data.ipl_stats import get_ipl_completed_matches, normalize
    except Exception as e:
        logger.error(f"IPL settlement: could not import ipl_stats: {e}")
        return None

    home_team = pick.get("home_team", "")
    away_team = pick.get("away_team", "")
    our_pick  = pick.get("pick", "")

    try:
        completed = get_ipl_completed_matches(game_date)
    except Exception as e:
        logger.error(f"IPL settlement: Cricbuzz fetch failed: {e}")
        return None

    norm_home = normalize(home_team)
    norm_away = normalize(away_team)
    norm_pick = normalize(our_pick)

    for m in completed:
        # Guard: teams in this season can meet more than once.
        # Only consider matches whose UTC date matches game_date so we never
        # accidentally settle today's pick using a previous encounter.
        try:
            m_date = datetime.fromtimestamp(
                m.get("start_ms", 0) / 1000, tz=timezone.utc
            ).date()
        except Exception:
            m_date = None

        if m_date != game_date:
            continue

        t1 = normalize(m.get("team1", ""))
        t2 = normalize(m.get("team2", ""))
        if (t1 == norm_home and t2 == norm_away) or (t1 == norm_away and t2 == norm_home):
            winner = m.get("winner")
            if winner is None:
                return None  # match complete but no clean winner (tie/no result)
            result = "WON" if normalize(winner) == norm_pick else "LOST"
            return (result, m.get("match_summary", ""))

    return None  # match not yet in completed list


def load_watchlist_today_settled(sport: str, today: date) -> List[Dict]:
    """
    Return today's settled watchlist picks for a given sport.
    Used by main.py to include same-day settled results in the display.
    """
    records = _load_watchlist_history()
    return [r for r in records if r.get("sport") == sport and r.get("date") == today.isoformat()]


def settle_watchlist_pending(now_utc: datetime) -> int:
    """
    Attempt to settle any pending watchlist picks whose games should now be final.

    A pick is eligible for settlement when:
        commence_time + WATCHLIST_PENDING_SPORTS[sport] hours < now_utc

    Uses ESPN scoreboard (completed=True gate) to confirm the game is final.
    Also checks the calendar day after the game date in case a match ran past midnight.

    Settled picks are written to watchlist_history.json and removed from the
    pending list.  Picks whose games are still in progress are left untouched.

    Adding a new sport (e.g. EPL) requires only an entry in WATCHLIST_PENDING_SPORTS
    and ESPN_WATCHLIST_PATHS — no other code changes.

    Returns the count of picks newly settled this run.
    """
    # Self-heal runs unconditionally — before the early-return so it always
    # executes even when there are no pending picks to settle.
    existing = _load_watchlist_history()
    _seen_hkeys: dict = {}
    _cleaned: List[Dict] = []
    for _r in existing:
        _hk = (_r.get("date", ""), _r.get("sport", ""), _r.get("game", ""), _r.get("pick", ""))
        if _hk not in _seen_hkeys:
            _seen_hkeys[_hk] = len(_cleaned)
            _cleaned.append(_r)
        elif _r.get("result") == "LOST" and _cleaned[_seen_hkeys[_hk]].get("result") == "WON":
            _cleaned[_seen_hkeys[_hk]] = _r  # replace erroneous WON with correct LOST
    if len(_cleaned) < len(existing):
        logger.info(f"History self-heal: removed {len(existing) - len(_cleaned)} duplicate record(s)")
        _save_watchlist_history(_cleaned)
        existing = _cleaned

    pending = _load_watchlist_pending()
    if not pending:
        return 0

    settled_keys = {(r["date"], r["sport"], r["pick"], r["game"]) for r in existing}
    new_records: List[Dict] = []
    still_pending: List[Dict] = []

    for pick in pending:
        sport = pick.get("sport", "IPL")
        duration_hrs = WATCHLIST_PENDING_SPORTS.get(sport)
        if duration_hrs is None:
            # Sport not in pending system — leave it alone
            still_pending.append(pick)
            continue

        commence_str = pick.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
        except Exception:
            logger.warning(f"Pending pick has unparseable commence_time '{commence_str}' — keeping")
            still_pending.append(pick)
            continue

        expected_end = commence_dt + timedelta(hours=duration_hrs)
        if now_utc < expected_end:
            # Game not expected to be finished yet — keep pending
            still_pending.append(pick)
            continue

        # Game should be done — resolve result
        game_date = commence_dt.date()

        match_summary = ""
        if sport == "IPL":
            # ESPN cricket/ipl returns 404 — use Cricbuzz completed match data instead
            ipl_settled = _settle_ipl_pick(pick, game_date)
            if ipl_settled is None:
                logger.debug(
                    f"IPL pending: result not final for {pick.get('game')} "
                    f"(game_date={game_date}) — retaining"
                )
                still_pending.append(pick)
                continue
            result, match_summary = ipl_settled
        else:
            scores = _fetch_watchlist_final_scores(sport, game_date)
            if not scores:
                # Try the next calendar day (covers late-finishing / rain-delayed matches)
                scores = _fetch_watchlist_final_scores(sport, game_date + timedelta(days=1))

            score_data = _find_game_score(
                scores, pick.get("home_team", ""), pick.get("away_team", "")
            )
            if not score_data:
                logger.debug(
                    f"{sport} pending: score not final for {pick.get('game')} "
                    f"(game_date={game_date}) — retaining"
                )
                still_pending.append(pick)
                continue

            result = _determine_outcome(
                pick.get("pick", ""), pick.get("bet_type", ""),
                pick.get("home_team", ""), pick.get("away_team", ""),
                score_data["home_score"], score_data["away_score"],
            )
            if result not in ("WON", "LOST"):
                still_pending.append(pick)
                continue

        key = (game_date.isoformat(), sport, pick.get("pick", ""), pick.get("game", ""))
        if key in settled_keys:
            # Already in history (e.g. from a previous run) — just drop from pending
            logger.debug(f"{sport} pending: {pick.get('pick')} already settled — removing from pending")
            continue

        new_records.append({
            "date":            game_date.isoformat(),
            "sport":           sport,
            "game":            pick.get("game", ""),
            "pick":            pick.get("pick", ""),
            "bet_type":        pick.get("bet_type", "Moneyline"),
            "home_team":       pick.get("home_team", ""),
            "away_team":       pick.get("away_team", ""),
            "edge_pct":        pick.get("edge_pct", 0),
            "confidence":      pick.get("confidence", "MEDIUM"),
            "model_prob_pct":  pick.get("model_prob_pct", 0),
            "market_prob_pct": pick.get("market_prob_pct", 0),
            "result":          result,
            "match_summary":   match_summary,
        })
        settled_keys.add(key)
        logger.info(f"{sport} pending pick settled: {pick.get('pick')} → {result}")

    if new_records:
        _save_watchlist_history(existing + new_records)

    _save_watchlist_pending(still_pending)
    return len(new_records)


def load_watchlist_performance() -> Dict[str, Dict]:
    """
    Returns {sport: {won, lost, total, win_rate_pct}} for NHL and IPL.
    Used by generator.py to populate the watchlist tracking tiles.

    Deduplicates by (date, game) before counting so that a history file
    containing multiple records for the same game (e.g. from a buggy
    settlement followed by a manual correction) is counted correctly.
    When duplicates exist the LOST record takes precedence over WON.
    """
    records = _load_watchlist_history()
    result: Dict[str, Dict] = {}
    for sport in ("NHL", "IPL", "WNBA", "MLS"):
        subset = [r for r in records if r.get("sport") == sport and r.get("result") in ("WON", "LOST")]
        # Deduplicate: one entry per (date, game, pick) so that multiple
        # legitimate picks for the same game (ML + spread + total) are all
        # counted, while the exact same pick settled twice with different
        # results (e.g. WON from buggy run + LOST from correction) is
        # collapsed to one — LOST beats WON.
        deduped: Dict[tuple, str] = {}
        for r in subset:
            key = (r.get("date", ""), r.get("game", ""), r.get("pick", ""))
            if key not in deduped or r.get("result") == "LOST":
                deduped[key] = r.get("result", "")
        won  = sum(1 for v in deduped.values() if v == "WON")
        lost = sum(1 for v in deduped.values() if v == "LOST")
        result[sport] = {
            "won":          won,
            "lost":         lost,
            "total":        won + lost,
            "win_rate_pct": round(won / (won + lost) * 100, 1) if (won + lost) > 0 else None,
        }
    return result
