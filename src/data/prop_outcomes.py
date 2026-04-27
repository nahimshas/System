"""
Fetch actual player stats from ESPN box scores to settle prop model accuracy.
Used by outcome_checker.check_and_settle_props() — called once per daily run
to close out yesterday's prop projections as HIT or MISS vs model line.
"""
import logging
import requests
from datetime import date
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Map prop_type strings → ESPN NBA box score stat key
NBA_STAT_KEY = {
    "Points Over":   "PTS",
    "Rebounds Over": "REB",
    "Assists Over":  "AST",
}

# Possible ESPN key names for pitcher strikeouts (varies by API version)
MLB_K_KEYS = ["K", "SO", "strikeouts", "Strikeouts"]


# ---------------------------------------------------------------------------
# ESPN helpers
# ---------------------------------------------------------------------------

def _get_scoreboard(sport_path: str, game_date: date) -> List[Dict]:
    url = f"{ESPN_BASE}/{sport_path}/scoreboard"
    try:
        r = requests.get(url, params={"dates": game_date.strftime("%Y%m%d")}, timeout=15)
        r.raise_for_status()
        return r.json().get("events", [])
    except Exception as e:
        logger.error(f"ESPN scoreboard fetch failed ({sport_path}, {game_date}): {e}")
        return []


def _get_summary(sport_path: str, event_id: str) -> Dict:
    url = f"{ESPN_BASE}/{sport_path}/summary"
    try:
        r = requests.get(url, params={"event": event_id}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN summary fetch failed (event {event_id}): {e}")
        return {}


def _is_completed(summary: Dict) -> bool:
    """Return True only if the game is fully final."""
    try:
        comp = summary.get("header", {}).get("competitions", [{}])[0]
        return comp.get("status", {}).get("type", {}).get("completed", False)
    except (IndexError, AttributeError):
        return False


def _name_match(a: str, b: str) -> bool:
    """
    Flexible name match — handles full names, partial first-initial abbreviations,
    and last-name-only comparisons.  Case-insensitive.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return True
    a_parts, b_parts = a.split(), b.split()
    # Last name match
    if a_parts and b_parts and a_parts[-1] == b_parts[-1]:
        return True
    # Substring: one name is contained in the other
    if a in b or b in a:
        return True
    return False


def _find_event_id(events: List[Dict], team: str, opponent: str) -> Optional[str]:
    """Return the ESPN event ID for the game between team and opponent."""
    team_l = team.lower()
    opp_l  = opponent.lower()
    for event in events:
        comps = event.get("competitions", [{}])
        if not comps:
            continue
        competitors = comps[0].get("competitors", [])
        all_names = []
        for c in competitors:
            t = c.get("team", {})
            all_names += [
                t.get("displayName", "").lower(),
                t.get("shortDisplayName", "").lower(),
                t.get("abbreviation", "").lower(),
                t.get("name", "").lower(),
            ]
        team_found = any(team_l in n or n in team_l for n in all_names if n)
        opp_found  = any(opp_l  in n or n in opp_l  for n in all_names if n)
        if team_found and opp_found:
            return event.get("id")
    return None


# ---------------------------------------------------------------------------
# Stat extraction
# ---------------------------------------------------------------------------

def _get_nba_player_stat(summary: Dict, player_name: str, stat_key: str) -> Optional[float]:
    """
    Walk the NBA box score and return a single player's stat value.
    stat_key: 'PTS', 'REB', 'AST'
    """
    for team_data in summary.get("boxscore", {}).get("players", []):
        for stat_group in team_data.get("statistics", []):
            keys = stat_group.get("keys", [])
            if stat_key not in keys:
                continue
            idx = keys.index(stat_key)
            for athlete_entry in stat_group.get("athletes", []):
                name = athlete_entry.get("athlete", {}).get("displayName", "")
                if _name_match(name, player_name):
                    stats = athlete_entry.get("stats", [])
                    if idx < len(stats):
                        try:
                            return float(stats[idx])
                        except (ValueError, TypeError):
                            pass
    return None


def _get_mlb_pitcher_ks(summary: Dict, pitcher_name: str) -> Optional[float]:
    """
    Walk the MLB box score and return a pitcher's strikeout total.
    Handles varying ESPN key names for the K stat.
    """
    for team_data in summary.get("boxscore", {}).get("players", []):
        for stat_group in team_data.get("statistics", []):
            if "pitch" not in stat_group.get("name", "").lower():
                continue
            keys = stat_group.get("keys", [])
            k_key = next((k for k in MLB_K_KEYS if k in keys), None)
            if k_key is None:
                continue
            idx = keys.index(k_key)
            for athlete_entry in stat_group.get("athletes", []):
                name = athlete_entry.get("athlete", {}).get("displayName", "")
                if _name_match(name, pitcher_name):
                    stats = athlete_entry.get("stats", [])
                    if idx < len(stats):
                        try:
                            return float(stats[idx])
                        except (ValueError, TypeError):
                            pass
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def check_prop_outcomes(props: List[Dict], game_date: date) -> List[Dict]:
    """
    For each prop, fetch the ESPN box score and compare actual stat to model_line.
    Skips team-level placeholder props (e.g. "LAD top-order batters").
    Returns a list of settled prop records (only fully completed games).
    """
    nba_events: Optional[List[Dict]] = None
    mlb_events: Optional[List[Dict]] = None
    summary_cache: Dict[str, Dict] = {}

    results: List[Dict] = []

    for prop in props:
        sport      = prop.get("sport", "").upper()
        player     = prop.get("player", "")
        team       = prop.get("team", "")
        opponent   = prop.get("opponent", "")
        prop_type  = prop.get("prop_type", "")
        model_line = float(prop.get("model_line", 0.0))
        confidence = prop.get("confidence", "MEDIUM")

        # Skip team-level placeholder props (Hits Over 1+ use team string, not a player name)
        if not player or any(w in player.lower() for w in ("batters", "lineup", "top-order")):
            continue

        actual_stat: Optional[float] = None

        # ── NBA ──────────────────────────────────────────────────────────────
        if sport == "NBA":
            stat_key = NBA_STAT_KEY.get(prop_type)
            if not stat_key:
                continue
            if nba_events is None:
                nba_events = _get_scoreboard("basketball/nba", game_date)
            event_id = _find_event_id(nba_events, team, opponent)
            if not event_id:
                logger.debug(f"No NBA event found: {team} vs {opponent} on {game_date}")
                continue
            if event_id not in summary_cache:
                summary_cache[event_id] = _get_summary("basketball/nba", event_id)
            summary = summary_cache[event_id]
            if not _is_completed(summary):
                continue
            actual_stat = _get_nba_player_stat(summary, player, stat_key)

        # ── MLB ───────────────────────────────────────────────────────────────
        elif sport == "MLB":
            if prop_type != "Strikeouts Over":
                continue  # Hits Over 1+ uses team placeholder — not individually settable
            if mlb_events is None:
                mlb_events = _get_scoreboard("baseball/mlb", game_date)
            event_id = _find_event_id(mlb_events, team, opponent)
            if not event_id:
                logger.debug(f"No MLB event found: {team} vs {opponent} on {game_date}")
                continue
            if event_id not in summary_cache:
                summary_cache[event_id] = _get_summary("baseball/mlb", event_id)
            summary = summary_cache[event_id]
            if not _is_completed(summary):
                continue
            actual_stat = _get_mlb_pitcher_ks(summary, player)

        if actual_stat is None:
            logger.debug(f"Stat not found in box score: {player} ({prop_type})")
            continue

        hit = actual_stat > model_line

        record = {
            "date":        game_date.isoformat(),
            "sport":       sport,
            "player":      player,
            "team":        team,
            "opponent":    opponent,
            "prop_type":   prop_type,
            "model_line":  model_line,
            "actual_stat": actual_stat,
            "hit":         hit,
            "confidence":  confidence,
        }
        results.append(record)
        logger.info(
            f"Prop settled: {player} {prop_type} — "
            f"Model {model_line} | Actual {actual_stat} | "
            f"{'✅ HIT' if hit else '❌ MISS'}"
        )

    return results
