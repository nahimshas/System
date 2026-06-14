"""Fetches injury reports from ESPN's unofficial API."""
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)

ESPN_NBA_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_MLB_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"
ESPN_NFL_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
ESPN_NHL_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/injuries"


def _fetch(url: str) -> List[Dict]:
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json().get("injuries", [])
    except Exception as e:
        logger.error(f"Injury fetch error ({url}): {e}")
        return []


def _parse_injuries(raw: List[Dict]) -> Dict[str, List[Dict]]:
    """Returns {team_name: [{player, status, detail}]}"""
    team_injuries: Dict[str, List[Dict]] = {}
    for entry in raw:
        team = entry.get("displayName", entry.get("team", {}).get("displayName", "Unknown"))
        for athlete in entry.get("injuries", []):
            athlete_info = athlete.get("athlete", {})
            status = athlete.get("status", "")
            detail = athlete.get("shortComment", athlete.get("longComment", ""))
            position = athlete_info.get("position", {}).get("abbreviation", "")
            # Only flag significant statuses
            if status.lower() in ("out", "doubtful", "questionable", "day-to-day"):
                team_injuries.setdefault(team, []).append({
                    "player": athlete_info.get("displayName", "Unknown"),
                    "position": position,
                    "status": status,
                    "detail": detail,
                })
    return team_injuries


def get_nba_injuries() -> Dict[str, List[Dict]]:
    return _parse_injuries(_fetch(ESPN_NBA_INJURIES))


def get_mlb_injuries() -> Dict[str, List[Dict]]:
    return _parse_injuries(_fetch(ESPN_MLB_INJURIES))


def get_nfl_injuries() -> Dict[str, List[Dict]]:
    return _parse_injuries(_fetch(ESPN_NFL_INJURIES))


def get_nhl_injuries() -> Dict[str, List[Dict]]:
    return _parse_injuries(_fetch(ESPN_NHL_INJURIES))


# Impact weights by position — how much a player's absence shifts win probability
# These are rough estimates; the system uses them as directional adjustments
NBA_POSITION_IMPACT = {
    "PG": 0.025, "SG": 0.020, "SF": 0.020, "PF": 0.018, "C": 0.018,
    "G": 0.022, "F": 0.019, "G-F": 0.020, "F-C": 0.018,
}
MLB_POSITION_IMPACT = {
    "SP": 0.040,   # Starting pitcher out = biggest swing
    "RP": 0.010,
    "C": 0.012, "1B": 0.010, "2B": 0.012, "3B": 0.012,
    "SS": 0.015, "LF": 0.010, "CF": 0.015, "RF": 0.010, "DH": 0.012,
}
NFL_POSITION_IMPACT = {
    "QB": 0.060,   # QB out = largest single-player swing in team sports
    "WR": 0.015, "RB": 0.015, "TE": 0.015,
    "OT": 0.012, "OG": 0.010, "C": 0.010,
    "DE": 0.018, "DT": 0.012, "LB": 0.015, "CB": 0.018, "S": 0.015,
}
NHL_POSITION_IMPACT = {
    "G": 0.045,    # Starting goalie out = massive impact
    "C": 0.020, "LW": 0.018, "RW": 0.018, "D": 0.015,
    "F": 0.018,    # generic forward
}

STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.70, "questionable": 0.45, "day-to-day": 0.20}
# day-to-day: player more likely to play than not (~20% chance they sit)
# questionable: genuinely uncertain (~45% chance they sit)
# doubtful: more likely out than not (~70% chance they sit)

_SPORT_POSITION_MAP = {
    "nba": NBA_POSITION_IMPACT,
    "mlb": MLB_POSITION_IMPACT,
    "nfl": NFL_POSITION_IMPACT,
    "nhl": NHL_POSITION_IMPACT,
}


# Cap on total injury drag. Raised 0.06 → 0.10 to let value-weighted star
# injuries register more than the old flat position model allowed.
INJURY_DRAG_CAP = 0.10


def injury_adjustment(team: str, injuries: Dict[str, List[Dict]], sport: str) -> float:
    """
    Returns negative win probability adjustment for a team based on injuries.

    Each injury is weighted by position × status × value_mult, where value_mult
    (stamped onto the injury dict by the sport's value-enrichment step; defaults
    to 1.0 when absent) scales by the injured player's importance — so an Aaron
    Judge injury docks far more than a bench player at the same position. Sports
    not yet value-enriched fall through with value_mult = 1.0 (prior behaviour).
    """
    position_map = _SPORT_POSITION_MAP.get(sport, NBA_POSITION_IMPACT)
    adjustment = 0.0
    for inj in injuries.get(team, []):
        pos = inj.get("position", "")
        status = inj.get("status", "questionable").lower()
        impact = position_map.get(pos, 0.010)
        weight = STATUS_WEIGHT.get(status, 0.35)
        value_mult = inj.get("value_mult", 1.0)   # 1.0 = position-only (unenriched)
        adjustment += impact * weight * value_mult
    return min(adjustment, INJURY_DRAG_CAP)
