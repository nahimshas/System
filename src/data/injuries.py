"""Fetches injury reports from ESPN's unofficial API."""
import logging
import requests
from typing import Dict, List

logger = logging.getLogger(__name__)

ESPN_NBA_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN_MLB_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries"


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
        team = entry.get("team", {}).get("displayName", "Unknown")
        for athlete in entry.get("injuries", []):
            athlete_info = athlete.get("athlete", {})
            status = athlete.get("status", "")
            detail = athlete.get("shortComment", athlete.get("longComment", ""))
            position = athlete_info.get("position", {}).get("abbreviation", "")
            # Only flag significant statuses
            if status.lower() in ("out", "doubtful", "questionable"):
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

STATUS_WEIGHT = {"out": 1.0, "doubtful": 0.75, "questionable": 0.35}


def injury_adjustment(team: str, injuries: Dict[str, List[Dict]], sport: str) -> float:
    """Returns negative win probability adjustment for a team based on injuries."""
    position_map = NBA_POSITION_IMPACT if sport == "nba" else MLB_POSITION_IMPACT
    adjustment = 0.0
    for inj in injuries.get(team, []):
        pos = inj.get("position", "")
        status = inj.get("status", "questionable").lower()
        impact = position_map.get(pos, 0.010)
        weight = STATUS_WEIGHT.get(status, 0.35)
        adjustment += impact * weight
    return min(adjustment, 0.06)   # cap at 6% total injury drag (market prices injuries quickly)
