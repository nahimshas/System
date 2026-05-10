"""
Fetches IPL (Indian Premier League) team stats via ESPN Cricinfo's public API.

Strategy:
  1. Scoreboard history (last 30 days) → per-team win/loss record for the
     current IPL season, used as the primary team-strength signal.
  2. Rest days → days since each team's last completed match.

No traditional standings endpoint is used; IPL is a round-robin tournament
and recent form within the season is a better signal than overall record.

ESPN sport path: cricket/ipl
"""
import logging
import time
import requests
from datetime import date, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_IPL = "https://site.api.espn.com/apis/site/v2/sports/cricket/ipl"

# Odds API team name → ESPN Cricinfo displayName (only entries that differ)
_ODDS_TO_ESPN: Dict[str, str] = {
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "RCB":                         "Royal Challengers Bengaluru",
    "Kings XI Punjab":             "Punjab Kings",
    "Delhi Daredevils":            "Delhi Capitals",
}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName (key used in all ctx dicts)."""
    return _ODDS_TO_ESPN.get(name, name)


def _get(url: str, params: dict = None) -> Optional[dict]:
    """GET → JSON or None on any error."""
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN IPL GET failed [{url}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Recent form — last 30 days of scoreboard (covers a full IPL season window)
# ---------------------------------------------------------------------------

def _fetch_season_form() -> Dict[str, Dict]:
    """
    Returns {team_name: {wins, losses, total, win_pct, recent_win_pct}}
    by scanning the last 30 days of completed IPL matches.

    Two windows:
      - Season form:  all matches found (up to 30 days back)
      - Recent form:  last 7 days only (hot/cold streak signal)
    """
    today = date.today()
    team_season: Dict[str, list] = {}   # list of (margin, won) for all matches
    team_recent: Dict[str, list] = {}   # same but last 7 days only

    for delta in range(1, 31):
        d = today - timedelta(days=delta)
        datestr = d.strftime("%Y%m%d")
        data = _get(f"{ESPN_IPL}/scoreboard", params={"dates": datestr})
        if not data:
            continue

        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            # Only count completed matches
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            try:
                # ESPN cricket scores come as run totals (integers)
                scores = {}
                for c in competitors:
                    name = c.get("team", {}).get("displayName", "")
                    score_str = c.get("score", "0")
                    # Score may be "182" or "182/6" — take runs before the slash
                    runs = int(str(score_str).split("/")[0] or 0)
                    if name:
                        scores[name] = runs

                if len(scores) != 2:
                    continue

                names = list(scores.keys())
                margin = scores[names[0]] - scores[names[1]]
                for i, name in enumerate(names):
                    won = (i == 0 and margin > 0) or (i == 1 and margin < 0)
                    signed_margin = margin if i == 0 else -margin
                    team_season.setdefault(name, []).append((signed_margin, won))
                    if delta <= 7:
                        team_recent.setdefault(name, []).append((signed_margin, won))
            except Exception:
                continue

        time.sleep(0.05)

    result: Dict[str, Dict] = {}
    all_teams = set(team_season) | set(team_recent)
    for name in all_teams:
        season_entries = team_season.get(name, [])
        recent_entries = team_recent.get(name, [])

        s_total = len(season_entries)
        s_wins  = sum(1 for _, w in season_entries if w)
        s_wpct  = s_wins / s_total if s_total else 0.5

        r_total = len(recent_entries)
        r_wins  = sum(1 for _, w in recent_entries if w)
        r_wpct  = r_wins / r_total if r_total else s_wpct  # fall back to season if no recent

        result[name] = {
            "win_pct":        s_wpct,
            "wins":           s_wins,
            "losses":         s_total - s_wins,
            "total":          s_total,
            "recent_win_pct": r_wpct,
            "recent_total":   r_total,
        }

    logger.info(f"IPL form: {len(result)} teams from last 30 days of scoreboard")
    return result


# ---------------------------------------------------------------------------
# Rest days — days since each team's last completed match
# ---------------------------------------------------------------------------

def _fetch_rest_days(team_names: List[str]) -> Dict[str, int]:
    """
    Returns {team_name: days_since_last_match}.
    Default is 3 (typical IPL schedule has 2-4 days between matches).
    """
    rest: Dict[str, int] = {}
    today = date.today()

    for delta in range(1, 15):
        if all(t in rest for t in team_names):
            break
        d = today - timedelta(days=delta)
        data = _get(f"{ESPN_IPL}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        if not data:
            continue
        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue
            for c in comp.get("competitors", []):
                name = c.get("team", {}).get("displayName", "")
                if name and name not in rest:
                    rest[name] = delta

    for t in team_names:
        rest.setdefault(t, 3)

    return rest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_ipl_context(today: date, team_names: List[str] = None) -> Dict:
    """
    Returns full IPL context dict for edge_finder:
    {
        "season_form":  {display_name: {win_pct, wins, losses, total,
                                        recent_win_pct, recent_total}},
        "rest_days":    {display_name: int},
    }
    """
    team_names = team_names or []
    norm_names = [normalize(n) for n in team_names]

    try:
        season_form = _fetch_season_form()
    except Exception as e:
        logger.error(f"IPL season form fetch failed: {e}")
        season_form = {}

    try:
        rest_days = _fetch_rest_days(norm_names)
    except Exception as e:
        logger.warning(f"IPL rest days fetch failed: {e}")
        rest_days = {n: 3 for n in norm_names}

    return {
        "season_form": season_form,
        "rest_days":   rest_days,
    }
