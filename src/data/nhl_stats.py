"""
Fetches NHL team stats via ESPN's public API.

Strategy:
  1. Standings endpoint → season points, goals-for/against for all 32 teams
  2. Scoreboard history → rest days (back-to-back detection)
  3. Recent schedule → last-14-day form estimate

ESPN NHL season year = the year the season ends (2024–25 season → 2025).
"""
import logging
import time
import requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_NHL      = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"
ESPN_NHL_V2   = "https://site.api.espn.com/apis/v2/sports/hockey/nhl"
ESPN_NHL_CORE = "https://sports.core.api.espn.com/v2/sports/hockey/nhl"

# Odds API name → ESPN displayName (only entries that differ)
_ODDS_TO_ESPN: Dict[str, str] = {
    "Montréal Canadiens": "Montreal Canadiens",
    "Montreal Canadiens":  "Montreal Canadiens",
}
_ESPN_TO_ODDS: Dict[str, str] = {v: k for k, v in _ODDS_TO_ESPN.items()}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName (key used in all ctx dicts)."""
    return _ODDS_TO_ESPN.get(name, name)


def _nhl_season() -> int:
    """
    ESPN NHL season year = the year the season ends.
    Oct 2025 – Jun 2026 → 2026.
    """
    today = date.today()
    # Season starting in October belongs to the following year on ESPN
    return today.year + 1 if today.month >= 10 else today.year


def _get(url: str, params: dict = None) -> Optional[dict]:
    """GET → JSON or None on any error."""
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN NHL GET failed [{url}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Season stats — standings endpoint (one call for all 32 teams)
# ---------------------------------------------------------------------------

def _fetch_all_team_stats() -> Dict[str, Dict]:
    """
    Returns {espn_display_name: stats_dict} for every NHL team from standings.

    ESPN's standings structure differs between regular season and playoffs:
      Regular season: conference → children (divisions) → standings.entries
      Playoffs:       conference → standings.entries  (no division layer)
    Both are handled by collecting entries from whichever layer has them.

    Stat field names also differ in playoff mode — avgGoalsFor/avgGoalsAgainst
    are absent; pointsFor/pointsAgainst/gamesPlayed must be used instead.
    """
    season = _nhl_season()
    data = _get(f"{ESPN_NHL_V2}/standings", params={"season": season})
    if not data:
        return {}

    # Collect all standing entries regardless of whether they sit under a
    # division layer (regular season) or directly on the conference (playoffs).
    all_entries = []
    for conf in data.get("children", []):
        divisions = conf.get("children", [])
        if divisions:
            # Regular season: conference → division → entries
            for div in divisions:
                all_entries.extend(div.get("standings", {}).get("entries", []))
        else:
            # Playoff mode: conference → entries (no division children)
            all_entries.extend(conf.get("standings", {}).get("entries", []))

    result: Dict[str, Dict] = {}
    for entry in all_entries:
        espn_name = entry.get("team", {}).get("displayName", "")
        if not espn_name:
            continue

        stat_map = {s["name"]: s.get("value", 0.0)
                    for s in entry.get("stats", []) if "name" in s}

        gp = float(stat_map.get("gamesPlayed", 0) or 0)

        # Goals per game: prefer per-game fields, fall back to totals/gamesPlayed.
        gpg_raw  = stat_map.get("avgGoalsFor") or stat_map.get("goalsFor")
        gapg_raw = stat_map.get("avgGoalsAgainst") or stat_map.get("goalsAgainst")
        if gpg_raw is None and gp > 0:
            gpg_raw  = stat_map.get("pointsFor", 0) / gp
        if gapg_raw is None and gp > 0:
            gapg_raw = stat_map.get("pointsAgainst", 0) / gp
        gpg  = float(gpg_raw  or 3.0)
        gapg = float(gapg_raw or 3.0)

        wins      = float(stat_map.get("wins",     0))
        losses    = float(stat_map.get("losses",   1))
        ot_losses = float(stat_map.get("otLosses", stat_map.get("otlosses", 0)))
        total     = wins + losses + ot_losses
        win_pct   = float(stat_map.get("winPercent",
                          wins / total if total > 0 else 0.5) or
                          (wins / total if total > 0 else 0.5))
        points    = float(stat_map.get("points", wins * 2))

        result[espn_name] = {
            "off_rtg":  gpg,           # goals/game scored
            "def_rtg":  gapg,          # goals/game allowed
            "net_rtg":  gpg - gapg,    # goal differential per game
            "gpg":      gpg,
            "gapg":     gapg,
            "win_pct":  win_pct,
            "wins":     int(wins),
            "losses":   int(losses),
            "ot_losses": int(ot_losses),
            "points":   int(points),
        }

    logger.info(f"NHL season stats: {len(result)} teams loaded")
    return result


# ---------------------------------------------------------------------------
# Recent form — last 14 days from scoreboard history
# ---------------------------------------------------------------------------

def _fetch_recent_form(team_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Estimates recent form from completed games in the last 14 days.
    Returns {display_name: {recent_net_rtg, recent_w_pct, recent_gpg, recent_gapg}}.
    Tracks goals scored and allowed separately so the totals model can blend them.
    """
    recent: Dict[str, Dict] = {}
    today = date.today()
    # Each entry: (goals_for, goals_against, won)
    team_scores: Dict[str, list] = {}

    for delta in range(1, 15):
        d = today - timedelta(days=delta)
        datestr = d.strftime("%Y%m%d")
        data = _get(f"{ESPN_NHL}/scoreboard", params={"dates": datestr})
        if not data:
            continue
        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if comp.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            teams = comp.get("competitors", [])
            if len(teams) != 2:
                continue
            try:
                scores = {t["team"]["displayName"]: int(t["score"]) for t in teams}
                names  = list(scores.keys())
                if len(names) != 2:
                    continue
                for name in names:
                    opp  = names[1] if name == names[0] else names[0]
                    gf   = scores[name]
                    ga   = scores[opp]
                    won  = gf > ga
                    team_scores.setdefault(name, []).append((gf, ga, won))
            except Exception:
                continue
        time.sleep(0.05)

    for name, entries in team_scores.items():
        if not entries:
            continue
        avg_gf  = sum(e[0] for e in entries) / len(entries)
        avg_ga  = sum(e[1] for e in entries) / len(entries)
        w_pct   = sum(1 for e in entries if e[2]) / len(entries)
        recent[name] = {
            "recent_net_rtg": avg_gf - avg_ga,
            "recent_w_pct":   w_pct,
            "recent_gpg":     avg_gf,
            "recent_gapg":    avg_ga,
        }

    for name, stats in team_map.items():
        if name not in recent:
            recent[name] = {
                "recent_net_rtg": stats.get("net_rtg", 0.0),
                "recent_w_pct":   stats.get("win_pct", 0.5),
                "recent_gpg":     stats.get("gpg", 3.0),
                "recent_gapg":    stats.get("gapg", 3.0),
            }

    return recent


# ---------------------------------------------------------------------------
# Rest days — days since each team's last game (NHL has many back-to-backs)
# ---------------------------------------------------------------------------

def _fetch_rest_days(team_names: List[str]) -> Dict[str, int]:
    """
    Returns {display_name: days_since_last_game}.
    Defaults to 2 (typical NHL cadence).
    Back-to-back = 1 day rest (game last night).
    """
    rest: Dict[str, int] = {}
    today = date.today()

    for delta in range(1, 10):   # look back up to 9 days
        if all(t in rest for t in team_names):
            break
        d = today - timedelta(days=delta)
        data = _get(f"{ESPN_NHL}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        if not data:
            continue
        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if comp.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            for t in comp.get("competitors", []):
                name = t.get("team", {}).get("displayName", "")
                if name and name not in rest:
                    rest[name] = delta

    for t in team_names:
        rest.setdefault(t, 2)

    return rest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_nhl_context(today: date, team_names: List[str] = None) -> Dict:
    """
    Returns full NHL context dict for edge_finder:
    {
        "season_stats":  {display_name: stats_dict},
        "recent_form":   {display_name: form_dict},
        "rest_days":     {display_name: int},
    }
    """
    team_names = team_names or []
    norm_names = [normalize(n) for n in team_names]

    season_stats = _fetch_all_team_stats()

    try:
        recent_form = _fetch_recent_form(season_stats)
    except Exception as e:
        logger.warning(f"NHL recent form fetch failed: {e}")
        recent_form = {n: {"recent_net_rtg": season_stats.get(n, {}).get("net_rtg", 0.0),
                            "recent_w_pct":  season_stats.get(n, {}).get("win_pct", 0.5)}
                       for n in norm_names}

    try:
        rest_days = _fetch_rest_days(norm_names)
    except Exception as e:
        logger.warning(f"NHL rest days fetch failed: {e}")
        rest_days = {n: 2 for n in norm_names}

    return {
        "season_stats": season_stats,
        "recent_form":  recent_form,
        "rest_days":    rest_days,
    }
