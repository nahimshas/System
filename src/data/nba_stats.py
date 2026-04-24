"""
Fetches NBA team stats and player leaders via ESPN's public API.
ESPN (site.api.espn.com) works from any IP including GitHub Actions —
no bot protection like stats.nba.com.

Key design:
  - All context dicts are keyed by ESPN display names (the value normalize() returns).
  - edge_finder already calls normalize() before looking up context.
  - props_analyzer must also call normalize() — updated to do so.
"""
import logging
import time
import requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_NBA    = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NBA_V2 = "https://site.api.espn.com/apis/v2/sports/basketball/nba"

# Odds API name  →  ESPN displayName  (only entries that differ)
_ODDS_TO_ESPN = {
    "Los Angeles Clippers": "LA Clippers",
}
# Reverse map for building season_stats keys from ESPN standings
_ESPN_TO_ODDS = {v: k for k, v in _ODDS_TO_ESPN.items()}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName."""
    return _ODDS_TO_ESPN.get(name, name)


def _espn_season() -> int:
    """ESPN season year = ending year of the season. April 2026 → 2026."""
    today = date.today()
    return today.year if today.month < 10 else today.year + 1


def _get(url: str, params: dict = None) -> Optional[dict]:
    """GET → JSON or None on any error."""
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN GET failed [{url}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Season stats — ONE call for all 30 teams via the standings endpoint
# ---------------------------------------------------------------------------

def _fetch_all_team_stats() -> Dict[str, Dict]:
    """
    Returns {espn_display_name: stats_dict} for every NBA team.
    Uses the standings endpoint which includes PPG / OPPG for all teams at once.
    """
    season = _espn_season()
    data = _get(f"{ESPN_NBA_V2}/standings", params={"season": season})
    if not data:
        return {}

    result: Dict[str, Dict] = {}

    # Standings structure: data["children"] → conferences → standings.entries
    for conf in data.get("children", []):
        entries = conf.get("standings", {}).get("entries", [])
        for entry in entries:
            espn_name = entry.get("team", {}).get("displayName", "")
            if not espn_name:
                continue

            stat_map = {s["name"]: s.get("value", 0.0)
                        for s in entry.get("stats", []) if "name" in s}

            # Points per game — ESPN may use several possible keys
            ppg  = float(stat_map.get("avgPointsFor",
                         stat_map.get("pointsFor",
                         stat_map.get("avgPoints", 110.0))) or 110.0)
            oppg = float(stat_map.get("avgPointsAgainst",
                         stat_map.get("pointsAgainst",
                         stat_map.get("avgPointsAllowed", 110.0))) or 110.0)
            wins  = float(stat_map.get("wins", 0))
            losses = float(stat_map.get("losses", 1))
            total = wins + losses
            win_pct = float(stat_map.get("winPercent",
                            wins / total if total > 0 else 0.5))

            # Store under ESPN display name so normalize() lookups work
            result[espn_name] = {
                "off_rtg":  ppg,           # PPG  ≈ offensive rating
                "def_rtg":  oppg,          # OPPG ≈ defensive rating
                "net_rtg":  ppg - oppg,    # point differential
                "pace":     100.0,         # ESPN doesn't expose pace; league avg
                "ppg":      ppg,
                "oppg":     oppg,
                "win_pct":  win_pct,
                "wins":     int(wins),
                "losses":   int(losses),
            }

    logger.info(f"ESPN standings: {len(result)} teams loaded (season {season})")
    return result


# ---------------------------------------------------------------------------
# Team ID map — needed for per-team detail calls
# ---------------------------------------------------------------------------

def _fetch_team_id_map() -> Dict[str, str]:
    """Returns {espn_display_name: espn_team_id} for all 30 NBA teams."""
    data = _get(f"{ESPN_NBA}/teams", params={"limit": 32})
    if not data:
        return {}

    result: Dict[str, str] = {}
    for sport in data.get("sports", []):
        for league in sport.get("leagues", []):
            for item in league.get("teams", []):
                t = item.get("team", {})
                name = t.get("displayName", "")
                tid  = t.get("id", "")
                if name and tid:
                    result[name] = tid
    return result


# ---------------------------------------------------------------------------
# Statistical leaders — top scorer / rebounder / assist man per team
# ---------------------------------------------------------------------------

def _fetch_team_leaders(team_id: str) -> Dict[str, Dict]:
    """
    Returns {category: {name, value, display}} for the team's top players.
    Categories: 'points', 'rebounds', 'assists', 'blocks', 'steals'
    Pulled from the team detail endpoint which embeds leaders.
    """
    data = _get(f"{ESPN_NBA}/teams/{team_id}")
    if not data:
        return {}

    leaders_raw = data.get("team", {}).get("leaders", [])
    result: Dict[str, Dict] = {}
    for cat in leaders_raw:
        cat_name = cat.get("name", "")
        top = cat.get("leaders", [])
        if not top:
            continue
        first = top[0]
        athlete = first.get("athlete", {})
        player_name = athlete.get("displayName") or athlete.get("fullName", "")
        if player_name:
            result[cat_name] = {
                "name":    player_name,
                "value":   float(first.get("value", 0)),
                "display": first.get("displayValue", ""),
            }
    return result


# ---------------------------------------------------------------------------
# Recent form — last 14 days from team schedule
# ---------------------------------------------------------------------------

def _fetch_recent_form(team_id: str, today: date, days: int = 14) -> Dict:
    """
    Fetches completed games in the last `days` days and returns:
      recent_net_rtg, recent_off_rtg, recent_def_rtg, recent_w_pct, last_game_date
    """
    season = _espn_season()
    data = _get(f"{ESPN_NBA}/teams/{team_id}/schedule", params={"season": season})
    if not data:
        return {}

    cutoff = today - timedelta(days=days)
    wins = losses = 0
    pts_for = pts_against = 0.0
    game_count = 0
    last_game_date: Optional[date] = None

    for event in data.get("events", []):
        raw_date = event.get("date", "")
        try:
            game_date = datetime.fromisoformat(raw_date.replace("Z", "+00:00")).date()
        except Exception:
            continue

        if game_date >= today or game_date < cutoff:
            continue

        if last_game_date is None or game_date > last_game_date:
            last_game_date = game_date

        for comp in event.get("competitions", []):
            competitors = comp.get("competitors", [])
            this = next((c for c in competitors if c.get("team", {}).get("id") == team_id), None)
            opp  = next((c for c in competitors if c.get("team", {}).get("id") != team_id), None)
            if not this:
                continue

            def _score(raw) -> float:
                if isinstance(raw, dict):
                    v = raw.get("value", raw.get("displayValue", 0))
                    try:
                        return float(v or 0)
                    except Exception:
                        return 0.0
                try:
                    return float(raw or 0)
                except Exception:
                    return 0.0

            s  = _score(this.get("score", 0))
            os = _score(opp.get("score", 0)) if opp else 0.0

            if s > 0:
                if this.get("winner", False):
                    wins += 1
                else:
                    losses += 1
                pts_for     += s
                pts_against += os
                game_count  += 1

    if game_count == 0:
        return {"last_game_date": last_game_date}

    rpg  = pts_for  / game_count
    ropg = pts_against / game_count
    return {
        "recent_net_rtg": rpg - ropg,
        "recent_off_rtg": rpg,
        "recent_def_rtg": ropg,
        "recent_w_pct":   wins / game_count,
        "recent_ppg":     rpg,
        "recent_oppg":    ropg,
        "last_game_date": last_game_date,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_nba_context(today: date, team_names: List[str] = None) -> Dict:
    """
    Fetch NBA context from ESPN.

    team_names: Odds API team names playing today.
      If provided → also fetches per-team leaders + recent form (more API calls,
      but gives props real player names and accurate rest-day data).
      If None → season stats only (no per-team detail calls).

    All returned dicts are keyed by ESPN display names (normalize() output).
    """
    logger.info("Fetching NBA stats from ESPN...")

    # ── Season stats for all 30 teams (1 call) ───────────────────────────────
    season_stats = _fetch_all_team_stats()
    if not season_stats:
        logger.error("ESPN standings unavailable — NBA model will have no stats")
        return {"season_stats": {}, "recent_form": {}, "rest_days": {}, "team_leaders": {}}

    recent_form:  Dict[str, Dict] = {}
    rest_days:    Dict[str, int]  = {}
    team_leaders: Dict[str, Dict] = {}

    # ── Per-team detail: leaders + recent form ───────────────────────────────
    if team_names:
        id_map = _fetch_team_id_map()   # {espn_display_name: team_id}

        for raw_name in team_names:
            espn_name = normalize(raw_name)   # Odds API → ESPN display name

            # Find ESPN team ID
            team_id = id_map.get(espn_name)
            if not team_id:
                # Fuzzy fallback
                for k, v in id_map.items():
                    if espn_name.lower() in k.lower() or k.lower() in espn_name.lower():
                        team_id = v
                        break
            if not team_id:
                logger.warning(f"No ESPN team ID found for: {raw_name}")
                rest_days[espn_name] = 1
                continue

            # Leaders for props
            leaders = _fetch_team_leaders(team_id)
            if leaders:
                team_leaders[espn_name] = leaders
            time.sleep(0.25)

            # Recent form + rest days
            recent = _fetch_recent_form(team_id, today)
            last_date = recent.get("last_game_date")
            rest_days[espn_name] = (
                max(0, (today - last_date).days - 1) if last_date else 1
            )
            if recent.get("recent_ppg"):
                recent_form[espn_name] = {
                    "recent_net_rtg": recent["recent_net_rtg"],
                    "recent_off_rtg": recent["recent_off_rtg"],
                    "recent_def_rtg": recent["recent_def_rtg"],
                    "recent_w_pct":   recent["recent_w_pct"],
                }
            time.sleep(0.25)

    logger.info(
        f"NBA context ready — season stats: {len(season_stats)} teams | "
        f"recent form: {len(recent_form)} | leaders: {len(team_leaders)}"
    )
    return {
        "season_stats": season_stats,
        "recent_form":  recent_form,
        "rest_days":    rest_days,
        "team_leaders": team_leaders,
    }
