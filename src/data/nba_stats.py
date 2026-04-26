"""
Fetches NBA team stats and player leaders via ESPN's public API.
ESPN (site.api.espn.com) is unblocked from any IP including GitHub Actions.

Leader-fetch strategy (most reliable first):
  1. Scoreboard  — today's games embed season-to-date leaders per team (1 call).
  2. Team detail — /teams/{id} sometimes embeds leaders in the team object.
  3. Statistics  — /teams/{id}/statistics has category leaders.
  Any of these that returns player names wins; the rest are skipped.
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
_ODDS_TO_ESPN: Dict[str, str] = {
    "Los Angeles Clippers": "LA Clippers",
}
_ESPN_TO_ODDS: Dict[str, str] = {v: k for k, v in _ODDS_TO_ESPN.items()}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName (key used in all ctx dicts)."""
    return _ODDS_TO_ESPN.get(name, name)


def _espn_season() -> int:
    """ESPN season year = ending year of the season.  April 2026 → 2026."""
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
    Keys match what normalize() returns so edge_finder lookups work directly.
    """
    season = _espn_season()
    data = _get(f"{ESPN_NBA_V2}/standings", params={"season": season})
    if not data:
        return {}

    result: Dict[str, Dict] = {}
    for conf in data.get("children", []):
        for entry in conf.get("standings", {}).get("entries", []):
            espn_name = entry.get("team", {}).get("displayName", "")
            if not espn_name:
                continue

            stat_map = {s["name"]: s.get("value", 0.0)
                        for s in entry.get("stats", []) if "name" in s}

            ppg  = float(stat_map.get("avgPointsFor",
                         stat_map.get("pointsFor",
                         stat_map.get("avgPoints", 110.0))) or 110.0)
            oppg = float(stat_map.get("avgPointsAgainst",
                         stat_map.get("pointsAgainst",
                         stat_map.get("avgPointsAllowed", 110.0))) or 110.0)
            wins   = float(stat_map.get("wins",   0))
            losses = float(stat_map.get("losses", 1))
            total  = wins + losses
            win_pct = float(stat_map.get("winPercent",
                            wins / total if total > 0 else 0.5))

            result[espn_name] = {
                "off_rtg": ppg,
                "def_rtg": oppg,
                "net_rtg": ppg - oppg,
                "pace":    100.0,   # ESPN doesn't expose pace; use league avg
                "ppg":     ppg,
                "oppg":    oppg,
                "win_pct": win_pct,
                "wins":    int(wins),
                "losses":  int(losses),
            }

    logger.info(f"ESPN standings: {len(result)} teams (season {season})")
    return result


# ---------------------------------------------------------------------------
# Leaders helpers — parse ESPN's standard leader-category structure
# ---------------------------------------------------------------------------

def _parse_leader_categories(categories: list) -> Dict[str, Dict]:
    """
    Parse a list of ESPN leader-category dicts into our standard format.
    Handles both 'points'/'rebounds'/'assists' name styles and
    'pointsPerGame'/'reboundsPerGame' etc.
    """
    result: Dict[str, Dict] = {}
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        raw_name = cat.get("name", "").lower()
        # Normalise verbose names to short keys
        key = (raw_name
               .replace("pergame", "")
               .replace("per game", "")
               .replace("total", "")
               .strip())
        # e.g. "pointspergame" → "points", "assists" → "assists"
        tops = cat.get("leaders", [])
        if not tops:
            continue
        first   = tops[0]
        athlete = first.get("athlete", {})
        p_name  = athlete.get("displayName") or athlete.get("fullName", "")
        if p_name and key:
            result[key] = {
                "name":    p_name,
                "value":   float(first.get("value", 0)),
                "display": first.get("displayValue", ""),
            }
    return result


# ---------------------------------------------------------------------------
# Leaders strategy 1 — scoreboard (best: covers all today's teams in 1 call)
# ---------------------------------------------------------------------------

def _fetch_leaders_from_scoreboard() -> Dict[str, Dict]:
    """
    Returns {espn_display_name: leaders_dict} for every team playing today.
    ESPN's scoreboard embeds season-to-date leaders in each game's competitor.
    """
    data = _get(f"{ESPN_NBA}/scoreboard")
    if not data:
        return {}

    result: Dict[str, Dict] = {}
    for event in data.get("events", []):
        for comp in event.get("competitions", []):
            for competitor in comp.get("competitors", []):
                espn_name = competitor.get("team", {}).get("displayName", "")
                if not espn_name:
                    continue
                leaders = _parse_leader_categories(competitor.get("leaders", []))
                if leaders:
                    result[espn_name] = leaders

    logger.info(f"Scoreboard leaders: {len(result)} teams with data")
    return result


# ---------------------------------------------------------------------------
# Leaders strategy 2 — per-team fallback (tries 3 ESPN endpoints)
# ---------------------------------------------------------------------------

def _fetch_leaders_per_team(team_id: str, espn_name: str) -> Dict[str, Dict]:
    """
    Tries /statistics, /teams/{id}, then a recursive search.
    Returns first non-empty leaders dict found.
    """
    endpoints = [
        f"{ESPN_NBA}/teams/{team_id}/statistics",
        f"{ESPN_NBA}/teams/{team_id}",
    ]
    for url in endpoints:
        data = _get(url)
        if not data:
            continue

        # Try standard "categories" path (statistics endpoint)
        cats = (data.get("results", {})
                    .get("stats", {})
                    .get("categories", []))
        if cats:
            leaders = _parse_leader_categories(cats)
            if leaders:
                logger.info(f"Leaders for {espn_name} via {url.split('/')[-1]}: {list(leaders.keys())}")
                return leaders

        # Try team.leaders path (team detail endpoint)
        team_leaders_raw = data.get("team", {}).get("leaders", [])
        if team_leaders_raw:
            leaders = _parse_leader_categories(team_leaders_raw)
            if leaders:
                logger.info(f"Leaders for {espn_name} via team.leaders: {list(leaders.keys())}")
                return leaders

    logger.warning(f"No leaders found for {espn_name} (id={team_id})")
    return {}


# ---------------------------------------------------------------------------
# Team ID map — needed for per-team calls
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
                t    = item.get("team", {})
                name = t.get("displayName", "")
                tid  = t.get("id", "")
                if name and tid:
                    result[name] = tid
    return result


# ---------------------------------------------------------------------------
# Recent form — last 14 days from team schedule
# ---------------------------------------------------------------------------

def _fetch_recent_form(team_id: str, today: date, days: int = 14) -> Dict:
    """
    Fetches the team's schedule and calculates PPG/OPPG/win% over last `days` days.
    Returns last_game_date (for rest-day calculation) even when no stats available.
    """
    season = _espn_season()
    data = _get(f"{ESPN_NBA}/teams/{team_id}/schedule", params={"season": season})
    if not data:
        return {}

    cutoff = today - timedelta(days=days)
    wins = losses = 0
    pts_for = pts_against = 0.0
    game_count  = 0
    last_game_date: Optional[date] = None

    def _parse_score(raw) -> float:
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

    for event in data.get("events", []):
        raw_date = event.get("date", "")
        try:
            dt_utc    = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            # Convert to Pacific time (matches how `today` is derived) so that
            # late-night games don't shift to the wrong date under UTC.
            pac_offset = -7 if 3 <= dt_utc.month <= 10 else -8
            game_date  = (dt_utc + timedelta(hours=pac_offset)).date()
        except Exception:
            continue

        if game_date >= today or game_date < cutoff:
            continue

        for comp in event.get("competitions", []):
            competitors = comp.get("competitors", [])
            this = next((c for c in competitors if c.get("team", {}).get("id") == team_id), None)
            opp  = next((c for c in competitors if c.get("team", {}).get("id") != team_id), None)
            if not this:
                continue

            s  = _parse_score(this.get("score", 0))
            os = _parse_score(opp.get("score", 0)) if opp else 0.0
            if s > 0:
                # Only count completed games (confirmed scores) for rest-day tracking
                if last_game_date is None or game_date > last_game_date:
                    last_game_date = game_date
                if this.get("winner", False):
                    wins += 1
                else:
                    losses += 1
                pts_for     += s
                pts_against += os
                game_count  += 1

    if game_count == 0:
        return {"last_game_date": last_game_date, "games_last_7": 0}

    # Count confirmed games in the tighter 7-day window for schedule-load fatigue
    seven_day_cutoff = today - timedelta(days=7)
    games_last_7 = 0
    for event in data.get("events", []):
        raw_date = event.get("date", "")
        if not raw_date:
            continue
        try:
            dt_utc = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            pac_off = -7 if 3 <= dt_utc.month <= 10 else -8
            gd = (dt_utc + timedelta(hours=pac_off)).date()
        except Exception:
            continue
        if not (seven_day_cutoff <= gd < today):
            continue
        for comp in event.get("competitions", []):
            c = next((x for x in comp.get("competitors", [])
                      if x.get("team", {}).get("id") == team_id), None)
            if c and _parse_score(c.get("score", 0)) > 0:
                games_last_7 += 1
                break

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
        "games_last_7":   games_last_7,
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_nba_context(today: date, team_names: List[str] = None) -> Dict:
    """
    Fetch NBA context from ESPN.

    team_names: Odds API team names playing today.
      If provided → fetches leaders (via scoreboard first) + recent form + rest days.
      If None     → season stats only.

    All context dicts keyed by ESPN display names (normalize() output).
    """
    logger.info("Fetching NBA stats from ESPN...")

    # ── Season stats — all 30 teams, one standings call ──────────────────────
    season_stats = _fetch_all_team_stats()
    if not season_stats:
        logger.error("ESPN standings unavailable — NBA model will have no stats")
        return {"season_stats": {}, "recent_form": {}, "rest_days": {}, "team_leaders": {}}

    recent_form:   Dict[str, Dict] = {}
    rest_days:     Dict[str, int]  = {}
    team_leaders:  Dict[str, Dict] = {}
    schedule_load: Dict[str, int]  = {}  # initialised here so it's always defined

    if not team_names:
        logger.info("No team_names supplied — skipping per-team detail calls")
    else:
        # ── Leaders — try scoreboard first (1 call for all today's teams) ────
        scoreboard_leaders = _fetch_leaders_from_scoreboard()
        espn_names_today   = [normalize(n) for n in team_names]

        for espn_name in espn_names_today:
            if espn_name in scoreboard_leaders:
                team_leaders[espn_name] = scoreboard_leaders[espn_name]

        # Which teams are still missing leaders? → per-team fallback
        missing = [n for n in espn_names_today if n not in team_leaders]
        if missing:
            logger.info(f"Scoreboard missing leaders for: {missing} — trying per-team endpoints")
            id_map = _fetch_team_id_map()
            for espn_name in missing:
                team_id = id_map.get(espn_name)
                if not team_id:
                    for k, v in id_map.items():
                        if espn_name.lower() in k.lower() or k.lower() in espn_name.lower():
                            team_id = v
                            break
                if not team_id:
                    logger.warning(f"No ESPN team ID for: {espn_name}")
                    continue
                leaders = _fetch_leaders_per_team(team_id, espn_name)
                if leaders:
                    team_leaders[espn_name] = leaders
                time.sleep(0.25)

        # ── Recent form + rest days — per team ───────────────────────────────
        # Build id_map if not already fetched
        if missing:
            pass  # id_map already built above
        else:
            id_map = _fetch_team_id_map()

        for raw_name in team_names:
            espn_name = normalize(raw_name)
            team_id   = id_map.get(espn_name)
            if not team_id:
                rest_days[espn_name]     = 1
                schedule_load[espn_name] = 0
                continue
            recent    = _fetch_recent_form(team_id, today)
            last_date = recent.get("last_game_date")
            rest_days[espn_name] = (
                max(0, (today - last_date).days - 1) if last_date else 1
            )
            schedule_load[espn_name] = recent.get("games_last_7", 0)
            if recent.get("recent_ppg"):
                recent_form[espn_name] = {
                    "recent_net_rtg": recent["recent_net_rtg"],
                    "recent_off_rtg": recent["recent_off_rtg"],
                    "recent_def_rtg": recent["recent_def_rtg"],
                    "recent_w_pct":   recent["recent_w_pct"],
                }
            time.sleep(0.25)

    leaders_with_names = sum(
        1 for v in team_leaders.values()
        if any(cat.get("name") for cat in v.values())
    )
    logger.info(
        f"NBA context ready — season stats: {len(season_stats)} teams | "
        f"recent form: {len(recent_form)} | leaders: {len(team_leaders)} "
        f"({leaders_with_names} with player names)"
    )
    return {
        "season_stats":   season_stats,
        "recent_form":    recent_form,
        "rest_days":      rest_days,
        "schedule_load":  schedule_load,
        "team_leaders":   team_leaders,
    }
