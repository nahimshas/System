"""
Fetches NFL team stats via ESPN's public API.

Strategy:
  1. Standings endpoint → season win/loss, points for/against for all 32 teams
  2. Scoreboard endpoint → rest days (days since last game)
  3. Team record / recent schedule → last-14-day form estimate

ESPN NFL season year = the year the season starts (2024 season → 2024).
"""
import logging
import time
import requests
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_NFL      = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
ESPN_NFL_V2   = "https://site.api.espn.com/apis/v2/sports/football/nfl"
ESPN_NFL_CORE = "https://sports.core.api.espn.com/v2/sports/football/nfl"

# Odds API name → ESPN displayName (only entries that differ)
_ODDS_TO_ESPN: Dict[str, str] = {
    "New York Giants":  "NY Giants",
    "New York Jets":    "NY Jets",
    "Los Angeles Rams": "Los Angeles Rams",   # same, no change needed
    "Las Vegas Raiders": "Las Vegas Raiders",
}
_ESPN_TO_ODDS: Dict[str, str] = {v: k for k, v in _ODDS_TO_ESPN.items()}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName (key used in all ctx dicts)."""
    return _ODDS_TO_ESPN.get(name, name)


def _nfl_season() -> int:
    """
    ESPN NFL season year = the year the season starts.
    August 2024 – February 2025 → season 2024.
    """
    today = date.today()
    # NFL season starts in September. If we're Jan–July, we're in the previous season.
    return today.year if today.month >= 8 else today.year - 1


def _get(url: str, params: dict = None) -> Optional[dict]:
    """GET → JSON or None on any error."""
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN NFL GET failed [{url}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Season stats — standings endpoint (one call for all 32 teams)
# ---------------------------------------------------------------------------

# Warm-start (season cold-start) constants.
# NFL team strength = net points/game, which is 0 for every team in ESPN's
# standings until real games are played. Without a prior, the model rates
# Week-1 games as coin-flips-plus-home-field and bets every rebuilding
# underdog against every strong favorite (proven Jul 2026). The warm start
# carries last season's ratings forward, regressed toward the mean (NFL teams
# revert hard year-to-year), and ramps out as current-season games accumulate.
NFL_WARMSTART_RAMP_GAMES = 6      # current-season games until pure current data
NFL_PRIOR_REGRESSION     = 0.67   # keep 67% of prior net rating (regress 1/3 → mean)
NFL_LEAGUE_AVG_PPG       = 21.5   # anchor for regressing ppg/oppg toward league mean


def _fetch_all_team_stats(season: int = None) -> Dict[str, Dict]:
    """
    Returns {espn_display_name: stats_dict} for every NFL team from standings.
    `season` defaults to the current NFL season; pass season-1 for the prior.
    """
    if season is None:
        season = _nfl_season()
    data = _get(f"{ESPN_NFL_V2}/standings", params={"season": season})
    if not data:
        return {}

    # ESPN's standings nesting changed to conf → teams (was conf → division →
    # teams). Walk the tree robustly: collect `standings.entries` at whatever
    # depth they appear, so both the old and new structures parse. (Without
    # this the old two-level loop silently returned ZERO teams → every game
    # cold-started. Fixed Jul 2026.)
    def _collect_entries(node) -> list:
        out = list((node.get("standings") or {}).get("entries", []) or [])
        for child in node.get("children", []) or []:
            out.extend(_collect_entries(child))
        return out

    result: Dict[str, Dict] = {}
    for conf in data.get("children", []):
            for entry in _collect_entries(conf):
                espn_name = entry.get("team", {}).get("displayName", "")
                if not espn_name:
                    continue

                stat_map = {s["name"]: s.get("value", 0.0)
                            for s in entry.get("stats", []) if "name" in s}

                ppg  = float(stat_map.get("avgPointsFor",
                             stat_map.get("pointsFor", 21.0)) or 21.0)
                oppg = float(stat_map.get("avgPointsAgainst",
                             stat_map.get("pointsAgainst", 21.0)) or 21.0)
                wins   = float(stat_map.get("wins",   0))
                losses = float(stat_map.get("losses", 0))
                ties   = float(stat_map.get("ties", 0))
                total  = wins + losses + ties
                win_pct = float(stat_map.get("winPercent",
                                wins / total if total > 0 else 0.5))

                result[espn_name] = {
                    "off_rtg":  ppg,           # points/game scored
                    "def_rtg":  oppg,          # points/game allowed
                    "net_rtg":  ppg - oppg,    # point differential per game
                    "ppg":      ppg,
                    "oppg":     oppg,
                    "win_pct":  win_pct,
                    "wins":     int(wins),
                    "losses":   int(losses),
                    "games_played": int(total),
                }

    logger.info(f"NFL season {season} stats: {len(result)} teams loaded")
    return result


def _apply_warm_start(current: Dict[str, Dict], prior: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Blend each team's current-season ratings with last-season's (regressed toward
    the mean), weighted by how many current games have been played. Week 1
    (0 games) → pure regressed prior; by ~Week 6 → pure current season.
    Mutates and returns `current`. If `prior` is empty, current is left as-is.
    """
    if not prior:
        logger.warning("NFL warm-start: no prior-season data — using current only")
        return current
    for name, cur in current.items():
        gp = cur.get("games_played", 0)
        w  = min(1.0, gp / NFL_WARMSTART_RAMP_GAMES) if NFL_WARMSTART_RAMP_GAMES else 1.0
        pri = prior.get(name, {})
        # Regress the prior toward the mean: net → 0, ppg/oppg → league avg.
        pri_net  = pri.get("net_rtg", 0.0) * NFL_PRIOR_REGRESSION
        pri_ppg  = NFL_LEAGUE_AVG_PPG + (pri.get("ppg",  NFL_LEAGUE_AVG_PPG) - NFL_LEAGUE_AVG_PPG) * NFL_PRIOR_REGRESSION
        pri_oppg = NFL_LEAGUE_AVG_PPG + (pri.get("oppg", NFL_LEAGUE_AVG_PPG) - NFL_LEAGUE_AVG_PPG) * NFL_PRIOR_REGRESSION
        cur["net_rtg"] = round(w * cur["net_rtg"] + (1 - w) * pri_net, 3)
        cur["ppg"]     = round(w * cur["ppg"]  + (1 - w) * pri_ppg, 3)
        cur["oppg"]    = round(w * cur["oppg"] + (1 - w) * pri_oppg, 3)
        cur["off_rtg"] = cur["ppg"]
        cur["def_rtg"] = cur["oppg"]
        cur["warm_start_weight"] = round(w, 3)   # 0 = pure prior, 1 = pure current
    active = sum(1 for c in current.values() if c.get("warm_start_weight", 1.0) < 1.0)
    if active:
        logger.info(f"NFL warm-start ACTIVE: {active} team(s) blending last-season priors "
                    f"(regressed ×{NFL_PRIOR_REGRESSION}, ramp {NFL_WARMSTART_RAMP_GAMES} games)")
    return current


# ---------------------------------------------------------------------------
# Recent form — last 14 days of games from scoreboard history
# ---------------------------------------------------------------------------

def _fetch_recent_form(team_map: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Estimates recent form from completed games in the last 14 days.
    Returns {display_name: {recent_net_rtg, recent_w_pct}}.
    Falls back to season stats if no recent games found.
    """
    recent: Dict[str, Dict] = {}
    today = date.today()

    # Collect last 14 days of scoreboard data
    team_scores: Dict[str, list] = {}   # espn_name → [(margin, win_bool), ...]

    for delta in range(1, 15):
        d = today - timedelta(days=delta)
        datestr = d.strftime("%Y%m%d")
        data = _get(f"{ESPN_NFL}/scoreboard", params={"dates": datestr})
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
                margin = scores[names[0]] - scores[names[1]]
                won    = margin > 0
                for i, name in enumerate(names):
                    sign = 1 if i == 0 else -1
                    team_scores.setdefault(name, []).append((sign * margin, won if i == 0 else not won))
            except Exception:
                continue
        time.sleep(0.05)

    for name, entries in team_scores.items():
        if not entries:
            continue
        avg_margin = sum(e[0] for e in entries) / len(entries)
        w_pct      = sum(1 for e in entries if e[1]) / len(entries)
        recent[name] = {
            "recent_net_rtg": avg_margin,
            "recent_w_pct":   w_pct,
        }

    # Fill gaps with season stats
    for name, stats in team_map.items():
        if name not in recent:
            recent[name] = {
                "recent_net_rtg": stats.get("net_rtg", 0.0),
                "recent_w_pct":   stats.get("win_pct", 0.5),
            }

    return recent


# ---------------------------------------------------------------------------
# Rest days — days since each team's last game
# ---------------------------------------------------------------------------

def _fetch_rest_days(team_names: List[str]) -> Dict[str, int]:
    """
    Returns {display_name: days_since_last_game}.
    Defaults to 7 (normal weekly cadence) if not found.
    NFL bye week = 14 days rest.
    """
    rest: Dict[str, int] = {}
    today = date.today()

    for delta in range(1, 22):   # look back up to 3 weeks
        if all(t in rest for t in team_names):
            break
        d = today - timedelta(days=delta)
        data = _get(f"{ESPN_NFL}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
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

    # Default 7 for any team not yet found
    for t in team_names:
        rest.setdefault(t, 7)

    return rest


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_nfl_context(today: date, team_names: List[str] = None) -> Dict:
    """
    Returns full NFL context dict for edge_finder:
    {
        "season_stats":  {display_name: stats_dict},
        "recent_form":   {display_name: form_dict},
        "rest_days":     {display_name: int},
    }
    """
    team_names = team_names or []
    norm_names = [normalize(n) for n in team_names]

    season_stats = _fetch_all_team_stats()
    # Warm start: blend last season's ratings in until the current season has
    # enough games to stand alone (prevents the Week-1 coin-flip cold start).
    try:
        prior_stats = _fetch_all_team_stats(_nfl_season() - 1)
        season_stats = _apply_warm_start(season_stats, prior_stats)
    except Exception as e:
        logger.warning(f"NFL warm-start failed (using current season only): {e}")

    try:
        recent_form = _fetch_recent_form(season_stats)
    except Exception as e:
        logger.warning(f"NFL recent form fetch failed: {e}")
        recent_form = {n: {"recent_net_rtg": season_stats.get(n, {}).get("net_rtg", 0.0),
                            "recent_w_pct":  season_stats.get(n, {}).get("win_pct", 0.5)}
                       for n in norm_names}

    try:
        rest_days = _fetch_rest_days(norm_names)
    except Exception as e:
        logger.warning(f"NFL rest days fetch failed: {e}")
        rest_days = {n: 7 for n in norm_names}

    return {
        "season_stats": season_stats,
        "recent_form":  recent_form,
        "rest_days":    rest_days,
    }
