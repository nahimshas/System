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

ESPN_NBA      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_NBA_V2   = "https://site.api.espn.com/apis/v2/sports/basketball/nba"
ESPN_NBA_V3   = "https://site.api.espn.com/apis/site/v3/sports/basketball/nba"
ESPN_NBA_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/nba"
ESPN_NBA_WEB  = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"

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
# Player props stats — roster + per-athlete averages
# ---------------------------------------------------------------------------

def _fetch_team_roster(team_id: str) -> Dict[str, str]:
    """Returns {normalized_player_name: athlete_id} for a team."""
    data = _get(f"{ESPN_NBA}/teams/{team_id}/roster")
    if not data:
        return {}
    result: Dict[str, str] = {}
    athletes = data.get("athletes", [])
    if not athletes:
        for group in data.get("roster", {}).get("athletes", []):
            athletes.extend(group.get("items", []))
    for a in athletes:
        name = a.get("displayName") or a.get("fullName", "")
        aid  = str(a.get("id", ""))
        if name and aid:
            result[name.lower().strip()] = aid
    return result


# ── v3 leaders endpoint: category name → stat key ───────────────────────────
# ONLY maps PER-GAME category names (e.g. "pointsPerGame").
# Plain totals like "points" are intentionally omitted to avoid picking up
# season totals (e.g. 227 pts) instead of per-game averages (e.g. 26.3 PPG).
_V3_CAT_TO_STAT: Dict[str, str] = {
    "pointspergame":               "pts",
    "reboundspergame":             "reb",
    "totalreboundspergame":        "reb",
    "assistspergame":              "ast",
    "stealspergame":               "stl",
    "blockspergame":               "blk",
    # 3PM — v3 endpoint uses "3PointsMadePerGame" as the category name
    "3pointsmadepergame":          "three_pm",
    "threepointfieldgoalsmade":    "three_pm",
    "avgthreepointfieldgoalsmade": "three_pm",
    "threepointers":               "three_pm",
    "threesmadepergame":           "three_pm",
}

# ── team_leaders / generic per-category mapping ──────────────────────────────
# Scoreboard team_leaders already strips "pergame" via _parse_leader_categories,
# so the keys here are the short forms ("points", "rebounds" …).
_CAT_TO_STAT: Dict[str, str] = {
    "points":                      "pts",
    "pointspergame":               "pts",
    "avgpoints":                   "pts",
    "pts":                         "pts",
    "rebounds":                    "reb",
    "reboundspergame":             "reb",
    "totalrebounds":               "reb",
    "avgtotalrebounds":            "reb",
    "avgrebounds":                 "reb",
    "reb":                         "reb",
    "assists":                     "ast",
    "assistspergame":              "ast",
    "avgassists":                  "ast",
    "ast":                         "ast",
    "steals":                      "stl",
    "stealspergame":               "stl",
    "avgsteals":                   "stl",
    "stl":                         "stl",
    "blocks":                      "blk",
    "blockspergame":               "blk",
    "avgblocks":                   "blk",
    "blk":                         "blk",
    "threepointfieldgoalsmade":    "three_pm",
    "avgthreepointfieldgoalsmade": "three_pm",
    "threepointers":               "three_pm",
    "avgthreepointers":            "three_pm",
    "threes":                      "three_pm",
    "threesmade":                  "three_pm",
    "3pm":                         "three_pm",
}

_BLANK_STATS = lambda team="": {
    "pts": 0.0, "reb": 0.0, "ast": 0.0,
    "stl": 0.0, "blk": 0.0, "three_pm": 0.0,
    "games": 40, "team": team,
}


def _parse_stat_vals(data: dict) -> Dict[str, float]:
    """Collect all stat name→value pairs from common ESPN category/split layouts."""
    vals: Dict[str, float] = {}

    def _absorb(categories):
        for cat in categories:
            if not isinstance(cat, dict):
                continue
            for s in cat.get("stats", []):
                k = s.get("name", "")
                v = s.get("value")
                if k and v is not None:
                    try:
                        vals[k] = float(v)
                    except (TypeError, ValueError):
                        pass

    # Layout A: {"splits": {"categories": [...]}}
    splits = data.get("splits", {})
    if isinstance(splits, dict):
        _absorb(splits.get("categories", []))

    # Layout B: {"categories": [...]}
    _absorb(data.get("categories", []))

    # Layout C: {"seasonTypes": [{"categories": [...]}]}
    for stype in data.get("seasonTypes", []):
        _absorb(stype.get("categories", []))
        if vals:
            break   # regular-season comes first

    # Layout D: {"entries": [{"statistics": {"splits": {"categories": [...]}}}]}
    for entry in data.get("entries", []):
        sdata = entry.get("statistics", {})
        sp = sdata.get("splits", {})
        if isinstance(sp, dict):
            _absorb(sp.get("categories", []))
        _absorb(sdata.get("categories", []))
        if vals:
            break

    return vals


def _stats_from_vals(vals: Dict[str, float]) -> Dict:
    def _pick(*keys: str) -> float:
        for k in keys:
            if k in vals:
                return vals[k]
        return 0.0

    return {
        "pts":      _pick("avgPoints", "points", "pointsPerGame"),
        "reb":      _pick("avgRebounds", "avgTotalRebounds", "reboundsPerGame", "totalRebounds"),
        "ast":      _pick("avgAssists", "assists", "assistsPerGame"),
        "stl":      _pick("avgSteals", "steals", "stealsPerGame"),
        "blk":      _pick("avgBlocks", "blocks", "blocksPerGame"),
        "three_pm": _pick("avgThreePointFieldGoalsMade", "avgThreesMade",
                          "threePointFieldGoalsMade", "threePointersMade"),
        "games":    int(_pick("gamesPlayed", "games") or 1),
    }


# ---------------------------------------------------------------------------
# Tier-1 source: ESPN v3 league leaders (1 call → top-25 per stat category)
# ---------------------------------------------------------------------------

def _fetch_league_leaders_v3() -> Dict[str, Dict]:
    """
    GET /leaders from the v3 endpoint — returns top-10 players per stat category
    league-wide (16 categories).  One call covers almost every player who gets
    prop lines.

    Response shape:
      {"leaders": {"categories": [{"name": "pointsPerGame", "abbreviation": "PTS",
                                   "leaders": [{"athlete": {...}, "team": {...},
                                               "value": 33.75}, ...]}, ...]}}

    Returns {norm_player_name: {pts, reb, ast, stl, blk, three_pm, games, team}}
    """
    data = _get(f"{ESPN_NBA_V3}/leaders")
    if not data:
        logger.warning("ESPN v3 leaders: no data returned")
        return {}

    # The actual payload lives one level deep under the "leaders" key
    leaders_blob = data.get("leaders", {})
    if isinstance(leaders_blob, dict):
        categories = leaders_blob.get("categories", [])
    else:
        # Fallback: maybe it's already a list on older API versions
        categories = leaders_blob if isinstance(leaders_blob, list) else []

    result: Dict[str, Dict] = {}

    for cat in categories:
        if not isinstance(cat, dict):
            continue
        # Normalise: "pointsPerGame" → "pointspergame" (keep "pergame" — used by _V3_CAT_TO_STAT
        # to distinguish per-game from season totals like plain "points")
        raw = (cat.get("name") or cat.get("abbreviation") or "").lower()
        raw = raw.replace(" ", "").replace("-", "")
        stat_key = _V3_CAT_TO_STAT.get(raw)
        if not stat_key:
            logger.debug(f"v3 leaders: skipping category '{raw}' (abbr={cat.get('abbreviation','')})")
            continue

        leaders_list = cat.get("leaders", [])
        logger.debug(f"v3 leaders '{raw}' → '{stat_key}': {len(leaders_list)} players")

        for entry in leaders_list:
            if not isinstance(entry, dict):
                continue
            athlete  = entry.get("athlete", {})
            p_name   = athlete.get("displayName") or athlete.get("fullName", "")
            if not p_name:
                continue
            value = float(entry.get("value", 0) or 0)
            if value == 0:
                continue

            # Team: embedded in entry (not in athlete sub-object)
            team_obj  = entry.get("team") or {}
            team_name = team_obj.get("displayName", "")

            norm_p = p_name.lower().strip()
            if norm_p not in result:
                result[norm_p] = _BLANK_STATS(team_name)
            result[norm_p][stat_key] = value
            if team_name and not result[norm_p]["team"]:
                result[norm_p]["team"] = team_name

    logger.info(f"ESPN v3 league leaders: {len(result)} unique players across {len(categories)} stat categories")
    return result


# ---------------------------------------------------------------------------
# Tier-2 source: nba_ctx team_leaders (already fetched, 0 extra API calls)
# ---------------------------------------------------------------------------

def _build_team_leaders_lookup(nba_ctx: Optional[Dict]) -> Dict[str, Dict]:
    """
    Extract per-player stats from nba_ctx['team_leaders'].
    Only covers the #1 leader per stat per team, but costs nothing extra.
    """
    lookup: Dict[str, Dict] = {}
    if not nba_ctx:
        return lookup
    for espn_name, categories in nba_ctx.get("team_leaders", {}).items():
        for cat_key, cat_data in categories.items():
            raw = cat_key.lower().replace(" ", "").replace("-", "").replace("pergame", "").replace("avg", "")
            stat_key = _CAT_TO_STAT.get(raw)
            if not stat_key:
                continue
            p_name = cat_data.get("name", "")
            value  = float(cat_data.get("value", 0) or 0)
            if not p_name:
                continue
            norm_p = p_name.lower().strip()
            if norm_p not in lookup:
                lookup[norm_p] = _BLANK_STATS(espn_name)
            lookup[norm_p][stat_key] = value
    logger.debug(f"team_leaders lookup: {len(lookup)} unique players")
    return lookup


# ---------------------------------------------------------------------------
# Tier-3 sources: per-athlete fallback endpoints
# ---------------------------------------------------------------------------

def _fetch_athlete_web_stats(athlete_id: str) -> Dict:
    """
    Fetch per-game season averages via site.web.api.espn.com (common/v3).
    Stats live at athlete.statsSummary.statistics — a flat list of
    {name, value} objects (avgPoints, avgRebounds, avgAssists, …).
    Works for ALL NBA players, not just leaders.
    Returns pts / reb / ast (steals, blocks, 3PM not included by this endpoint).
    """
    data = _get(f"{ESPN_NBA_WEB}/athletes/{athlete_id}")
    if not data:
        return {}
    stats_list = (data.get("athlete", {})
                      .get("statsSummary", {})
                      .get("statistics", []))
    if not stats_list:
        return {}
    vals = {s["name"]: float(s.get("value", 0) or 0)
            for s in stats_list if "name" in s and "value" in s}
    return _stats_from_vals(vals) if vals else {}


def _fetch_athlete_site_stats(athlete_id: str) -> Dict:
    """
    Fetch full per-game season stats (incl. stl/blk/3PM) via the ESPN site API.
    Used to supplement tier-3 players who only get pts/reb/ast from the web endpoint.
    """
    data = _get(f"{ESPN_NBA}/athletes/{athlete_id}/statistics")
    if not data:
        return {}
    vals = _parse_stat_vals(data)
    return _stats_from_vals(vals) if vals else {}


# ---------------------------------------------------------------------------
# Public: player props stats with 3-tier fallback
# ---------------------------------------------------------------------------

def _find_in_lookup(player_name: str, lookup: Dict) -> Optional[Dict]:
    norm = player_name.lower().strip()
    if norm in lookup:
        return lookup[norm]
    parts = norm.split()
    for lk_name, data in lookup.items():
        if all(p in lk_name for p in parts):
            return data
    return None


def get_nba_player_props_stats(
    player_names:     List[str],
    team_names_today: List[str],
    nba_ctx:          Optional[Dict] = None,
) -> Dict[str, Dict]:
    """
    Fetch per-game season averages for players with Odds API prop lines.

    Tier 1 — ESPN v3 /leaders (1 call, top-25 per stat, league-wide).
    Tier 2 — nba_ctx['team_leaders'] (0 extra calls, #1 per stat per team).
    Tier 3 — roster lookup + per-athlete fallback endpoints:
              statisticslog → web stats → gamelog.

    Returns {player_name: {"stats": {pts/reb/ast/stl/blk/three_pm/games}, "team": espn_name}}
    """
    if not player_names or not team_names_today:
        return {}

    # ── Tier 1 + 2: bulk lookups ────────────────────────────────────────────
    v3_lookup          = _fetch_league_leaders_v3()
    team_leader_lookup = _build_team_leaders_lookup(nba_ctx)

    result:    Dict[str, Dict] = {}
    remaining: List[str]       = []

    for player_name in player_names:
        entry = _find_in_lookup(player_name, v3_lookup)
        if not entry:
            entry = _find_in_lookup(player_name, team_leader_lookup)
        if entry:
            stats = {k: v for k, v in entry.items() if k != "team"}
            result[player_name] = {"stats": stats, "team": entry.get("team", "")}
        else:
            remaining.append(player_name)

    # Players found in Tier 1/2 only via stl/blk (pts=0 AND reb=0) still need
    # their scoring/rebounding stats filled in via the web API.
    need_supplement = [
        p for p in player_names
        if p in result
        and result[p]["stats"].get("pts", 0) == 0
        and result[p]["stats"].get("reb", 0) == 0
    ]

    logger.info(
        f"NBA props — bulk tiers: {len(result)}/{len(player_names)} resolved, "
        f"{len(remaining)} unresolved + {len(need_supplement)} need stat supplement"
    )

    # ── Tier 3: web API for unresolved + partial-stats players ───────────────
    # site.web.api.espn.com/apis/common/v3/athletes/{id} returns pts/reb/ast
    # for ALL players via athlete.statsSummary.statistics.
    # Note: statisticslog and gamelog are both 404 during playoffs; skip them.
    needs_tier3 = remaining + need_supplement
    if needs_tier3:
        id_map = _fetch_team_id_map()

        all_rosters: Dict[str, tuple] = {}
        for espn_name in set(team_names_today):
            team_id = id_map.get(espn_name)
            if not team_id:
                continue
            roster = _fetch_team_roster(team_id)
            for norm_name, aid in roster.items():
                all_rosters[norm_name] = (aid, espn_name)
            time.sleep(0.15)

        fb_new = fb_supp = 0
        for player_name in needs_tier3:
            norm  = player_name.lower().strip()
            entry = all_rosters.get(norm)
            if not entry:
                for roster_norm, val in all_rosters.items():
                    if all(p in roster_norm for p in norm.split()):
                        entry = val
                        break
            if not entry:
                logger.debug(f"No ESPN athlete ID for: {player_name}")
                continue

            athlete_id, espn_team = entry
            web_stats = _fetch_athlete_web_stats(athlete_id)
            time.sleep(0.15)

            if not (web_stats.get("pts", 0) > 0 or web_stats.get("reb", 0) > 0):
                continue

            if player_name in result:
                # Supplement: fill pts/reb/ast into the existing entry (keeps stl/blk from v3)
                existing = result[player_name]["stats"]
                for k in ("pts", "reb", "ast", "games"):
                    if existing.get(k, 0) == 0 and web_stats.get(k, 0) > 0:
                        existing[k] = web_stats[k]
                if not result[player_name]["team"] and espn_team:
                    result[player_name]["team"] = espn_team
                fb_supp += 1
            else:
                # New tier-3 player: web API only returns pts/reb/ast.
                # Try site API to fill stl/blk/three_pm so peripheral props aren't silently dropped.
                site_stats = _fetch_athlete_site_stats(athlete_id)
                time.sleep(0.15)
                merged = dict(web_stats)
                for k in ("stl", "blk", "three_pm"):
                    if site_stats.get(k, 0) > 0:
                        merged[k] = site_stats[k]
                result[player_name] = {"stats": merged, "team": espn_team}
                fb_new += 1

        logger.info(
            f"Web API fallback: {fb_new} new players resolved, "
            f"{fb_supp} existing entries supplemented with pts/reb/ast"
        )

    logger.info(f"NBA player props stats: {len(result)}/{len(player_names)} players resolved total")
    return result


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
