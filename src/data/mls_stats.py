"""
MLS (Major League Soccer) statistics and context data.

Data sources:
  - ASA API (no auth): https://app.americansocceranalysis.com/api/v1/
  - ESPN standings: https://site.api.espn.com/apis/v2/sports/soccer/usa.1/standings
  - ESPN scoreboard: https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard
  - ESPN injuries: https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/teams/{id}/injuries
"""
import logging
import requests
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ASA_BASE = "https://app.americansocceranalysis.com/api/v1"
ESPN_SOCCER_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1"
ESPN_SOCCER_STANDINGS = "https://site.api.espn.com/apis/v2/sports/soccer/usa.1/standings"

_REQUEST_TIMEOUT = 15


def normalize(name: str, ctx_dict: dict) -> str:
    """Return the closest key in ctx_dict matching name, or name itself."""
    name_lower = name.lower()
    for key in ctx_dict:
        if key.lower() == name_lower:
            return key
    for key in ctx_dict:
        if name_lower in key.lower() or key.lower() in name_lower:
            return key
    # Partial word match
    name_words = set(name_lower.split())
    best, best_score = name, 0
    for key in ctx_dict:
        key_words = set(key.lower().split())
        score = len(name_words & key_words)
        if score > best_score:
            best, best_score = key, score
    return best if best_score > 0 else name


def _get(url: str, params: dict = None) -> Optional[dict]:
    try:
        r = requests.get(url, params=params or {}, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"HTTP fetch failed ({url}): {e}")
        return None


def _get_list(url: str, params: dict = None) -> List:
    result = _get(url, params)
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        # Some ASA endpoints wrap in a list at top level, others in a key
        for key in ("data", "teams", "games"):
            if key in result and isinstance(result[key], list):
                return result[key]
    return []


# ---------------------------------------------------------------------------
# ASA team ID → name mapping
# ---------------------------------------------------------------------------

def _get_asa_team_map() -> Dict[str, str]:
    """Returns {team_id: team_name} from ASA /mls/teams endpoint."""
    teams = _get_list(f"{ASA_BASE}/mls/teams")
    mapping: Dict[str, str] = {}
    for t in teams:
        tid = t.get("team_id") or t.get("id")
        name = (t.get("team_name") or t.get("name") or "").strip()
        if tid and name:
            mapping[tid] = name
    logger.info(f"ASA team map: {len(mapping)} entries")
    return mapping


# ---------------------------------------------------------------------------
# Season stats from ASA /mls/teams/xgoals
# ---------------------------------------------------------------------------

def _get_asa_season_stats(season: int, team_map: Dict[str, str]) -> Dict[str, dict]:
    """Returns {team_name: {xgf_per_game, xga_per_game, ...}}"""
    records = _get_list(f"{ASA_BASE}/mls/teams/xgoals", {"season_name": str(season)})
    if not records:
        return {}
    result: Dict[str, dict] = {}
    for rec in records:
        tid = rec.get("team_id") or rec.get("id")
        name = team_map.get(tid) if tid else None
        if not name:
            name = (rec.get("team_name") or rec.get("name") or "").strip()
        if not name:
            continue
        gp = rec.get("games_played") or rec.get("count_games") or 1
        if gp == 0:
            gp = 1
        xgf = rec.get("xGoals") or rec.get("xgoals") or rec.get("xgf") or 0.0
        xga = rec.get("xGoalsAgainst") or rec.get("xgoals_against") or rec.get("xga") or 0.0
        result[name] = {
            "xgf_per_game": round(xgf / gp, 3),
            "xga_per_game": round(xga / gp, 3),
            "xgd_per_game": round((xgf - xga) / gp, 3),
            "home_xgf": 0.0,
            "home_xga": 0.0,
            "away_xgf": 0.0,
            "away_xga": 0.0,
            "home_games": 0,
            "away_games": 0,
            "gf": 0,
            "ga": 0,
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
    logger.info(f"ASA season stats ({season}): {len(result)} teams")
    return result


# ---------------------------------------------------------------------------
# Per-game xG data from ASA /mls/games/xgoals
# ---------------------------------------------------------------------------

def _get_asa_game_data(season: int, team_map: Dict[str, str]) -> List[dict]:
    """Returns list of game records with home/away team names and xG values."""
    games = _get_list(f"{ASA_BASE}/mls/games/xgoals", {"season_name": str(season)})
    result = []
    for g in games:
        htid = g.get("home_team_id")
        atid = g.get("away_team_id")
        home_name = team_map.get(htid, "") if htid else ""
        away_name = team_map.get(atid, "") if atid else ""
        if not home_name:
            home_name = g.get("home_team_name") or g.get("home_team") or ""
        if not away_name:
            away_name = g.get("away_team_name") or g.get("away_team") or ""

        # Handle various field name conventions
        home_xg = (
            g.get("home_team_xgoals")
            or g.get("xGoals_home")
            or g.get("home_xg")
            or g.get("home_xgoals")
            or 0.0
        )
        away_xg = (
            g.get("away_team_xgoals")
            or g.get("xGoals_away")
            or g.get("away_xg")
            or g.get("away_xgoals")
            or 0.0
        )
        dt_str = g.get("date_time_utc") or g.get("date") or ""
        result.append({
            "home_team": home_name,
            "away_team": away_name,
            "home_xg": float(home_xg),
            "away_xg": float(away_xg),
            "date_str": dt_str,
        })
    return result


def _compute_home_away_splits(
    game_records: List[dict], season_stats: Dict[str, dict]
) -> None:
    """Mutates season_stats to fill home/away xG averages from per-game data."""
    home_xgf: Dict[str, List[float]] = {}
    home_xga: Dict[str, List[float]] = {}
    away_xgf: Dict[str, List[float]] = {}
    away_xga: Dict[str, List[float]] = {}

    for g in game_records:
        ht = g["home_team"]
        at = g["away_team"]
        if ht:
            home_xgf.setdefault(ht, []).append(g["home_xg"])
            home_xga.setdefault(ht, []).append(g["away_xg"])
        if at:
            away_xgf.setdefault(at, []).append(g["away_xg"])
            away_xga.setdefault(at, []).append(g["home_xg"])

    for team, stats in season_stats.items():
        # Match by partial name to handle slight naming differences
        ht_key = next((k for k in home_xgf if k.lower() == team.lower()), None)
        if ht_key is None:
            ht_key = next((k for k in home_xgf if team.lower() in k.lower() or k.lower() in team.lower()), None)

        at_key = next((k for k in away_xgf if k.lower() == team.lower()), None)
        if at_key is None:
            at_key = next((k for k in away_xgf if team.lower() in k.lower() or k.lower() in team.lower()), None)

        if ht_key and home_xgf.get(ht_key):
            hf = home_xgf[ht_key]
            ha = home_xga.get(ht_key, [])
            stats["home_xgf"] = round(sum(hf) / len(hf), 3)
            stats["home_xga"] = round(sum(ha) / len(ha), 3) if ha else stats["xga_per_game"]
            stats["home_games"] = len(hf)

        if at_key and away_xgf.get(at_key):
            af = away_xgf[at_key]
            aa = away_xga.get(at_key, [])
            stats["away_xgf"] = round(sum(af) / len(af), 3)
            stats["away_xga"] = round(sum(aa) / len(aa), 3) if aa else stats["xga_per_game"]
            stats["away_games"] = len(af)


def _compute_recent_form(
    game_records: List[dict], n_games: int = 8
) -> Dict[str, dict]:
    """Returns {team_name: {recent_xgf, recent_xga, recent_xgd, games}}"""
    # Sort by date descending
    def _parse_dt(s: str):
        if not s:
            return datetime.min
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    sorted_games = sorted(game_records, key=lambda g: _parse_dt(g["date_str"]), reverse=True)

    # Collect per-team games (ordered most recent first)
    team_games: Dict[str, List[dict]] = {}
    for g in sorted_games:
        for side, team, xgf_key, xga_key in [
            ("home", g["home_team"], "home_xg", "away_xg"),
            ("away", g["away_team"], "away_xg", "home_xg"),
        ]:
            if not team:
                continue
            entry = {"xgf": g[xgf_key], "xga": g[xga_key]}
            team_games.setdefault(team, []).append(entry)

    result: Dict[str, dict] = {}
    for team, games in team_games.items():
        recent = games[:n_games]
        n = len(recent)
        if n == 0:
            continue
        rxgf = sum(r["xgf"] for r in recent) / n
        rxga = sum(r["xga"] for r in recent) / n
        result[team] = {
            "recent_xgf": round(rxgf, 3),
            "recent_xga": round(rxga, 3),
            "recent_xgd": round(rxgf - rxga, 3),
            "games": n,
        }
    return result


# ---------------------------------------------------------------------------
# ESPN standings → W/L/D, GF/GA, points
# ---------------------------------------------------------------------------

def _get_espn_standings(season_stats: Dict[str, dict]) -> None:
    """Mutates season_stats to add W/L/D/GF/GA/points from ESPN standings."""
    data = _get(ESPN_SOCCER_STANDINGS)
    if not data:
        return

    for group in (data.get("standings") or data.get("children") or []):
        entries = group.get("standings", {}).get("entries", []) if isinstance(group, dict) else []
        if not entries:
            # Try direct entries
            entries = group.get("entries", []) if isinstance(group, dict) else []
        for entry in entries:
            team = entry.get("team", {})
            tname = team.get("displayName") or team.get("name") or ""
            if not tname:
                continue
            # Match to season_stats
            matched = normalize(tname, season_stats)
            if matched not in season_stats:
                continue
            stats_ref = season_stats[matched]
            for stat in entry.get("stats", []):
                sname = stat.get("name", "")
                sval = stat.get("value")
                if sval is None:
                    continue
                try:
                    sval = int(float(sval))
                except Exception:
                    continue
                if sname in ("wins", "W"):
                    stats_ref["wins"] = sval
                elif sname in ("losses", "L"):
                    stats_ref["losses"] = sval
                elif sname in ("ties", "draws", "D"):
                    stats_ref["draws"] = sval
                elif sname in ("pointsFor", "goalsFor", "gf", "GF"):
                    stats_ref["gf"] = sval
                elif sname in ("pointsAgainst", "goalsAgainst", "ga", "GA"):
                    stats_ref["ga"] = sval
                elif sname in ("points", "pts", "PTS"):
                    stats_ref["points"] = sval


# ---------------------------------------------------------------------------
# Rest days from ESPN scoreboard
# ---------------------------------------------------------------------------

def _get_rest_days(today: date, team_names: List[str]) -> Dict[str, int]:
    """
    Returns {team_name: days_since_last_game}.
    Fetches recent ESPN scoreboards to find the most recently completed game per team.
    """
    rest: Dict[str, int] = {t: 5 for t in team_names}  # default 5

    # Check up to 14 days back
    for days_back in range(1, 15):
        check_date = today - timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")
        url = f"{ESPN_SOCCER_BASE}/scoreboard"
        data = _get(url, {"dates": date_str})
        if not data:
            continue

        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            for c in competitors:
                tname = c.get("team", {}).get("displayName", "")
                if not tname:
                    continue
                matched = normalize(tname, {t: t for t in team_names})
                if matched in rest:
                    # Update only if this is the first (most recent) game found
                    if rest[matched] == 5:  # still at default
                        rest[matched] = days_back

    return rest


# ---------------------------------------------------------------------------
# Injuries from ESPN
# ---------------------------------------------------------------------------

def _get_espn_team_ids() -> Dict[str, int]:
    """Returns {team_display_name: espn_team_id}"""
    data = _get(f"{ESPN_SOCCER_BASE}/teams")
    if not data:
        return {}
    result: Dict[str, int] = {}
    sports = data.get("sports", [])
    for sport in sports:
        for league in sport.get("leagues", []):
            for team in league.get("teams", []):
                t = team.get("team", {})
                tid = t.get("id")
                name = t.get("displayName") or t.get("name") or ""
                if tid and name:
                    try:
                        result[name] = int(tid)
                    except (ValueError, TypeError):
                        pass
    # Also try top-level teams key
    for team in data.get("teams", []):
        t = team.get("team", team)
        tid = t.get("id")
        name = t.get("displayName") or t.get("name") or ""
        if tid and name:
            try:
                result[name] = int(tid)
            except (ValueError, TypeError):
                pass
    logger.info(f"ESPN MLS team IDs: {len(result)} teams")
    return result


def _get_espn_injuries(team_id_map: Dict[str, int]) -> Dict[str, List[dict]]:
    """Returns {team_name: [{player, position, status, goals_contrib}]}"""
    injuries: Dict[str, List[dict]] = {}

    for team_name, team_id in team_id_map.items():
        url = f"{ESPN_SOCCER_BASE}/teams/{team_id}/injuries"
        data = _get(url)
        if not data:
            continue
        team_injuries = []
        for item in data.get("injuries", []) or []:
            athlete = item.get("athlete", {})
            player = athlete.get("displayName") or athlete.get("fullName") or ""
            position = athlete.get("position", {}).get("abbreviation", "") if isinstance(athlete.get("position"), dict) else ""
            status = item.get("type", {}).get("description", "") or item.get("status", "")
            # Estimate goals_contrib: GK = 0.0, outfield = 0.3 placeholder
            is_gk = "G" == position.upper()[:1] or "GK" in position.upper()
            goals_contrib = 0.0 if is_gk else 0.3
            if player:
                team_injuries.append({
                    "player": player,
                    "position": position,
                    "status": status,
                    "goals_contrib": goals_contrib,
                })
        if team_injuries:
            injuries[team_name] = team_injuries
            logger.debug(f"MLS injuries — {team_name}: {len(team_injuries)} player(s)")

    return injuries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mls_injuries() -> Dict[str, List[dict]]:
    """Standalone function returning MLS injuries dict. Called from outcome_checker/main."""
    team_id_map = _get_espn_team_ids()
    if not team_id_map:
        logger.warning("MLS injuries: could not fetch team IDs from ESPN")
        return {}
    return _get_espn_injuries(team_id_map)


def get_mls_context(today: date, team_names: List[str]) -> dict:
    """
    Main entry point. Returns:
    {
        "season_stats": { team_name: {...} },
        "recent_form":  { team_name: {...} },
        "rest_days":    { team_name: int },
        "injuries":     { team_name: [...] },
    }
    """
    current_year = today.year

    # 1. ASA team ID mapping
    team_map = _get_asa_team_map()

    # 2. Season stats — try current year first, fall back if fewer than 5 entries
    season_stats = _get_asa_season_stats(current_year, team_map)
    if len(season_stats) < 5:
        logger.info(f"MLS: {current_year} has only {len(season_stats)} team entries — trying {current_year - 1}")
        fallback_stats = _get_asa_season_stats(current_year - 1, team_map)
        if len(fallback_stats) >= len(season_stats):
            season_stats = fallback_stats

    # 3. Per-game data for home/away splits and recent form
    game_records = _get_asa_game_data(current_year, team_map)
    if len(game_records) < 5:
        logger.info(f"MLS: {current_year} has only {len(game_records)} game records — trying {current_year - 1}")
        prev_records = _get_asa_game_data(current_year - 1, team_map)
        if len(prev_records) > len(game_records):
            game_records = prev_records

    # 4. Compute home/away splits (mutates season_stats)
    if game_records:
        _compute_home_away_splits(game_records, season_stats)

    # 5. Recent form (last 8 games)
    recent_form_raw = _compute_recent_form(game_records)
    # Map raw team names → season_stats names
    recent_form: Dict[str, dict] = {}
    for raw_name, form_data in recent_form_raw.items():
        matched = normalize(raw_name, season_stats) if season_stats else raw_name
        recent_form[matched] = form_data

    # 6. ESPN standings (mutates season_stats)
    _get_espn_standings(season_stats)

    # 7. Rest days
    rest_days = _get_rest_days(today, team_names)

    # 8. Injuries
    team_id_map = _get_espn_team_ids()
    injuries = _get_espn_injuries(team_id_map) if team_id_map else {}

    logger.info(
        f"MLS context: {len(season_stats)} teams in season stats, "
        f"{len(recent_form)} in recent form, "
        f"{len(injuries)} teams with injuries"
    )

    return {
        "season_stats": season_stats,
        "recent_form": recent_form,
        "rest_days": rest_days,
        "injuries": injuries,
    }
