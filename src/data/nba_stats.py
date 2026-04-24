"""Fetches NBA team stats, schedules, and back-to-back info via nba_api or direct HTTP."""
import logging
import time
import requests
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Headers required by stats.nba.com — browser-like fingerprint to avoid 403s
_NBA_STATS_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _current_season() -> str:
    today = date.today()
    year = today.year
    if today.month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


# Normalizes Odds API team names to match nba_api / NBA Stats API names
_NAME_MAP = {
    "Los Angeles Clippers": "LA Clippers",
    "Los Angeles Lakers": "Los Angeles Lakers",
}

def normalize(name: str) -> str:
    return _NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Direct HTTP helpers (primary method — more reliable than nba_api)
# ---------------------------------------------------------------------------

def _parse_nba_result_set(data: dict, result_set_index: int = 0) -> List[Dict]:
    """Parse NBA Stats API resultSets response into list of dicts."""
    rs = data["resultSets"][result_set_index]
    headers = rs["headers"]
    return [dict(zip(headers, row)) for row in rs["rowSet"]]


def _team_stats_http(season: str) -> Optional[Dict[str, Any]]:
    """Fetch team advanced stats directly from stats.nba.com."""
    try:
        url = "https://stats.nba.com/stats/leaguedashteamstats"
        params = {
            "Season": season,
            "SeasonType": "Regular Season",
            "MeasureType": "Advanced",
            "PerMode": "PerGame",
            "PlusMinus": "N",
            "PaceAdjust": "N",
            "Rank": "N",
            "Outcome": "",
            "Location": "",
            "Month": "0",
            "SeasonSegment": "",
            "DateFrom": "",
            "DateTo": "",
            "OpponentTeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "GameSegment": "",
            "Period": "0",
            "LastNGames": "0",
            "Conference": "",
            "Division": "",
        }
        r = requests.get(url, headers=_NBA_STATS_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        rows = _parse_nba_result_set(r.json())
        result = {}
        for row in rows:
            name = row.get("TEAM_NAME", "")
            if not name:
                continue
            result[name] = {
                "off_rtg": float(row.get("OFF_RATING") or 110),
                "def_rtg": float(row.get("DEF_RATING") or 110),
                "net_rtg": float(row.get("NET_RATING") or 0),
                "pace":    float(row.get("PACE") or 100),
            }
        logger.info(f"NBA advanced stats via HTTP: {len(result)} teams")
        return result if result else None
    except Exception as e:
        logger.error(f"NBA stats HTTP failed: {e}")
        return None


def _recent_team_records_http(season: str, days: int = 14) -> Dict[str, Dict]:
    """Fetch recent-form stats directly from stats.nba.com."""
    try:
        since = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y")
        today = date.today().strftime("%m/%d/%Y")
        url = "https://stats.nba.com/stats/leaguedashteamstats"
        params = {
            "Season": season,
            "SeasonType": "Regular Season",
            "MeasureType": "Advanced",
            "PerMode": "PerGame",
            "PlusMinus": "N",
            "PaceAdjust": "N",
            "Rank": "N",
            "DateFrom": since,
            "DateTo": today,
            "Outcome": "",
            "Location": "",
            "Month": "0",
            "SeasonSegment": "",
            "OpponentTeamID": "0",
            "VsConference": "",
            "VsDivision": "",
            "GameSegment": "",
            "Period": "0",
            "LastNGames": "0",
            "Conference": "",
            "Division": "",
        }
        r = requests.get(url, headers=_NBA_STATS_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        rows = _parse_nba_result_set(r.json())
        result = {}
        for row in rows:
            name = row.get("TEAM_NAME", "")
            if not name:
                continue
            result[name] = {
                "recent_net_rtg": float(row.get("NET_RATING") or 0),
                "recent_off_rtg": float(row.get("OFF_RATING") or 110),
                "recent_def_rtg": float(row.get("DEF_RATING") or 110),
                "recent_w_pct":   float(row.get("W_PCT") or 0.5),
            }
        return result
    except Exception as e:
        logger.error(f"NBA recent form HTTP failed: {e}")
        return {}


def _last_game_dates_http(season: str) -> Dict[str, date]:
    """Fetch last game date per team directly from stats.nba.com."""
    try:
        url = "https://stats.nba.com/stats/leaguegamelog"
        params = {
            "Season": season,
            "SeasonType": "Regular Season",
            "PlayerOrTeam": "T",
            "Direction": "DESC",
            "Sorter": "DATE",
            "DateFrom": "",
            "DateTo": "",
            "Counter": "0",
        }
        r = requests.get(url, headers=_NBA_STATS_HEADERS, params=params, timeout=15)
        r.raise_for_status()
        rows = _parse_nba_result_set(r.json())

        def parse_date(s: str) -> date:
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(str(s)[:20].strip(), fmt).date()
                except ValueError:
                    continue
            return date.today()

        result: Dict[str, date] = {}
        for row in rows:
            team = row.get("TEAM_NAME", "")
            game_date_raw = row.get("GAME_DATE", "")
            if not team or not game_date_raw:
                continue
            d = parse_date(str(game_date_raw))
            if team not in result or d > result[team]:
                result[team] = d
        return result
    except Exception as e:
        logger.error(f"NBA last game dates HTTP failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# nba_api fallback helpers (used only if direct HTTP fails)
# ---------------------------------------------------------------------------

def _team_stats_nba_api(season: str) -> Optional[Dict[str, Any]]:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_simple="Advanced",
        )
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        if df.empty:
            return None
        result = {}
        for _, row in df.iterrows():
            result[row["TEAM_NAME"]] = {
                "off_rtg": float(row.get("OFF_RATING") or 110),
                "def_rtg": float(row.get("DEF_RATING") or 110),
                "net_rtg": float(row.get("NET_RATING") or 0),
                "pace":    float(row.get("PACE") or 100),
            }
        return result if result else None
    except Exception as e:
        logger.warning(f"nba_api advanced stats failed: {e}")
        return None


def _recent_team_records_nba_api(season: str, days: int = 14) -> Dict[str, Dict]:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        since = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y")
        today = date.today().strftime("%m/%d/%Y")
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_simple="Advanced",
            date_from_nullable=since,
            date_to_nullable=today,
        )
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        result = {}
        for _, row in df.iterrows():
            result[row["TEAM_NAME"]] = {
                "recent_net_rtg": float(row.get("NET_RATING") or 0),
                "recent_off_rtg": float(row.get("OFF_RATING") or 110),
                "recent_def_rtg": float(row.get("DEF_RATING") or 110),
                "recent_w_pct":   float(row.get("W_PCT") or 0.5),
            }
        return result
    except Exception as e:
        logger.warning(f"nba_api recent form failed: {e}")
        return {}


def _last_game_dates_nba_api(season: str) -> Dict[str, date]:
    try:
        from nba_api.stats.endpoints import leaguegamelog
        resp = leaguegamelog.LeagueGameLog(
            season=season,
            player_or_team_abbreviation="T",
        )
        time.sleep(0.6)
        df = resp.get_data_frames()[0]

        def parse_date(d):
            s = str(d)
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%b %d, %Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s[:len(fmt)+2].strip(), fmt).date()
                except ValueError:
                    continue
            return date.today()

        df["GAME_DATE"] = df["GAME_DATE"].apply(parse_date)
        result = {}
        for team, grp in df.groupby("TEAM_NAME"):
            result[team] = grp["GAME_DATE"].max()
        return result
    except Exception as e:
        logger.warning(f"nba_api last game dates failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Public interface — tries direct HTTP first, falls back to nba_api
# ---------------------------------------------------------------------------

def _team_stats() -> Optional[Dict[str, Any]]:
    season = _current_season()
    result = _team_stats_http(season)
    if not result:
        logger.info("Falling back to nba_api for team stats...")
        result = _team_stats_nba_api(season)
    return result


def _recent_team_records(days: int = 14) -> Dict[str, Dict]:
    season = _current_season()
    result = _recent_team_records_http(season, days)
    if not result:
        result = _recent_team_records_nba_api(season, days)
    return result


def _last_game_dates() -> Dict[str, date]:
    season = _current_season()
    result = _last_game_dates_http(season)
    if not result:
        result = _last_game_dates_nba_api(season)
    return result


def get_nba_context(today: date) -> Dict:
    logger.info("Fetching NBA stats...")
    season_stats = _team_stats() or {}
    recent_form = _recent_team_records() or {}
    last_game = _last_game_dates()

    rest_days: Dict[str, int] = {}
    for team, last_date in last_game.items():
        delta = (today - last_date).days
        rest_days[team] = max(0, delta - 1)

    if season_stats:
        logger.info(f"NBA season stats loaded for {len(season_stats)} teams")
    else:
        logger.warning("NBA season stats unavailable — props and model will use defaults")

    return {
        "season_stats": season_stats,
        "recent_form": recent_form,
        "rest_days": rest_days,
    }
