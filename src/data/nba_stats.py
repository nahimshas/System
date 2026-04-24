"""Fetches NBA team stats, schedules, and back-to-back info via nba_api."""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _current_season() -> str:
    today = date.today()
    year = today.year
    if today.month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


# Normalizes Odds API team names to match nba_api names
_NAME_MAP = {
    "Los Angeles Clippers": "LA Clippers",
    "Los Angeles Lakers": "Los Angeles Lakers",
}

def normalize(name: str) -> str:
    return _NAME_MAP.get(name, name)


def _team_stats() -> Optional[Dict[str, Any]]:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=_current_season(),
            measure_type_simple="Advanced",
        )
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        result = {}
        for _, row in df.iterrows():
            result[row["TEAM_NAME"]] = {
                "off_rtg": float(row.get("OFF_RATING", 110)),
                "def_rtg": float(row.get("DEF_RATING", 110)),
                "net_rtg": float(row.get("NET_RATING", 0)),
                "pace": float(row.get("PACE", 100)),
            }
        return result
    except Exception as e:
        logger.error(f"NBA team stats error: {e}")
        return None


def _recent_team_records(days: int = 14) -> Dict[str, Dict]:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        since = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y")
        today = date.today().strftime("%m/%d/%Y")
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=_current_season(),
            measure_type_simple="Advanced",
            date_from_nullable=since,
            date_to_nullable=today,
        )
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        result = {}
        for _, row in df.iterrows():
            result[row["TEAM_NAME"]] = {
                "recent_net_rtg": float(row.get("NET_RATING", 0)),
                "recent_off_rtg": float(row.get("OFF_RATING", 110)),
                "recent_def_rtg": float(row.get("DEF_RATING", 110)),
                "recent_w_pct": float(row.get("W_PCT", 0.5)),
            }
        return result
    except Exception as e:
        logger.error(f"NBA recent form error: {e}")
        return {}


def _last_game_dates() -> Dict[str, date]:
    try:
        from nba_api.stats.endpoints import leaguegamelog
        resp = leaguegamelog.LeagueGameLog(
            season=_current_season(),
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
        logger.error(f"NBA last game dates error: {e}")
        return {}


def get_nba_context(today: date) -> Dict:
    logger.info("Fetching NBA stats...")
    season_stats = _team_stats() or {}
    recent_form = _recent_team_records() or {}
    last_game = _last_game_dates()

    rest_days: Dict[str, int] = {}
    for team, last_date in last_game.items():
        delta = (today - last_date).days
        rest_days[team] = max(0, delta - 1)

    return {
        "season_stats": season_stats,
        "recent_form": recent_form,
        "rest_days": rest_days,
    }
