"""Fetches NBA team stats, schedules, and back-to-back info via nba_api."""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — nba_api can be slow to load
def _team_stats() -> Optional[Dict[str, Any]]:
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        resp = leaguedashteamstats.LeagueDashTeamStats(
            per_mode_simple="Per100Possessions",
            season="2024-25",
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


def _schedule_for_date(target_date: date) -> List[Dict]:
    try:
        from nba_api.stats.endpoints import scoreboardv2
        date_str = target_date.strftime("%m/%d/%Y")
        resp = scoreboardv2.ScoreboardV2(game_date=date_str)
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        games = []
        for _, row in df.iterrows():
            games.append({
                "game_id": str(row.get("GAME_ID", "")),
                "home_team_id": int(row.get("HOME_TEAM_ID", 0)),
                "away_team_id": int(row.get("VISITOR_TEAM_ID", 0)),
                "home_team_name": str(row.get("HOME_TEAM_NICKNAME", "")),
                "away_team_name": str(row.get("VISITOR_TEAM_NICKNAME", "")),
            })
        return games
    except Exception as e:
        logger.error(f"NBA schedule error: {e}")
        return []


def _recent_team_records(days: int = 14) -> Dict[str, Dict]:
    """Returns win% and net rating for last N days to capture form."""
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        since = (date.today() - timedelta(days=days)).strftime("%m/%d/%Y")
        today = date.today().strftime("%m/%d/%Y")
        resp = leaguedashteamstats.LeagueDashTeamStats(
            per_mode_simple="Per100Possessions",
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
    """Returns the date of each team's most recent game (for rest days calc)."""
    try:
        from nba_api.stats.endpoints import leaguegamelog
        resp = leaguegamelog.LeagueGameLog(season="2024-25", player_or_team_abbreviation="T")
        time.sleep(0.6)
        df = resp.get_data_frames()[0]
        df["GAME_DATE"] = df["GAME_DATE"].apply(lambda d: datetime.strptime(d, "%Y-%m-%dT%H:%M:%S").date()
                                                 if "T" in str(d) else datetime.strptime(d, "%b %d, %Y").date()
                                                 if len(str(d)) > 8 else date.today())
        result = {}
        for team, grp in df.groupby("TEAM_NAME"):
            result[team] = grp["GAME_DATE"].max()
        return result
    except Exception as e:
        logger.error(f"NBA last game dates error: {e}")
        return {}


def get_nba_context(today: date) -> Dict:
    """Returns all NBA context needed for analysis: stats, form, rest."""
    logger.info("Fetching NBA stats...")
    season_stats = _team_stats() or {}
    recent_form = _recent_team_records() or {}
    last_game = _last_game_dates()

    rest_days: Dict[str, int] = {}
    for team, last_date in last_game.items():
        delta = (today - last_date).days
        rest_days[team] = max(0, delta - 1)   # 0 = played yesterday (B2B)

    return {
        "season_stats": season_stats,
        "recent_form": recent_form,
        "rest_days": rest_days,
    }
