"""Fetches MLB stats via the official MLB Stats API (no key needed) and pybaseball."""
import logging
import requests
from datetime import date
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
MLB_API = "https://statsapi.mlb.com/api/v1"


def _get(path: str, params: Dict = {}) -> Optional[Any]:
    try:
        r = requests.get(f"{MLB_API}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"MLB API error {path}: {e}")
        return None


def get_todays_games(today: date) -> List[Dict]:
    date_str = today.strftime("%Y-%m-%d")
    data = _get("/schedule", {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team,linescore"})
    if not data:
        return []
    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})
            home_pitcher = home.get("probablePitcher", {})
            away_pitcher = away.get("probablePitcher", {})
            games.append({
                "game_pk": g.get("gamePk"),
                "home_team": home.get("team", {}).get("name", ""),
                "away_team": away.get("team", {}).get("name", ""),
                "home_team_id": home.get("team", {}).get("id"),
                "away_team_id": away.get("team", {}).get("id"),
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "venue": g.get("venue", {}).get("name", ""),
            })
    logger.info(f"Found {len(games)} MLB games for {date_str}")
    return games


def get_pitcher_stats(pitcher_id: int) -> Dict:
    """Fetches season stats for a pitcher from the MLB Stats API."""
    if not pitcher_id:
        return {}
    data = _get(f"/people/{pitcher_id}/stats", {
        "stats": "season",
        "group": "pitching",
        "season": date.today().year,
    })
    if not data:
        return {}
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return {}
        s = splits[0]["stat"]
        era = float(s.get("era", "4.50").replace("-", "4.50") or "4.50")
        ip = float(s.get("inningsPitched", "0").replace("-", "0") or "0")
        k = int(s.get("strikeOuts", 0))
        bb = int(s.get("baseOnBalls", 0))
        hr = int(s.get("homeRunsAllowed", s.get("homeRuns", 0)))
        k_per_9 = (k / ip * 9) if ip > 0 else 7.0
        bb_per_9 = (bb / ip * 9) if ip > 0 else 3.0
        hr_per_9 = (hr / ip * 9) if ip > 0 else 1.2
        # Simplified FIP: (13*HR + 3*BB - 2*K) / IP + FIP_constant (~3.2)
        fip = ((13 * hr + 3 * bb - 2 * k) / ip + 3.20) if ip > 0 else era
        return {
            "era": era,
            "fip": round(fip, 2),
            "k_per_9": round(k_per_9, 1),
            "bb_per_9": round(bb_per_9, 1),
            "hr_per_9": round(hr_per_9, 2),
            "innings_pitched": ip,
        }
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"Pitcher stats parse error (id={pitcher_id}): {e}")
        return {}


def get_team_batting_stats(team_id: int) -> Dict:
    """Fetches team season batting stats."""
    if not team_id:
        return {}
    data = _get(f"/teams/{team_id}/stats", {
        "stats": "season",
        "group": "hitting",
        "season": date.today().year,
    })
    if not data:
        return {}
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return {}
        s = splits[0]["stat"]
        avg = float(s.get("avg", "0") or "0")
        obp = float(s.get("obp", "0") or "0")
        slg = float(s.get("slg", "0") or "0")
        ops = obp + slg
        runs = int(s.get("runs", 0))
        games = int(s.get("gamesPlayed", 1)) or 1
        return {
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": ops,
            "runs_per_game": round(runs / games, 2),
        }
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Team batting parse error (id={team_id}): {e}")
        return {}


def get_bullpen_stats(team_id: int) -> Dict:
    """Fetches team bullpen ERA for the season as a proxy for bullpen quality."""
    if not team_id:
        return {}
    data = _get(f"/teams/{team_id}/stats", {
        "stats": "season",
        "group": "pitching",
        "season": date.today().year,
    })
    if not data:
        return {}
    try:
        splits = data["stats"][0]["splits"]
        if not splits:
            return {}
        s = splits[0]["stat"]
        era = float(s.get("era", "4.50").replace("-", "4.50") or "4.50")
        whip = float(s.get("whip", "1.30").replace("-", "1.30") or "1.30")
        return {"bullpen_era": era, "bullpen_whip": whip}
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Bullpen stats parse error: {e}")
        return {}


# Park factors (runs scored relative to league average, >1.0 = hitter friendly)
PARK_FACTORS: Dict[str, float] = {
    "Coors Field": 1.30,
    "Great American Ball Park": 1.12,
    "Citizens Bank Park": 1.10,
    "Fenway Park": 1.08,
    "Yankee Stadium": 1.06,
    "Globe Life Field": 1.05,
    "Truist Park": 1.03,
    "American Family Field": 1.02,
    "Chase Field": 1.01,
    "Minute Maid Park": 1.00,
    "Dodger Stadium": 0.97,
    "Oracle Park": 0.94,
    "Petco Park": 0.93,
    "T-Mobile Park": 0.93,
    "Oakland Coliseum": 0.93,
}

def get_park_factor(venue: str) -> float:
    for k, v in PARK_FACTORS.items():
        if k.lower() in venue.lower() or venue.lower() in k.lower():
            return v
    return 1.00


def get_team_batting_leaders(team_id: int, top_n: int = 3, min_pa: int = 15) -> List[Dict]:
    """
    Returns the top N batters for a team sorted by OBP.
    Used to generate individual 'Hits Over (1+)' prop picks.
    min_pa is intentionally low to cover early-season data.
    """
    data = _get("/stats", {
        "stats":    "season",
        "group":    "hitting",
        "gameType": "R",
        "season":   date.today().year,
        "teamId":   team_id,
        "limit":    50,
    })
    if not data:
        return []

    players: List[Dict] = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            player = split.get("player", {})
            stat   = split.get("stat", {})
            name   = player.get("fullName", "")
            if not name:
                continue
            pa = int(stat.get("plateAppearances", 0))
            if pa < min_pa:
                continue
            try:
                avg = float(stat.get("avg", ".000").replace("-", "0") or "0")
                obp = float(stat.get("obp", ".000").replace("-", "0") or "0")
            except ValueError:
                continue
            players.append({
                "name": name,
                "id":   player.get("id"),
                "avg":  round(avg, 3),
                "obp":  round(obp, 3),
                "pa":   pa,
            })

    players.sort(key=lambda p: p["obp"], reverse=True)
    return players[:top_n]


def get_team_schedule_load(team_id: int, today: date) -> int:
    """
    Returns the number of confirmed games this team played in the last 7 days.
    Used for schedule-fatigue adjustment in the model.
    """
    if not team_id:
        return 0
    from datetime import timedelta
    seven_ago = today - timedelta(days=7)
    date_from = seven_ago.strftime("%Y-%m-%d")
    date_to   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    data = _get("/schedule", {
        "sportId": 1,
        "teamId":  team_id,
        "startDate": date_from,
        "endDate":   date_to,
        "hydrate": "linescore",
    })
    if not data:
        return 0
    count = 0
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                count += 1
    return count
