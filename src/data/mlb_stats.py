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
    data = _get("/schedule", {"sportId": 1, "date": date_str, "hydrate": "probablePitcher,team,linescore,venue"})
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
        era           = float(s.get("era", "4.50").replace("-", "4.50") or "4.50")
        ip            = float(s.get("inningsPitched", "0").replace("-", "0") or "0")
        k             = int(s.get("strikeOuts", 0))
        bb            = int(s.get("baseOnBalls", 0))
        hr            = int(s.get("homeRunsAllowed", s.get("homeRuns", 0)))
        hits_allowed  = int(s.get("hits", 0))
        games_started = int(s.get("gamesStarted", 0))
        k_per_9  = (k / ip * 9) if ip > 0 else 7.0
        bb_per_9 = (bb / ip * 9) if ip > 0 else 3.0
        hr_per_9 = (hr / ip * 9) if ip > 0 else 1.2
        whip = round((hits_allowed + bb) / ip, 2) if ip > 0 else 1.30
        avg_ip_per_start = round(ip / games_started, 1) if games_started > 0 else None
        # FIP: (13*HR + 3*BB - 2*K) / IP + FIP_constant (~3.2)
        fip = ((13 * hr + 3 * bb - 2 * k) / ip + 3.20) if ip > 0 else era

        # xFIP: like FIP but replaces actual HR with expected HR from fly-ball rate
        # airOuts ≈ fly balls + pop-ups (MLB API proxy for FB count)
        # League HR/FB rate ≈ 10 %
        air_outs = int(s.get("airOuts", 0))
        xfip = ((13 * (air_outs * 0.10) + 3 * bb - 2 * k) / ip + 3.20) if ip > 0 else fip

        # BABIP from API (e.g. ".285") — signals ERA luck when anomalously low
        babip_raw = s.get("babip")
        try:
            babip = float((babip_raw or "").replace("-", "") or "0") if babip_raw else None
        except ValueError:
            babip = None

        return {
            "era":             era,
            "fip":             round(fip, 2),
            "xfip":            round(xfip, 2),
            "whip":            whip,
            "babip":           round(babip, 3) if babip is not None else None,
            "k_per_9":         round(k_per_9, 1),
            "bb_per_9":        round(bb_per_9, 1),
            "hr_per_9":        round(hr_per_9, 2),
            "innings_pitched": ip,
            "games_started":   games_started,
            "avg_ip_per_start": avg_ip_per_start,
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
        so = int(s.get("strikeOuts", 0))
        pa = int(s.get("plateAppearances", 1)) or 1
        k_pct = round(so / pa, 3)   # team strikeout rate as batters (league avg ~0.228)
        return {
            "avg":          avg,
            "obp":          obp,
            "slg":          slg,
            "ops":          ops,
            "runs_per_game": round(runs / games, 2),
            "k_pct":        k_pct,
        }
    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Team batting parse error (id={team_id}): {e}")
        return {}


def get_bullpen_stats(team_id: int) -> Dict:
    """
    Fetches relief-only ERA/WHIP by querying individual pitcher stats for the team
    and filtering to pitchers with zero starts (pure relievers).
    This avoids the double-counting problem of using team ERA which includes starters.
    Falls back to 4.20/1.30 league averages if insufficient relief IP is found.
    """
    if not team_id:
        return {}
    data = _get("/stats", {
        "stats":      "season",
        "group":      "pitching",
        "gameType":   "R",
        "teamId":     team_id,
        "season":     date.today().year,
        "limit":      100,
        "playerPool": "ALL",   # required — without this the API silently returns only starters
    })
    if not data:
        return {}

    total_er   = 0.0
    total_ip   = 0.0
    total_hits = 0
    total_bb   = 0

    try:
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                s = split.get("stat", {})
                games_started = int(s.get("gamesStarted", 0))
                if games_started > 0:
                    continue  # skip starters and swingmen who made starts

                ip_str = (s.get("inningsPitched") or "0").replace("-", "0")
                ip = float(ip_str)
                if ip < 2:
                    continue  # skip pitchers with negligible work

                er  = float(s.get("earnedRuns", 0) or 0)
                h   = int(s.get("hits",         0) or 0)
                bb  = int(s.get("baseOnBalls",  0) or 0)
                total_er   += er
                total_ip   += ip
                total_hits += h
                total_bb   += bb

        if total_ip < 10:
            logger.warning(f"Insufficient relief IP for team {team_id} — using league averages")
            return {"bullpen_era": 4.20, "bullpen_whip": 1.30}

        era  = round(total_er / total_ip * 9, 2)
        whip = round((total_hits + total_bb) / total_ip, 2)
        return {"bullpen_era": era, "bullpen_whip": whip}

    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        logger.warning(f"Bullpen stats parse error (team {team_id}): {e}")
        return {}


# Park factors (runs scored relative to league average, >1.0 = hitter friendly)
PARK_FACTORS: Dict[str, float] = {
    # Hitter-friendly (> 1.02)
    "Coors Field": 1.30,              # COL — extreme altitude
    "Great American Ball Park": 1.12, # CIN
    "Citizens Bank Park": 1.10,       # PHI
    "Fenway Park": 1.08,              # BOS
    "Yankee Stadium": 1.07,           # NYY
    "Rogers Centre": 1.05,            # TOR — artificial turf
    "Globe Life Field": 1.05,         # TEX
    "Wrigley Field": 1.03,            # CHC
    "Truist Park": 1.03,              # ATL
    "American Family Field": 1.02,    # MIL
    # Slight hitter / neutral (0.99 – 1.02)
    "Chase Field": 1.01,              # ARI
    "Camden Yards": 1.01,             # BAL
    "Kauffman Stadium": 1.00,         # KC
    "Minute Maid Park": 1.00,         # HOU
    "Angel Stadium": 0.99,            # LAA
    "Target Field": 0.99,             # MIN
    "Nationals Park": 0.99,           # WSH
    "Busch Stadium": 0.98,            # STL
    "Citi Field": 0.97,               # NYM
    "Dodger Stadium": 0.97,           # LAD (also "UNIQLO Field at Dodger Stadium")
    # Pitcher-friendly (< 0.97)
    "Progressive Field": 0.96,        # CLE
    "PNC Park": 0.96,                 # PIT
    "Tropicana Field": 0.96,          # TB
    "Comerica Park": 0.95,            # DET
    "Guaranteed Rate Field": 0.95,    # CWS (also "Rate Field")
    "loanDepot park": 0.94,           # MIA (lower-case 'l' as ESPN/MLB API spells it)
    "Oracle Park": 0.94,              # SF
    "Petco Park": 0.93,               # SD
    "T-Mobile Park": 0.93,            # SEA
    "Oakland Coliseum": 0.93,         # OAK
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


def get_batter_props_stats(team_id: int, player_names: List[str], min_pa: int = 30) -> Dict[str, Dict]:
    """
    Fetch per-game hitting stats for specific batters for prop projections.
    Returns {player_name: {avg, obp, hits_pg, tb_pg, hr_pg, r_pg, rbi_pg, hrr_pg, pa, games}}
    """
    if not team_id or not player_names:
        return {}
    data = _get("/stats", {
        "stats": "season", "group": "hitting", "gameType": "R",
        "season": date.today().year, "teamId": team_id, "limit": 50,
    })
    if not data:
        return {}

    norm_targets = {n.lower().strip(): n for n in player_names}
    result: Dict[str, Dict] = {}

    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            player = split.get("player", {})
            stat   = split.get("stat", {})
            name   = player.get("fullName", "")
            if not name:
                continue
            norm    = name.lower().strip()
            matched = norm_targets.get(norm)
            if not matched:
                for tn, orig in norm_targets.items():
                    if all(p in norm for p in tn.split()):
                        matched = orig
                        break
            if not matched:
                continue

            pa    = int(stat.get("plateAppearances", 0))
            games = int(stat.get("gamesPlayed", 1)) or 1
            if pa < min_pa:
                continue
            try:
                avg  = float(stat.get("avg",  "0").replace("-", "0") or "0")
                obp  = float(stat.get("obp",  "0").replace("-", "0") or "0")
                hits = int(stat.get("hits",       0))
                tb   = int(stat.get("totalBases", 0))
                hr   = int(stat.get("homeRuns",   0))
                runs = int(stat.get("runs",        0))
                rbi  = int(stat.get("rbi",         0))
                so   = int(stat.get("strikeOuts",  0))
            except (ValueError, TypeError):
                continue

            k_pct = round(so / pa, 3) if pa > 0 else 0.228

            result[matched] = {
                "avg":     round(avg, 3),
                "obp":     round(obp, 3),
                "hits_pg": round(hits / games, 3),
                "tb_pg":   round(tb   / games, 3),
                "hr_pg":   round(hr   / games, 4),
                "r_pg":    round(runs / games, 3),
                "rbi_pg":  round(rbi  / games, 3),
                "hrr_pg":  round((hits + runs + rbi) / games, 3),
                "k_pct":   k_pct,
                "pa":      pa,
                "games":   games,
            }

    return result


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
