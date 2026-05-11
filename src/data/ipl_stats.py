"""
Fetches IPL (Indian Premier League) team stats via ESPN Cricinfo's public API.

Strategy:
  1. Single 30-day scoreboard scan → per-team win/loss record, rest days,
     per-venue historical stats, and head-to-head records for today's matchups.
  2. Today's scoreboard → venue name for each upcoming match.
  3. ipl_venue_config.json → static dew risk / boundary / home-adv config,
     reviewed and updated once per season each March.

ESPN sport path: cricket/ipl
"""
import json
import logging
import time
import requests
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ESPN_IPL = "https://site.api.espn.com/apis/site/v2/sports/cricket/ipl"

_VENUE_CONFIG_PATH = Path(__file__).parent / "ipl_venue_config.json"

# Odds API team name → ESPN Cricinfo displayName (only entries that differ)
_ODDS_TO_ESPN: Dict[str, str] = {
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "RCB":                         "Royal Challengers Bengaluru",
    "Kings XI Punjab":             "Punjab Kings",
    "Delhi Daredevils":            "Delhi Capitals",
}


def normalize(name: str) -> str:
    """Convert Odds API team name → ESPN displayName (key used in all ctx dicts)."""
    return _ODDS_TO_ESPN.get(name, name)


def _get(url: str, params: dict = None) -> Optional[dict]:
    """GET → JSON or None on any error."""
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"ESPN IPL GET failed [{url}]: {e}")
        return None


def _load_venue_config() -> Dict[str, Dict]:
    """Load static venue config from JSON. Returns {} on failure."""
    try:
        with open(_VENUE_CONFIG_PATH) as f:
            raw = json.load(f)
        # Strip metadata keys that start with "_"
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning(f"IPL venue config load failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Today's match venues — one API call to get upcoming match venue names
# ---------------------------------------------------------------------------

def _fetch_todays_venues(today: date) -> Dict[Tuple[str, str], str]:
    """
    Returns {(home_display, away_display): venue_fullName} for today's matches.
    Checks both completed and upcoming events (no status filter).
    """
    data = _get(f"{ESPN_IPL}/scoreboard", params={"dates": today.strftime("%Y%m%d")})
    if not data:
        return {}

    venues: Dict[Tuple[str, str], str] = {}
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Extract venue — ESPN cricket events carry a "venues" array
        venue_name = ""
        event_venues = event.get("venues", [])
        if event_venues:
            venue_name = event_venues[0].get("fullName", "")

        # Identify home and away
        home_name = away_name = ""
        for c in competitors:
            team_name = c.get("team", {}).get("displayName", "")
            if c.get("homeAway", "") == "home":
                home_name = team_name
            else:
                away_name = team_name

        if home_name and away_name and venue_name:
            venues[(home_name, away_name)] = venue_name

    return venues


# ---------------------------------------------------------------------------
# Single 30-day scan — form, rest days, venue stats, H2H
# ---------------------------------------------------------------------------

def _fetch_season_data(
    today: date,
    matchups: List[Tuple[str, str]],
) -> Dict:
    """
    One pass over the last 30 days of completed IPL scoreboard events.

    Collects:
      - season_form:  {team: {win_pct, wins, losses, total, recent_win_pct, recent_total}}
      - rest_days:    {team: int}  days since last completed match
      - venue_stats:  {venue: {total_matches, home_wins, home_win_pct, avg_score}}
      - h2h:          {(home, away): {home_wins, away_wins, total, home_h2h_pct}}

    Two form windows:
      - Season:  all completed matches in last 30 days
      - Recent:  last 7 days only (hot/cold streak signal)
    """
    team_season:     Dict[str, list] = {}
    team_recent:     Dict[str, list] = {}
    team_last_match: Dict[str, int]  = {}
    venue_data:      Dict[str, Dict] = {}
    h2h_raw:         Dict[Tuple[str, str], Dict] = {m: {"home_wins": 0, "away_wins": 0, "total": 0} for m in matchups}

    for delta in range(1, 31):
        d = today - timedelta(days=delta)
        data = _get(f"{ESPN_IPL}/scoreboard", params={"dates": d.strftime("%Y%m%d")})
        if not data:
            time.sleep(0.05)
            continue

        for event in data.get("events", []):
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]

            if not comp.get("status", {}).get("type", {}).get("completed"):
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            venue_name = ""
            event_venues = event.get("venues", [])
            if event_venues:
                venue_name = event_venues[0].get("fullName", "")

            try:
                home_name = away_name = ""
                home_score = away_score = 0
                for c in competitors:
                    team_name = c.get("team", {}).get("displayName", "")
                    score_str = c.get("score", "0")
                    runs = int(str(score_str).split("/")[0] or 0)
                    if c.get("homeAway", "") == "home":
                        home_name  = team_name
                        home_score = runs
                    else:
                        away_name  = team_name
                        away_score = runs

                if not home_name or not away_name:
                    continue

                home_won = home_score > away_score
                margin   = home_score - away_score

                # --- Team form ---
                for name, won, sgn in [
                    (home_name, home_won,  margin),
                    (away_name, not home_won, -margin),
                ]:
                    team_season.setdefault(name, []).append((sgn, won))
                    if delta <= 7:
                        team_recent.setdefault(name, []).append((sgn, won))
                    team_last_match.setdefault(name, delta)

                # --- Venue stats ---
                if venue_name:
                    vd = venue_data.setdefault(venue_name, {
                        "total_matches": 0,
                        "home_wins": 0,
                        "score_sum": 0,
                    })
                    vd["total_matches"] += 1
                    vd["score_sum"]     += max(home_score, away_score)
                    if home_won:
                        vd["home_wins"] += 1

                # --- H2H ---
                for matchup in matchups:
                    mh, ma = matchup
                    if (home_name == mh and away_name == ma):
                        h2h_raw[matchup]["total"]    += 1
                        h2h_raw[matchup]["home_wins" if home_won else "away_wins"] += 1
                    elif (home_name == ma and away_name == mh):
                        # Reversed fixture — treat historical home team as "home" in that game
                        h2h_raw[matchup]["total"] += 1
                        # home_won here means the *historical* home team (== ma in our matchup) won
                        # which is the away side from our matchup's perspective
                        if home_won:
                            h2h_raw[matchup]["away_wins"] += 1
                        else:
                            h2h_raw[matchup]["home_wins"] += 1

            except Exception:
                continue

        time.sleep(0.05)

    # --- Build season_form ---
    season_form: Dict[str, Dict] = {}
    for name in set(team_season) | set(team_recent):
        se = team_season.get(name, [])
        re = team_recent.get(name, [])
        s_total = len(se)
        s_wins  = sum(1 for _, w in se if w)
        r_total = len(re)
        r_wins  = sum(1 for _, w in re if w)
        s_wpct  = s_wins / s_total if s_total else 0.5
        season_form[name] = {
            "win_pct":        s_wpct,
            "wins":           s_wins,
            "losses":         s_total - s_wins,
            "total":          s_total,
            "recent_win_pct": r_wins / r_total if r_total else s_wpct,
            "recent_total":   r_total,
        }

    logger.info(f"IPL form: {len(season_form)} teams from last 30 days of scoreboard")

    # --- Build venue_stats ---
    venue_stats: Dict[str, Dict] = {}
    for venue, vd in venue_data.items():
        n = vd["total_matches"]
        if n > 0:
            venue_stats[venue] = {
                "total_matches":  n,
                "home_wins":      vd["home_wins"],
                "home_win_pct":   vd["home_wins"] / n,
                "avg_score":      round(vd["score_sum"] / n, 1),
            }

    # --- Build h2h ---
    h2h: Dict[Tuple[str, str], Dict] = {}
    for matchup, hd in h2h_raw.items():
        if hd["total"] > 0:
            h2h[matchup] = {
                **hd,
                "home_h2h_pct": hd["home_wins"] / hd["total"],
            }

    return {
        "season_form": season_form,
        "rest_days":   team_last_match,
        "venue_stats": venue_stats,
        "h2h":         h2h,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_ipl_context(
    today: date,
    team_names: List[str] = None,
    matchups: List[Tuple[str, str]] = None,
) -> Dict:
    """
    Returns full IPL context dict for edge_finder:
    {
        "season_form":  {display_name: {win_pct, wins, losses, total,
                                        recent_win_pct, recent_total}},
        "rest_days":    {display_name: int},
        "venue_stats":  {venue_name: {total_matches, home_wins, home_win_pct, avg_score}},
        "venue_config": {venue_name: {dew_risk, boundary_rating, home_adv_modifier, ...}},
        "match_venues": {(home_display, away_display): venue_name},
        "h2h":          {(home_display, away_display): {home_wins, away_wins, total, home_h2h_pct}},
    }

    matchups — list of (home_display_name, away_display_name) in ESPN normalized form,
               used to scope H2H lookups. Pass [] or omit if not needed.
    """
    team_names = team_names or []
    matchups   = matchups   or []

    # Normalize matchup team names
    norm_matchups = [(normalize(h), normalize(a)) for h, a in matchups]

    venue_config = _load_venue_config()

    try:
        season_data = _fetch_season_data(today, norm_matchups)
    except Exception as e:
        logger.error(f"IPL season data fetch failed: {e}")
        season_data = {
            "season_form": {},
            "rest_days":   {},
            "venue_stats": {},
            "h2h":         {},
        }

    # Fill default rest days for any team not seen in last 30 days
    rest_days = season_data["rest_days"]
    for name in [normalize(n) for n in team_names]:
        rest_days.setdefault(name, 3)

    try:
        match_venues = _fetch_todays_venues(today)
    except Exception as e:
        logger.warning(f"IPL today's venues fetch failed: {e}")
        match_venues = {}

    return {
        "season_form":  season_data["season_form"],
        "rest_days":    rest_days,
        "venue_stats":  season_data["venue_stats"],
        "venue_config": venue_config,
        "match_venues": match_venues,
        "h2h":          season_data["h2h"],
    }
