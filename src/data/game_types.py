"""
Per-game season-type detection — replaces the hardcoded playoff calendars.

ESPN's scoreboard events carry a per-game `season.type` (verified live:
1 = preseason, 2 = regular season, 3 = postseason, 5 = NBA play-in), present
before games start, from feeds the system already depends on for settlement.
Per-game classification beats any calendar — even a perfectly maintained one —
because mixed days exist (mid-April NBA has regular, play-in, and playoff
games inside the same week) and schedules shift (Olympic breaks, lockouts).

Canonical types stamped onto game dicts as `season_game_type`:

  "exhibition"  — preseason / All-Star / offseason events → SKIPPED entirely
                  (rosters and effort are meaningless; no model applies)
  "regular"     — regular season → normal treatment
  "play_in"     — NBA play-in → playoff treatment, tagged separately in the
                  shadow log so CLV-per-type can later justify (or kill)
                  finer distinctions with evidence
  "postseason"  — real playoffs → playoff treatment
  "superbowl"   — postseason + neutral site → playoff treatment with home
                  advantage zeroed

Failure mode: any error leaves games unstamped, and the analyzers fall back
to the legacy calendar windows (`_is_*_playoff()` in edge_finder) — worst
case is exactly the pre-deployment behavior. Never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# Sport slug (registry) → ESPN path. Only sports whose feeds expose season.type
# are listed; others (IPL via Cricbuzz, WC single-tournament) keep their
# existing date-based logic.
_ESPN_PATHS = {
    "nba":  "basketball/nba",
    "mlb":  "baseball/mlb",
    "nfl":  "football/nfl",
    "nhl":  "hockey/nhl",
    "wnba": "basketball/wnba",
    "mls":  "soccer/usa.1",
}

# ESPN season.type → canonical label. Unknown types map to None (no stamp →
# calendar fallback) rather than guessing.
_TYPE_MAP = {
    1: "exhibition",    # preseason
    2: "regular",
    3: "postseason",
    4: "exhibition",    # offseason events (All-Star etc.)
    5: "play_in",       # NBA play-in tournament
}

# Process-level cache: (slug, YYYYMMDD) → list of event dicts
_cache: Dict[Tuple[str, str], List[Dict]] = {}


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except Exception:
        return None


def _fetch_events(slug: str, date_str_compact: str) -> List[Dict]:
    """Fetch ESPN events for a sport+date; returns [] on any failure."""
    key = (slug, date_str_compact)
    if key in _cache:
        return _cache[key]
    path = _ESPN_PATHS.get(slug)
    if not path:
        _cache[key] = []
        return []
    try:
        r = requests.get(
            f"{ESPN_BASE}/{path}/scoreboard",
            params={"dates": date_str_compact}, timeout=15,
        )
        r.raise_for_status()
        events = r.json().get("events", []) or []
    except Exception as e:
        logger.warning(f"Game-type fetch failed ({slug} {date_str_compact}): {e}")
        events = []
    _cache[key] = events
    return events


def _canonical_type(event: Dict) -> Optional[str]:
    """Map one ESPN event to a canonical game type (None = unknown)."""
    stype = (event.get("season") or {}).get("type")
    label = _TYPE_MAP.get(stype)
    if label != "postseason":
        return label
    # Postseason refinement: the Super Bowl is the one postseason game played
    # at a neutral site — home advantage must be zeroed.
    try:
        notes = (event.get("competitions", [{}])[0].get("notes") or [{}])
        headline = (notes[0].get("headline") or "").lower()
        if "super bowl" in headline:
            return "superbowl"
    except Exception:
        pass
    return "postseason"


def _team_match(query: str, candidate: str) -> bool:
    """Fuzzy match using the settlement matcher (aliases + token overlap)."""
    q, c = (query or "").lower(), (candidate or "").lower()
    if not q or not c:
        return False
    if q in c or c in q:
        return True
    try:
        from src.data.outcome_checker import _team_token_overlap_match
        return _team_token_overlap_match(query, candidate)
    except Exception:
        return False


def stamp_game_types(games: List[Dict], slug: str, today: str) -> int:
    """
    Stamp `season_game_type` (and `season_game_note` when ESPN provides a
    headline) onto each game dict from the Odds API. Matches by team names +
    commence-time proximity. Games that can't be matched stay unstamped →
    analyzers use the legacy calendar fallback.

    `today` is "YYYY-MM-DD". Looks at today and tomorrow's scoreboards
    (evening games cross the UTC date line). Returns the number of games
    stamped. Never raises.
    """
    stamped = 0
    try:
        if slug not in _ESPN_PATHS or not games:
            return 0
        base = datetime.fromisoformat(today)
        events: List[Dict] = []
        for d in (base, base + timedelta(days=1)):
            events.extend(_fetch_events(slug, d.strftime("%Y%m%d")))
        if not events:
            return 0

        for game in games:
            g_home = game.get("home_team", "")
            g_away = game.get("away_team", "")
            g_dt   = _parse_iso(game.get("commence_time", ""))
            for ev in events:
                comps = ev.get("competitions", [{}])
                competitors = comps[0].get("competitors", []) if comps else []
                e_home = next((c.get("team", {}).get("displayName", "")
                               for c in competitors if c.get("homeAway") == "home"), "")
                e_away = next((c.get("team", {}).get("displayName", "")
                               for c in competitors if c.get("homeAway") == "away"), "")
                if not ((_team_match(g_home, e_home) and _team_match(g_away, e_away))
                        or (_team_match(g_home, e_away) and _team_match(g_away, e_home))):
                    continue
                ev_dt = _parse_iso(ev.get("date", ""))
                if g_dt and ev_dt and abs((ev_dt - g_dt).total_seconds()) > 6 * 3600:
                    continue
                gtype = _canonical_type(ev)
                if gtype:
                    game["season_game_type"] = gtype
                    try:
                        headline = (ev.get("competitions", [{}])[0].get("notes") or [{}])[0].get("headline", "")
                        if headline:
                            game["season_game_note"] = headline
                    except Exception:
                        pass
                    stamped += 1
                break

        if stamped:
            types = {g.get("season_game_type") for g in games if g.get("season_game_type")}
            logger.info(f"Game types ({slug}): {stamped}/{len(games)} stamped — {sorted(types)}")
        return stamped
    except Exception as e:
        logger.warning(f"Game-type stamping failed ({slug}, non-fatal): {e}")
        return 0
