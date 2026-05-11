"""
Fetches IPL (Indian Premier League) match data via Cricbuzz HTML scraping.

Strategy:
  1. Auto-discover the current season's series ID from the Cricbuzz live-scores page.
  2. Fetch the series matches page, which embeds all match data (teams, venue, scores,
     innings order) in Next.js push calls — no API key required.
  3. Derive per-team form, rest days, venue stats, chasing win%, NRR proxy, and H2H
     from the completed matches. Also extract today's upcoming match venues.

The ESPN Cricinfo site.api endpoint for IPL (/sports/cricket/ipl/scoreboard) is
permanently unavailable (404) outside the active season window. Cricbuzz is used
as the reliable free replacement.

Cricbuzz series IDs change each year; auto-discovery handles this automatically.
"""
import logging
import json
import re
import time
import requests
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CRICBUZZ_BASE      = "https://www.cricbuzz.com"
_VENUE_CONFIG_PATH = Path(__file__).parent / "ipl_venue_config.json"
_UNAVAIL_PATH      = Path(__file__).parent / "ipl_unavailabilities.json"

# Odds API / ESPN team name → canonical display name used as dict key everywhere.
_ODDS_TO_ESPN: Dict[str, str] = {
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "RCB":                         "Royal Challengers Bengaluru",
    "Kings XI Punjab":             "Punjab Kings",
    "Delhi Daredevils":            "Delhi Capitals",
}

# Cricbuzz team names that differ from canonical. Add as discovered.
_CRICBUZZ_TO_CANONICAL: Dict[str, str] = {
    "Royal Challengers Bengaluru": "Royal Challengers Bengaluru",
}

# Venue city → home team (canonical name). Used to determine home side per match.
_CITY_TO_TEAM: Dict[str, str] = {
    "Mumbai":     "Mumbai Indians",
    "Chennai":    "Chennai Super Kings",
    "Bengaluru":  "Royal Challengers Bengaluru",
    "Bangalore":  "Royal Challengers Bengaluru",
    "Kolkata":    "Kolkata Knight Riders",
    "Delhi":      "Delhi Capitals",
    "Hyderabad":  "Sunrisers Hyderabad",
    "Mohali":     "Punjab Kings",
    "Chandigarh": "Punjab Kings",
    "Jaipur":     "Rajasthan Royals",
    "Ahmedabad":  "Gujarat Titans",
    "Lucknow":    "Lucknow Super Giants",
}

# Regex to locate the start of each IPL matchInfo block.
_MATCHINFO_ANCHOR_RE = re.compile(r'"matchInfo":\{"matchId":(\d+),"seriesId":9241')

# Score regex used within a single-match chunk (matchInfo→next matchInfo).
_SCORE_RE = re.compile(
    r'"team1Score":\{"inngs1":\{"inningsId":1,"runs":(\d+),"wickets":(\d+)'
    r'[^}]+\}\},"team2Score":\{"inngs1":\{"inningsId":2,"runs":(\d+),"wickets":(\d+)'
)

# Module-level cache so we only discover/fetch once per pipeline run.
_series_cache:  Dict[str, str]  = {}   # {year_str: slug}  e.g. {"2026": "ipl-2026"}
_matches_cache: Optional[List]  = None


def normalize(name: str) -> str:
    """Convert Odds API or Cricbuzz team name → canonical key used in all ctx dicts."""
    return _ODDS_TO_ESPN.get(name, _CRICBUZZ_TO_CANONICAL.get(name, name))


def _get_html(url: str, **kwargs) -> Optional[str]:
    """GET → HTML/text or None on any error."""
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
            timeout=15,
            **kwargs,
        )
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error(f"Cricbuzz GET failed [{url}]: {e}")
        return None


def _load_venue_config() -> Dict[str, Dict]:
    """Load static venue config from JSON. Returns {} on failure."""
    try:
        with open(_VENUE_CONFIG_PATH) as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception as e:
        logger.warning(f"IPL venue config load failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Series ID auto-discovery
# ---------------------------------------------------------------------------

def _discover_series_id(year: int) -> Optional[str]:
    """
    Fetch the Cricbuzz live-scores page and find the series ID for IPL {year}.
    Caches (series_id, slug) so subsequent calls are free.
    Returns the series ID string, or None if not found.
    """
    year_key = str(year)
    if year_key in _series_cache:
        return _series_cache[year_key]["id"]

    html = _get_html(f"{CRICBUZZ_BASE}/live-cricket-scores")
    if not html:
        return None

    # Links are like /cricket-series/9241/ipl-2026
    for y in [year, year - 1, year + 1]:
        hits = re.findall(rf"/cricket-series/(\d+)/(ipl-{y})\b", html)
        if hits:
            sid, slug = hits[0]
            _series_cache[year_key] = {"id": sid, "slug": slug}
            logger.info(f"IPL series {year}: id={sid} slug={slug}")
            return sid

    logger.warning(f"Could not discover Cricbuzz IPL series ID for {year}")
    return None


# ---------------------------------------------------------------------------
# Match data extraction from Cricbuzz series page
# ---------------------------------------------------------------------------

def _extract_payload(html: str) -> str:
    """
    Pull the Next.js push payload from the series matches page.
    Cricbuzz embeds data as: self.__next_f.push([1,"<escaped-json>"])
    Returns the unescaped concatenated payload, or "".
    """
    pushes = re.findall(r'self\.__next_f\.push\(\[1,"(.+?)"\]\)', html, re.DOTALL)
    parts = []
    for p in pushes:
        if "Indian Premier League" in p:
            try:
                parts.append(json.loads(f'"{p}"'))
            except Exception:
                parts.append(p.replace('\\"', '"').replace('\\\\', '\\'))
    return "\n".join(parts)


def _fetch_series_matches(series_id: str) -> List[Dict]:
    """
    Parse all IPL matches from the Cricbuzz series matches page.

    Each returned dict:
      {match_id, state, status, team1, team2, ground, city,
       start_ms, team1_runs, team1_wkts, team2_runs, team2_wkts}
    """
    global _matches_cache
    if _matches_cache is not None:
        return _matches_cache

    # Use the cached slug if available; fall back to constructing from current year.
    slug = next(
        (v["slug"] for v in _series_cache.values() if v["id"] == series_id),
        None,
    )
    if not slug:
        from datetime import date as _date
        slug = f"ipl-{_date.today().year}"

    html = _get_html(f"{CRICBUZZ_BASE}/cricket-series/{series_id}/{slug}/matches")
    if not html:
        _matches_cache = []
        return []

    payload = _extract_payload(html)
    if not payload:
        _matches_cache = []
        return []

    # Chunk-based extraction: slice payload between consecutive matchInfo anchors.
    # This avoids brittle combined regexes that fail on nested JSON fields.
    anchor_positions = [
        (m.group(1), m.start())
        for m in _MATCHINFO_ANCHOR_RE.finditer(payload)
    ]

    matches = []
    seen: set = set()
    for i, (mid, pos) in enumerate(anchor_positions):
        if mid in seen:
            continue
        seen.add(mid)

        next_pos = anchor_positions[i + 1][1] if i + 1 < len(anchor_positions) else len(payload)
        chunk = payload[pos:next_pos]

        # Skip non-T20 formats (playoff warm-up games etc.)
        if '"matchFormat":"T20"' not in chunk:
            continue

        def _field(pattern: str, grp: int = 1) -> Optional[str]:
            m = re.search(pattern, chunk)
            return m.group(grp) if m else None

        state    = _field(r'"state":"(Complete|Preview|Live)"')
        status   = _field(r'"status":"([^"]+)"')
        t1       = _field(r'"team1":\{"teamId":\d+,"teamName":"([^"]+)"')
        t2       = _field(r'"team2":\{"teamId":\d+,"teamName":"([^"]+)"')
        ground   = _field(r'"ground":"([^"]+)"')
        city     = _field(r'"city":"([^"]+)"')
        start_ms = _field(r'"startDate":"?(\d+)')

        if not all([state, status, t1, t2, ground, city, start_ms]):
            continue

        entry: Dict = {
            "match_id":  mid,
            "state":     state,
            "status":    status,
            "team1":     normalize(t1),
            "team2":     normalize(t2),
            "ground":    ground,
            "city":      city,
            "start_ms":  int(start_ms),
        }

        # Extract innings scores (only present for Complete matches)
        sm = _SCORE_RE.search(chunk)
        if sm:
            entry.update({
                "team1_runs": int(sm.group(1)), "team1_wkts": int(sm.group(2)),
                "team2_runs": int(sm.group(3)), "team2_wkts": int(sm.group(4)),
            })

        matches.append(entry)

    logger.info(f"IPL: {len(matches)} matches parsed from Cricbuzz (series {series_id})")
    _matches_cache = matches
    return matches


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_winner(status: str, team1: str, team2: str) -> Optional[str]:
    """
    Extract winner from Cricbuzz status string.
    "Royal Challengers Bengaluru won by 2 wkts" → "Royal Challengers Bengaluru"
    Returns canonical team name, or None if match wasn't completed normally.
    """
    lower = status.lower()
    if "won by" not in lower:
        return None  # Tie, no result, abandoned, etc.
    # The winner is the team named before "won by"
    for team in (team1, team2):
        if status.startswith(team):
            return team
    # Fallback: check if either team name appears anywhere before "won by"
    idx = lower.find("won by")
    prefix = status[:idx].strip()
    for team in (team1, team2):
        if team.lower() in prefix.lower():
            return team
    return None


def _home_team(city: str, team1: str, team2: str) -> str:
    """
    Return the home team name for a match.
    Uses city → team mapping; falls back to team1 for neutral venues.
    """
    home = _CITY_TO_TEAM.get(city, "")
    if home in (team1, team2):
        return home
    return team1  # neutral venue: treat team1 as home (schedule convention)


def _match_date_ist(start_ms: int) -> date:
    """Convert Cricbuzz startDate (Unix ms, UTC) to calendar date in IST (UTC+5:30)."""
    utc_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    ist_dt  = utc_dt + timedelta(hours=5, minutes=30)
    return ist_dt.date()


def _is_night_match(start_ms: int) -> bool:
    """Return True if the match starts at or after 17:00 IST (evening match)."""
    utc_dt = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
    ist_hour = (utc_dt + timedelta(hours=5, minutes=30)).hour
    return ist_hour >= 17


# ---------------------------------------------------------------------------
# Season data computation
# ---------------------------------------------------------------------------

def _fetch_season_data(
    today: date,
    matchups: List[Tuple[str, str]],
    series_id: str,
) -> Dict:
    """
    Build all derived stats from completed IPL matches this season.

    Returns:
      season_form  — {team: {win_pct, wins, losses, total, recent_win_pct,
                              recent_total, avg_margin}}
      rest_days    — {team: int}  days since last completed match
      venue_stats  — {ground: {total_matches, home_wins, home_win_pct,
                                avg_first_innings, chasing_wins, chasing_win_pct}}
      h2h          — {(home, away): {home_wins, away_wins, total, home_h2h_pct}}
    """
    matches = _fetch_series_matches(series_id)

    team_season:     Dict[str, list] = {}   # list of (signed_margin, won)
    team_recent:     Dict[str, list] = {}   # last 7 days
    team_last_match: Dict[str, int]  = {}   # days since last match
    venue_data:      Dict[str, Dict] = {}
    h2h_raw:         Dict[Tuple[str, str], Dict] = {
        m: {"home_wins": 0, "away_wins": 0, "total": 0} for m in matchups
    }

    for match in matches:
        if match["state"] != "Complete":
            continue

        team1    = match["team1"]
        team2    = match["team2"]
        ground   = match["ground"]
        city     = match["city"]
        start_ms = match["start_ms"]

        winner = _parse_winner(match["status"], team1, team2)
        if winner is None:
            continue  # abandoned / no result

        home_team = _home_team(city, team1, team2)
        away_team = team2 if home_team == team1 else team1
        home_won  = (winner == home_team)

        match_date = _match_date_ist(start_ms)
        days_ago   = (today - match_date).days
        if days_ago < 0:
            continue  # future match somehow in list

        # ── Run/wicket margins for NRR proxy ─────────────────────────────
        t1r = match.get("team1_runs", 0)
        t2r = match.get("team2_runs", 0)
        # team1 always bats first (inningsId 1); team2 chases (inningsId 2)
        first_innings_runs = t1r
        chase_successful   = (winner == team2)   # team2 is always the chasing side

        # Signed run margin from chasing team's perspective:
        # +ve = chasing team won (by wickets); -ve = defending team won (by runs)
        run_margin = t2r - t1r  # +ve if chaser won, -ve if defender won

        # ── Form records ─────────────────────────────────────────────────
        for team in (team1, team2):
            team_won     = (winner == team)
            # use run_margin from that team's perspective
            signed = run_margin if team == team2 else -run_margin
            team_season.setdefault(team, []).append((signed, team_won))
            if days_ago <= 7:
                team_recent.setdefault(team, []).append((signed, team_won))
            team_last_match.setdefault(team, days_ago)

        # ── Venue stats ───────────────────────────────────────────────────
        vd = venue_data.setdefault(ground, {
            "total_matches":   0,
            "home_wins":       0,
            "first_inns_sum":  0,
            "chasing_wins":    0,
        })
        vd["total_matches"]  += 1
        vd["home_wins"]      += int(home_won)
        if first_innings_runs:
            vd["first_inns_sum"] += first_innings_runs
        vd["chasing_wins"]   += int(chase_successful)

        # ── H2H ──────────────────────────────────────────────────────────
        for matchup in matchups:
            mh, ma = matchup
            # Match this fixture regardless of which side was team1/team2 historically
            if {home_team, away_team} == {mh, ma}:
                h2h_raw[matchup]["total"] += 1
                if winner == mh:
                    h2h_raw[matchup]["home_wins"] += 1
                else:
                    h2h_raw[matchup]["away_wins"] += 1

    # ── Build season_form ────────────────────────────────────────────────
    season_form: Dict[str, Dict] = {}
    for team in set(team_season) | set(team_recent):
        se = team_season.get(team, [])
        re_ = team_recent.get(team, [])
        s_total = len(se)
        s_wins  = sum(1 for _, w in se if w)
        r_total = len(re_)
        r_wins  = sum(1 for _, w in re_ if w)
        s_wpct  = s_wins / s_total if s_total else 0.5
        avg_margin = (sum(m for m, _ in se) / s_total) if s_total else 0.0
        season_form[team] = {
            "win_pct":        s_wpct,
            "wins":           s_wins,
            "losses":         s_total - s_wins,
            "total":          s_total,
            "recent_win_pct": r_wins / r_total if r_total else s_wpct,
            "recent_total":   r_total,
            "avg_margin":     round(avg_margin, 1),
        }

    # ── Build venue_stats ────────────────────────────────────────────────
    venue_stats: Dict[str, Dict] = {}
    for ground, vd in venue_data.items():
        n = vd["total_matches"]
        if n > 0:
            venue_stats[ground] = {
                "total_matches":    n,
                "home_wins":        vd["home_wins"],
                "home_win_pct":     vd["home_wins"] / n,
                "avg_first_innings": round(vd["first_inns_sum"] / n, 1) if vd["first_inns_sum"] else None,
                "chasing_wins":     vd["chasing_wins"],
                "chasing_win_pct":  vd["chasing_wins"] / n,
            }

    # ── Build H2H ────────────────────────────────────────────────────────
    h2h: Dict[Tuple[str, str], Dict] = {}
    for matchup, hd in h2h_raw.items():
        if hd["total"] > 0:
            h2h[matchup] = {**hd, "home_h2h_pct": hd["home_wins"] / hd["total"]}

    return {
        "season_form": season_form,
        "rest_days":   team_last_match,
        "venue_stats": venue_stats,
        "h2h":         h2h,
    }


# ---------------------------------------------------------------------------
# Today's match venues
# ---------------------------------------------------------------------------

def _fetch_todays_venues(today: date, matches: List[Dict]) -> Dict[Tuple[str, str], str]:
    """
    Extract venue ground name for each of today's matches (Preview or Live).
    Returns {(home_canonical, away_canonical): ground_name}.
    """
    venues: Dict[Tuple[str, str], str] = {}
    for match in matches:
        if match["state"] not in ("Preview", "Live"):
            continue
        match_date = _match_date_ist(match["start_ms"])
        if match_date != today:
            continue
        city      = match["city"]
        team1     = match["team1"]
        team2     = match["team2"]
        home      = _home_team(city, team1, team2)
        away      = team2 if home == team1 else team1
        ground    = match["ground"]
        venues[(home, away)] = ground
    return venues


# ---------------------------------------------------------------------------
# Player unavailabilities
# ---------------------------------------------------------------------------

def _load_unavailabilities(today: date) -> Dict[str, List[str]]:
    """
    Load confirmed player absences from the static config file.
    Checks today + next 2 days so a pipeline run on May 10 picks up
    absences declared for a May 11 match.
    Returns {canonical_team_name: [player_name, ...]}.
    """
    try:
        with open(_UNAVAIL_PATH) as f:
            raw = json.load(f)
    except Exception as e:
        logger.warning(f"IPL unavailabilities load failed: {e}")
        return {}

    result: Dict[str, List[str]] = {}
    for delta in range(3):
        key = (today + timedelta(days=delta)).isoformat()
        day_data = raw.get(key, {})
        if not isinstance(day_data, dict):
            continue
        for team, players in day_data.items():
            if team.startswith("_") or not isinstance(players, list):
                continue
            canon = normalize(team)
            for p in players:
                if p and p not in result.get(canon, []):
                    result.setdefault(canon, []).append(p)

    if result:
        logger.info(f"IPL unavailabilities loaded: {result}")
    return result


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
                                      recent_win_pct, recent_total, avg_margin}},
      "rest_days":    {display_name: int},
      "venue_stats":  {ground: {total_matches, home_wins, home_win_pct,
                                avg_first_innings, chasing_wins, chasing_win_pct}},
      "venue_config": {ground: {dew_risk, boundary_rating, home_adv_modifier, ...}},
      "match_venues": {(home_display, away_display): ground_name},
      "h2h":          {(home_display, away_display): {home_wins, away_wins,
                                                       total, home_h2h_pct}},
      "match_flags":  {(home_display, away_display): {is_night: bool}},
    }
    """
    team_names = team_names or []
    matchups   = matchups   or []
    norm_matchups = [(normalize(h), normalize(a)) for h, a in matchups]

    venue_config    = _load_venue_config()
    unavailabilities = _load_unavailabilities(today)

    series_id = _discover_series_id(today.year)
    if not series_id:
        # Try previous year (IPL sometimes starts in March but discovery may lag)
        series_id = _discover_series_id(today.year - 1)

    if not series_id:
        logger.error("IPL: could not discover Cricbuzz series ID — returning empty context")
        return _empty_context(venue_config, team_names)

    try:
        season_data = _fetch_season_data(today, norm_matchups, series_id)
    except Exception as e:
        logger.error(f"IPL season data fetch failed: {e}")
        season_data = {"season_form": {}, "rest_days": {}, "venue_stats": {}, "h2h": {}}

    # Ensure every team playing today has a rest_days entry
    rest_days = season_data["rest_days"]
    for name in [normalize(n) for n in team_names]:
        rest_days.setdefault(name, 3)

    # Today's match venues + night-match flags
    all_matches  = _matches_cache or []
    match_venues: Dict[Tuple[str, str], str]  = {}
    match_flags:  Dict[Tuple[str, str], Dict] = {}
    try:
        match_venues = _fetch_todays_venues(today, all_matches)
        for (home, away), ground in match_venues.items():
            # Find this match's start_ms for day/night detection
            for m in all_matches:
                if m["state"] in ("Preview", "Live") and m["ground"] == ground:
                    if {m["team1"], m["team2"]} == {home, away}:
                        match_flags[(home, away)] = {"is_night": _is_night_match(m["start_ms"])}
                        break
    except Exception as e:
        logger.warning(f"IPL today's venues fetch failed: {e}")

    return {
        "season_form":     season_data["season_form"],
        "rest_days":       rest_days,
        "venue_stats":     season_data["venue_stats"],
        "venue_config":    venue_config,
        "match_venues":    match_venues,
        "h2h":             season_data["h2h"],
        "match_flags":     match_flags,
        "unavailabilities": unavailabilities,
    }


def get_ipl_completed_matches(today: date) -> List[Dict]:
    """
    Return completed IPL matches from Cricbuzz for the current season.

    Each dict: {team1, team2, winner (canonical name or None), start_ms}
    Used by outcome_checker.settle_watchlist_pending() to settle IPL picks
    without relying on the ESPN cricket/ipl endpoint (which always returns 404).
    """
    series_id = _discover_series_id(today.year)
    if not series_id:
        return []
    matches = _fetch_series_matches(series_id)
    result = []
    for m in matches:
        if m.get("state") != "Complete":
            continue
        t1 = m.get("team1", "")
        t2 = m.get("team2", "")
        result.append({
            "team1":        t1,
            "team2":        t2,
            "winner":       _parse_winner(m.get("status", ""), t1, t2),
            "start_ms":     m.get("start_ms", 0),
            "match_summary": m.get("status", ""),
        })
    return result


def _empty_context(venue_config: Dict, team_names: List[str]) -> Dict:
    return {
        "season_form":     {},
        "rest_days":       {normalize(n): 3 for n in team_names},
        "venue_stats":     {},
        "venue_config":    venue_config,
        "match_venues":    {},
        "h2h":             {},
        "match_flags":     {},
        "unavailabilities": {},
    }
