"""
FIFA World Cup 2026 statistics / context — Elo strength ratings.

International national teams have little usable shared-competition form during
the group stage, so the World Cup model is driven by Elo ratings rather than
xG (as MLS is). This module:

  1. Loads seed Elo ratings (src/data/wc_elo_seed.json).
  2. Merges any dynamically-learned ratings persisted in state/wc_elo.json.
  3. Self-updates ratings from completed ESPN matches (soccer/fifa.world) using
     the eloratings.net goal-difference-weighted update, tracking processed
     event IDs so no match is ever counted twice.
  4. Returns a context dict consumed by edge_finder.analyze_wc_game.

All network access is best-effort: if ESPN is unreachable the last-saved (or
seed) ratings are used unchanged. The tournament is watchlist-only, so a stale
rating never risks money.

Data source:
  - ESPN scoreboard: https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard
"""
import json
import logging
import math
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Country-name aliases for matching the Odds API / ESPN spelling to the Elo seed
# keys. Keyed by the canonicalised (accent/punctuation-stripped, lowercased) form.
_WC_ALIASES = {
    "czechia": "czech republic",
    "korea republic": "south korea",
    "korea dpr": "north korea",
    "ir iran": "iran",
    "turkiye": "turkey",
    "cabo verde": "cape verde",
    "usa": "united states",
    "united states of america": "united states",
}


def _canon(s: str) -> str:
    """Canonicalise a country name for matching: strip accents (Curaçao→curacao),
    drop punctuation (Bosnia & Herzegovina / Bosnia-Herzegovina → bosnia
    herzegovina; Côte d'Ivoire → cote divoire), lowercase, then apply aliases.
    Fixes silent default-Elo assignment from name-spelling mismatches."""
    if not s:
        return ""
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s.lower())
    s = " ".join(s.split())
    return _WC_ALIASES.get(s, s)

ESPN_WC_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
_SEED_PATH  = Path(__file__).parent / "wc_elo_seed.json"
_VENUE_PATH = Path(__file__).parent / "wc_venue_config.json"
_STATE_PATH = Path("state/wc_elo.json")
_TOURNAMENT_START = date(2026, 6, 11)
_REQUEST_TIMEOUT = 15


def normalize(name: str, ctx_dict: dict) -> str:
    """Return the closest key in ctx_dict matching name, or name itself.

    Matching is accent/punctuation/alias-insensitive (via _canon) so Odds API /
    ESPN spellings reconcile with the Elo-seed keys — e.g. Curaçao↔Curacao,
    Bosnia & Herzegovina↔Bosnia-Herzegovina, Türkiye↔Turkey, Czechia↔Czech
    Republic. Without this, a spelling mismatch silently fell through to the
    default Elo rating, leaving the model unable to tell that team apart.
    """
    if not name:
        return name
    nc = _canon(name)
    # exact canonical match
    for key in ctx_dict:
        if _canon(key) == nc:
            return key
    # substring canonical match
    for key in ctx_dict:
        kc = _canon(key)
        if nc and (nc in kc or kc in nc):
            return key
    # token-overlap fallback
    name_words = set(nc.split())
    best, best_score = name, 0
    for key in ctx_dict:
        score = len(name_words & set(_canon(key).split()))
        if score > best_score:
            best, best_score = key, score
    return best if best_score > 0 else name


def _load_seed() -> Dict[str, float]:
    try:
        data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        return {k: float(v) for k, v in data.items() if not k.startswith("_")}
    except Exception as e:
        logger.error(f"WC Elo seed load failed: {e}")
        return {}


def _load_state() -> Dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"WC Elo state load failed: {e}")
    return {}


def _save_state(state: Dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"WC Elo state save failed: {e}")


def _expected(r_home: float, r_away: float) -> float:
    """Elo win expectancy for the home side (draw counts as 0.5)."""
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away) / 400.0))


def _goal_multiplier(goal_diff: int) -> float:
    """eloratings.net-style goal-difference weight on the K-factor."""
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _scan_completed_matches(upto: date) -> List[Dict]:
    """
    Fetch all COMPLETED World Cup matches from the tournament start through *upto*
    (bounded window). Returns a list of {eid, date, home, away, hs, as_}.
    Best-effort: days that fail to fetch are skipped.
    """
    matches: List[Dict] = []
    day = max(_TOURNAMENT_START, upto - timedelta(days=45))  # bound the scan window
    while day <= upto:
        date_str = day.strftime("%Y%m%d")
        try:
            r = requests.get(ESPN_WC_SCOREBOARD, params={"dates": date_str},
                             timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.debug(f"WC: ESPN fetch failed for {day}: {e}")
            day += timedelta(days=1)
            continue

        for event in data.get("events", []):
            eid = str(event.get("id", ""))
            comps = event.get("competitions", [])
            if not eid or not comps:
                continue
            comp = comps[0]
            if not comp.get("status", {}).get("type", {}).get("completed", False):
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            home = home_c.get("team", {}).get("displayName", "")
            away = away_c.get("team", {}).get("displayName", "")
            try:
                hs = int(float(home_c.get("score", 0)))
                as_ = int(float(away_c.get("score", 0)))
            except (TypeError, ValueError):
                continue
            if not home or not away:
                continue
            matches.append({"eid": eid, "date": day, "home": home, "away": away, "hs": hs, "as_": as_})

        day += timedelta(days=1)

    return matches


def _apply_elo(elo: Dict[str, float], processed: set, matches: List[Dict]) -> int:
    """Apply Elo updates for any match not already in *processed* (mutates both)."""
    from src.config import WC_ELO_K, WC_HOST_NATIONS, WC_HOST_ELO_BONUS, WC_ELO_DEFAULT

    new_count = 0
    for m in matches:
        if m["eid"] in processed:
            continue
        home, away, hs, as_ = m["home"], m["away"], m["hs"], m["as_"]
        r_home = elo.get(normalize(home, elo), WC_ELO_DEFAULT)
        r_away = elo.get(normalize(away, elo), WC_ELO_DEFAULT)
        host_bonus = WC_HOST_ELO_BONUS if any(
            h.lower() in home.lower() or home.lower() in h.lower() for h in WC_HOST_NATIONS
        ) else 0.0
        we_home = _expected(r_home + host_bonus, r_away)
        w_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        mult = _goal_multiplier(hs - as_)
        delta = WC_ELO_K * mult * (w_home - we_home)
        elo[normalize(home, elo)] = round(r_home + delta, 1)
        elo[normalize(away, elo)] = round(r_away - delta, 1)
        processed.add(m["eid"])
        new_count += 1
        logger.info(f"WC Elo updated from {home} {hs}-{as_} {away} (Δ{delta:+.1f})")
    return new_count


def _build_rest_and_standings(matches: List[Dict], group_end: date):
    """
    From completed matches, compute:
      - last_played: {team: date of most recent match}        (rest signal)
      - standings:   {team: {"pts", "gp"}} for group-stage games only (dead-rubber)
    """
    last_played: Dict[str, date] = {}
    standings: Dict[str, Dict] = {}
    for m in matches:
        d, home, away, hs, as_ = m["date"], m["home"], m["away"], m["hs"], m["as_"]
        for t in (home, away):
            if t not in last_played or d > last_played[t]:
                last_played[t] = d
        if d <= group_end:
            for t in (home, away):
                standings.setdefault(t, {"pts": 0, "gp": 0})
            standings[home]["gp"] += 1
            standings[away]["gp"] += 1
            if hs > as_:
                standings[home]["pts"] += 3
            elif hs < as_:
                standings[away]["pts"] += 3
            else:
                standings[home]["pts"] += 1
                standings[away]["pts"] += 1
    return last_played, standings


def _load_venues() -> List[Dict]:
    try:
        data = json.loads(_VENUE_PATH.read_text(encoding="utf-8"))
        return data.get("venues", [])
    except Exception as e:
        logger.warning(f"WC venue config load failed: {e}")
        return []


def _match_venue(venue_name: str, city: str, venues: List[Dict]) -> Optional[Dict]:
    """Loose match of an ESPN venue/city string to a configured venue."""
    hay = f"{venue_name} {city}".lower()
    if not hay.strip():
        return None
    for v in venues:
        for alias in v.get("aliases", []):
            if alias.lower() in hay:
                return {"name": v["name"], "altitude_m": v["altitude_m"], "climate": v["climate"]}
    return None


def _fetch_fixtures_with_venues(target_date: date, venues: List[Dict]) -> Dict:
    """
    Fetch the ESPN scoreboard for *target_date* and map each fixture
    (lowercased home, away) → matched venue info (or None). Best-effort.
    """
    fixtures: Dict = {}
    if not venues:
        return fixtures
    date_str = target_date.strftime("%Y%m%d")
    try:
        r = requests.get(ESPN_WC_SCOREBOARD, params={"dates": date_str}, timeout=_REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug(f"WC: fixture/venue fetch failed for {target_date}: {e}")
        return fixtures

    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
        home = home_c.get("team", {}).get("displayName", "")
        away = away_c.get("team", {}).get("displayName", "")
        venue = comp.get("venue", {}) or {}
        vname = venue.get("fullName", "")
        city = (venue.get("address", {}) or {}).get("city", "")
        if home and away:
            # Key by canonicalised names under BOTH orderings, so the lookup
            # reconciles Odds-API vs ESPN spellings (accents/aliases) and doesn't
            # depend on which side each feed calls "home".
            vinfo = _match_venue(vname, city, venues)
            fixtures[(_canon(home), _canon(away))] = vinfo
            fixtures[(_canon(away), _canon(home))] = vinfo
    return fixtures


def get_wc_context(today_date: date, team_names: Optional[List[str]] = None) -> Dict:
    """
    Build the World Cup model context:
      - elo:         seed merged with learned ratings, updated from results
      - league_avg:  mean Elo
      - last_played: {team: date of most recent completed match}  (rest signal)
      - standings:   {team: {"pts", "gp"}} from group-stage results (dead-rubber)
      - fixtures:    {(home_lc, away_lc): venue_info|None}          (altitude/heat)

    All network access is best-effort; any missing piece degrades to "no signal"
    so the model silently falls back to pure Elo. Watchlist-only — never risks money.
    """
    from src.config import WC_GROUP_STAGE_END

    elo = _load_seed()
    state = _load_state()
    # Merge any learned ratings on top of the seed.
    for team, rating in (state.get("elo") or {}).items():
        elo[team] = float(rating)
    processed = set(state.get("processed_ids") or [])

    last_played: Dict[str, date] = {}
    standings: Dict[str, Dict] = {}
    fixtures: Dict = {}
    venues = _load_venues()

    try:
        yesterday = today_date - timedelta(days=1)
        matches = _scan_completed_matches(yesterday) if yesterday >= _TOURNAMENT_START else []
        if matches:
            if _apply_elo(elo, processed, matches):
                _save_state({"elo": elo, "processed_ids": sorted(processed)})
            group_end = date.fromisoformat(WC_GROUP_STAGE_END)
            last_played, standings = _build_rest_and_standings(matches, group_end)
    except Exception as e:
        logger.warning(f"WC results scan skipped: {e}")

    try:
        fixtures = _fetch_fixtures_with_venues(today_date, venues)
    except Exception as e:
        logger.debug(f"WC fixture/venue map skipped: {e}")

    league_avg = (sum(elo.values()) / len(elo)) if elo else 1620.0
    return {
        "elo": elo,
        "league_avg": league_avg,
        "last_played": last_played,
        "standings": standings,
        "fixtures": fixtures,
    }


def get_wc_injuries() -> Dict:
    """
    World Cup squad-availability data is not wired to a reliable feed yet.
    Returns an empty dict; the model handles missing injuries gracefully.
    """
    return {}
