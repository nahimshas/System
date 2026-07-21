"""
Liga MX statistics / context — Elo strength ratings.

No free club-xG feed exists for Liga MX, so strength is Elo-based (like the
World Cup, unlike MLS). Ratings are:

  1. Bootstrapped on first run by replaying ~1 year of completed ESPN Liga MX
     results from a neutral 1500 baseline (no hand-seeded table needed).
  2. Self-updated from new completed results on every run (eloratings.net
     goal-difference-weighted K-factor), tracking processed event ids so each
     match moves the ratings exactly once.

State persists in state/ligamx_elo.json:
  {"elo": {team: rating}, "processed": [eid, ...], "bootstrapped": true}

ESPN Liga MX endpoints (no auth):
  - scoreboard: https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard
"""
import json
import logging
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ESPN_LIGAMX_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard"
_STATE_PATH = Path("state/ligamx_elo.json")
_REQUEST_TIMEOUT = 15

# Odds-API ↔ ESPN name divergences that canon()+token-overlap don't resolve.
# Extend as the first live runs surface unmatched names (logged as WARNING).
_ALIASES = {
    "america": "america",
    "club america": "america",
    "cd guadalajara": "guadalajara",
    "guadalajara chivas": "guadalajara",
    "chivas": "guadalajara",
    "cruz azul": "cruz azul",
    "uanl": "tigres",
    "tigres uanl": "tigres",
    "unam": "pumas",
    "pumas unam": "pumas",
    "club leon": "leon",
    "fc juarez": "juarez",
    "atletico san luis": "san luis",
    "queretaro": "queretaro",
    "mazatlan fc": "mazatlan",
}

# Tokens dropped when canonicalising a Mexican club name.
_SKIP_TOKENS = {"club", "cd", "cf", "fc", "deportivo", "de", "the"}


def canon(s: str) -> str:
    """Lowercase, strip accents + club-prefix tokens, then apply the alias map.
    'Club América' → 'america', 'CD Guadalajara' → 'guadalajara'."""
    low = unicodedata.normalize("NFKD", (s or "").lower())
    low = "".join(c for c in low if not unicodedata.combining(c))
    low = low.replace(".", " ").strip()
    if low in _ALIASES:
        return _ALIASES[low]
    toks = [t for t in low.split() if t not in _SKIP_TOKENS]
    stripped = " ".join(toks)
    return _ALIASES.get(stripped, stripped)


def normalize(name: str, elo_map: Dict) -> str:
    """Resolve a team name to the key used in the Elo map.

    Tries: exact canon key → substring match → token-overlap match. Falls back
    to the canon key (which then reads as an unseeded/default team)."""
    key = canon(name)
    if key in elo_map:
        return key
    ktoks = set(key.split())
    best, best_ov = None, 0
    for k in elo_map:
        if key and (key in k or k in key):
            return k
        ov = len(ktoks & set(k.split()))
        if ov > best_ov:
            best, best_ov = k, ov
    return best if best_ov >= 1 else key


def _load_state() -> Dict:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception as e:
        logger.warning(f"Liga MX Elo state load failed: {e}")
    return {"elo": {}, "processed": [], "bootstrapped": False}


def _save_state(state: Dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"Liga MX Elo state save failed: {e}")


def _expected(r_home: float, r_away: float) -> float:
    """Elo win expectancy for the home side (draw counts as 0.5)."""
    return 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400.0))


def _goal_multiplier(goal_diff: int) -> float:
    """eloratings.net-style goal-difference weight on the K-factor."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def _scan_completed_matches(since: date, upto: date) -> List[Dict]:
    """Fetch completed Liga MX matches in [since, upto], chunked by ~30 days.
    Returns chronologically-sorted {eid, date, home, away, hs, as_}."""
    matches: List[Dict] = []
    seen_eids = set()
    chunk_start = since
    while chunk_start <= upto:
        chunk_end = min(chunk_start + timedelta(days=29), upto)
        ds = f"{chunk_start.strftime('%Y%m%d')}-{chunk_end.strftime('%Y%m%d')}"
        try:
            r = requests.get(ESPN_LIGAMX_SCOREBOARD, params={"dates": ds}, timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.debug(f"Liga MX: ESPN fetch failed for {ds}: {e}")
            chunk_start = chunk_end + timedelta(days=1)
            continue

        for event in data.get("events", []):
            eid = str(event.get("id", ""))
            comps = event.get("competitions", [])
            if not eid or eid in seen_eids or not comps:
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
            try:
                mdate = date.fromisoformat((event.get("date", "") or "")[:10])
            except Exception:
                mdate = chunk_start
            matches.append({"eid": eid, "date": mdate, "home": home, "away": away, "hs": hs, "as_": as_})
            seen_eids.add(eid)

        chunk_start = chunk_end + timedelta(days=1)

    matches.sort(key=lambda m: (m["date"], m["eid"]))
    return matches


def _apply_elo(elo: Dict[str, float], processed: set, matches: List[Dict]) -> int:
    """Apply Elo updates for any match not already in *processed* (mutates both).
    Home side gets the home-advantage bonus in the win-expectancy step."""
    from src.config import LIGAMX_ELO_K, LIGAMX_HOME_ELO_BONUS, LIGAMX_ELO_DEFAULT

    new_count = 0
    for m in matches:
        if m["eid"] in processed:
            continue
        home, away, hs, as_ = m["home"], m["away"], m["hs"], m["as_"]
        hk, ak = normalize(home, elo), normalize(away, elo)
        r_home = elo.get(hk, LIGAMX_ELO_DEFAULT)
        r_away = elo.get(ak, LIGAMX_ELO_DEFAULT)
        we_home = _expected(r_home + LIGAMX_HOME_ELO_BONUS, r_away)
        w_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        mult = _goal_multiplier(hs - as_)
        delta = LIGAMX_ELO_K * mult * (w_home - we_home)
        elo[hk] = round(r_home + delta, 1)
        elo[ak] = round(r_away - delta, 1)
        processed.add(m["eid"])
        new_count += 1
    return new_count


def _refresh_elo(today_date: date) -> Dict:
    """Load state, bootstrap on first run, apply any new results, persist, return."""
    from src.config import LIGAMX_BOOTSTRAP_DAYS

    state = _load_state()
    elo = state.get("elo", {})
    processed = set(state.get("processed", []))

    if not state.get("bootstrapped"):
        logger.info("Liga MX Elo: bootstrapping from ~1yr of ESPN results (1500 baseline)")
        since = today_date - timedelta(days=LIGAMX_BOOTSTRAP_DAYS)
        matches = _scan_completed_matches(since, today_date)
        n = _apply_elo(elo, processed, matches)
        logger.info(f"Liga MX Elo bootstrap: {n} matches replayed, {len(elo)} teams rated")
        state["bootstrapped"] = True
    else:
        # Incremental: only scan the recent window for newly-completed games.
        since = today_date - timedelta(days=21)
        matches = _scan_completed_matches(since, today_date)
        n = _apply_elo(elo, processed, matches)
        if n:
            logger.info(f"Liga MX Elo: {n} new result(s) applied")

    # last_played (rest signal) from the most recent scan window.
    last_played: Dict[str, date] = {}
    for m in matches:
        for t in (m["home"], m["away"]):
            k = normalize(t, elo)
            if k not in last_played or m["date"] > last_played[k]:
                last_played[k] = m["date"]

    state["elo"] = elo
    state["processed"] = sorted(processed)
    _save_state(state)
    return {"elo": elo, "last_played": last_played}


def get_ligamx_context(today_date: date, team_names: Optional[List[str]] = None) -> Dict:
    """Return {'elo': {team: rating}, 'last_played': {team: date}} for the analyzer.
    team_names is accepted for signature-parity with other sports (unused —
    Elo is global, not per-slate)."""
    try:
        return _refresh_elo(today_date)
    except Exception as e:
        logger.error(f"Liga MX context build failed: {e}")
        return {"elo": {}, "last_played": {}}


def get_ligamx_injuries() -> Dict:
    """No reliable free Liga MX injury feed — injuries are unmodeled (stub,
    matching the WC treatment). Returns {} so the analyzer skips injury logic."""
    return {}
