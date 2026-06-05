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
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ESPN_WC_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
_SEED_PATH  = Path(__file__).parent / "wc_elo_seed.json"
_STATE_PATH = Path("state/wc_elo.json")
_TOURNAMENT_START = date(2026, 6, 11)
_REQUEST_TIMEOUT = 15


def normalize(name: str, ctx_dict: dict) -> str:
    """Return the closest key in ctx_dict matching name, or name itself."""
    if not name:
        return name
    name_lower = name.lower()
    for key in ctx_dict:
        if key.lower() == name_lower:
            return key
    for key in ctx_dict:
        if name_lower in key.lower() or key.lower() in name_lower:
            return key
    name_words = set(name_lower.split())
    best, best_score = name, 0
    for key in ctx_dict:
        score = len(name_words & set(key.lower().split()))
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


def _update_elo_from_results(elo: Dict[str, float], processed: set, upto: date) -> int:
    """
    Fetch completed ESPN matches from the tournament start through *upto* and
    apply Elo updates for any event not already in *processed*. Returns the
    number of newly-processed matches. Mutates elo and processed in place.
    """
    from src.config import (
        WC_ELO_K, WC_HOST_NATIONS, WC_HOST_ELO_BONUS, WC_ELO_DEFAULT,
    )

    new_count = 0
    day = max(_TOURNAMENT_START, upto - timedelta(days=45))  # bound the scan window
    while day <= upto:
        date_str = day.strftime("%Y%m%d")
        try:
            r = requests.get(ESPN_WC_SCOREBOARD, params={"dates": date_str},
                             timeout=_REQUEST_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.debug(f"WC Elo: ESPN fetch failed for {day}: {e}")
            day += timedelta(days=1)
            continue

        for event in data.get("events", []):
            eid = str(event.get("id", ""))
            if not eid or eid in processed:
                continue
            comps = event.get("competitions", [])
            if not comps:
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
            processed.add(eid)
            new_count += 1
            logger.info(f"WC Elo updated from {home} {hs}-{as_} {away} (Δ{delta:+.1f})")

        day += timedelta(days=1)

    return new_count


def get_wc_context(today_date: date, team_names: Optional[List[str]] = None) -> Dict:
    """
    Build the World Cup model context: current Elo ratings (seed merged with
    learned values, then updated from results through yesterday).

    Returns:
        {"elo": {team: rating}, "league_avg": float}
    """
    elo = _load_seed()
    state = _load_state()
    # Merge any learned ratings on top of the seed.
    for team, rating in (state.get("elo") or {}).items():
        elo[team] = float(rating)
    processed = set(state.get("processed_ids") or [])

    try:
        yesterday = today_date - timedelta(days=1)
        if yesterday >= _TOURNAMENT_START:
            n = _update_elo_from_results(elo, processed, yesterday)
            if n:
                _save_state({"elo": elo, "processed_ids": sorted(processed)})
    except Exception as e:
        logger.warning(f"WC Elo update skipped: {e}")

    league_avg = (sum(elo.values()) / len(elo)) if elo else 1620.0
    return {"elo": elo, "league_avg": league_avg}


def get_wc_injuries() -> Dict:
    """
    World Cup squad-availability data is not wired to a reliable feed yet.
    Returns an empty dict; the model handles missing injuries gracefully.
    """
    return {}
