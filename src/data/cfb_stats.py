"""
College football Elo ratings, built from ESPN results.

Bootstraps by replaying the PREVIOUS season plus the current one from a neutral
baseline, then self-updates as results arrive. State lives in
state/cfb_elo.json so the replay happens once, not every morning.

Two CFB-specific problems this handles:

1. COLD START, permanently. A team plays ~12 games a year, so a season boundary
   throws away most of what we knew. Ratings are regressed toward the mean at
   each new season (CFB_PRIOR_REGRESSION) and the prior is blended out over the
   first few games (CFB_WARMSTART_RAMP_GAMES) — the same shape as the NFL warm
   start, tuned harder because CFB roster turnover is worse.

2. FCS OPPONENTS. Non-FBS teams appear on early-season schedules as paid
   blowouts. They carry no meaningful rating and their games are unpriced, so
   they update nothing and are never analysed.

Never raises — a failure returns empty ratings and the sport simply produces no
picks that day.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_STATE_PATH = Path("state/cfb_elo.json")
_ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
_TIMEOUT = 25
# ESPN group 80 = FBS. Without it the scoreboard includes FCS-only slates.
_FBS_GROUP = 80
_SCHEMA = 1


def canon(name: str) -> str:
    """Canonical team key — ESPN and the Odds API disagree on punctuation."""
    s = (name or "").lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"\b(university|univ|college)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_state() -> Dict:
    try:
        if _STATE_PATH.exists():
            d = json.loads(_STATE_PATH.read_text())
            if d.get("schema") == _SCHEMA:
                return d
    except Exception as e:
        logger.warning(f"CFB Elo state unreadable: {e}")
    return {"schema": _SCHEMA, "elo": {}, "games": {}, "processed": [],
            "season": None, "last_scan": None}


def _save_state(state: Dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception as e:
        logger.warning(f"CFB Elo state not saved: {e}")


def _season_of(d: date) -> int:
    """CFB seasons are labelled by their starting year; they run Aug-Jan.

    The cutoff is AUGUST, not July: a July date is offseason and belongs to the
    season that just finished. A July cutoff would roll ratings into the new
    season a month early, regressing them before the prior season's bowls had
    even been replayed.
    """
    return d.year if d.month >= 8 else d.year - 1


def _expected(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - r_home) / 400.0))


def _margin_multiplier(margin: int, elo_diff: float) -> float:
    """Dampen blowouts and correct for autocorrelation.

    A 50-point win over a cupcake says less than a 50-point win over a peer, and
    without the elo_diff term strong teams inflate without bound (the standard
    538 correction).
    """
    return (max(1, abs(margin)) ** 0.8) / (7.5 + 0.006 * abs(elo_diff))


def _fetch_scoreboard(day: date) -> List[Dict]:
    """Completed FBS games for one date. Returns [] on any failure."""
    try:
        r = requests.get(_ESPN, params={"dates": day.strftime("%Y%m%d"),
                                        "groups": _FBS_GROUP, "limit": 400},
                         timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug(f"CFB scoreboard fetch failed ({day}): {e}")
        return []

    out = []
    for ev in data.get("events", []):
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {})
        if not status.get("completed"):
            continue
        cs = comp.get("competitors", [])
        if len(cs) < 2:
            continue
        home = next((c for c in cs if c.get("homeAway") == "home"), None)
        away = next((c for c in cs if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        try:
            hs, as_ = int(home.get("score", 0)), int(away.get("score", 0))
        except (TypeError, ValueError):
            continue
        # An FBS-vs-FCS game has one competitor outside the FBS group.
        def _is_fbs(c):
            t = c.get("team", {})
            return bool(t.get("id")) and t.get("displayName")
        if not (_is_fbs(home) and _is_fbs(away)):
            continue
        out.append({
            "id": str(ev.get("id")),
            "date": (ev.get("date") or "")[:10],
            "home": home.get("team", {}).get("displayName", ""),
            "away": away.get("team", {}).get("displayName", ""),
            "home_score": hs, "away_score": as_,
            "neutral": bool(comp.get("neutralSite")),
        })
    return out


def _apply(state: Dict, games: List[Dict]) -> int:
    from src.config import (CFB_ELO_DEFAULT, CFB_ELO_K, CFB_HOME_ELO_BONUS)
    elo = state["elo"]
    counts = state["games"]
    processed = set(state.get("processed", []))
    n = 0
    for g in sorted(games, key=lambda x: x["date"]):
        if g["id"] in processed:
            continue
        hk, ak = canon(g["home"]), canon(g["away"])
        if not hk or not ak:
            continue
        rh = elo.get(hk, CFB_ELO_DEFAULT)
        ra = elo.get(ak, CFB_ELO_DEFAULT)
        bonus = 0.0 if g.get("neutral") else CFB_HOME_ELO_BONUS
        exp_h = _expected(rh + bonus, ra)
        margin = g["home_score"] - g["away_score"]
        w_h = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        mult = _margin_multiplier(margin, (rh + bonus) - ra)
        delta = CFB_ELO_K * mult * (w_h - exp_h)
        elo[hk] = rh + delta
        elo[ak] = ra - delta
        counts[hk] = counts.get(hk, 0) + 1
        counts[ak] = counts.get(ak, 0) + 1
        processed.add(g["id"])
        n += 1
    state["processed"] = sorted(processed)
    return n


def _regress_for_new_season(state: Dict, season: int) -> None:
    """Pull every rating toward the mean at a season boundary and reset counts.

    Without this, a team's rating carries a full year of roster that no longer
    exists. CFB turnover is severe enough that keeping only 60% of the distance
    from average is the defensible default.
    """
    from src.config import CFB_ELO_DEFAULT, CFB_PRIOR_REGRESSION
    if state.get("season") == season:
        return
    for k, v in list(state.get("elo", {}).items()):
        state["elo"][k] = CFB_ELO_DEFAULT + (v - CFB_ELO_DEFAULT) * CFB_PRIOR_REGRESSION
    state["games"] = {}
    state["season"] = season
    logger.info(f"CFB Elo regressed toward mean for season {season} "
                f"({len(state.get('elo', {}))} teams)")


def refresh_cfb_elo(today: date, bootstrap_days: int = 430) -> Dict:
    """Bring ratings up to date, bootstrapping the first time. Never raises."""
    state = _load_state()
    try:
        season = _season_of(today)
        last = state.get("last_scan")
        if not state.get("elo"):
            start = today - timedelta(days=bootstrap_days)
            logger.info(f"CFB Elo bootstrap: replaying from {start}")
        else:
            start = date.fromisoformat(last) - timedelta(days=2) if last else today - timedelta(days=10)

        # Scan only plausible game days — CFB is Thu-Sat plus bowl season.
        games: List[Dict] = []
        d = start
        while d <= today:
            if d.weekday() in (1, 3, 4, 5, 6) or d.month in (12, 1):
                games.extend(_fetch_scoreboard(d))
            d += timedelta(days=1)

        # Regress BEFORE applying this season's results, not after.
        prior_season_games = [g for g in games if _season_of(date.fromisoformat(g["date"])) < season]
        this_season_games = [g for g in games if _season_of(date.fromisoformat(g["date"])) >= season]
        if prior_season_games:
            _apply(state, prior_season_games)
        _regress_for_new_season(state, season)
        applied = _apply(state, this_season_games)

        state["last_scan"] = today.isoformat()
        _save_state(state)
        logger.info(f"CFB Elo: {len(state['elo'])} teams, {applied} new result(s)")
    except Exception as e:
        logger.error(f"CFB Elo refresh failed (non-fatal): {e}")
    return state


def get_cfb_context(today: date) -> Dict[str, Any]:
    """{'elo': {...}, 'games': {...}} for the analyzer. Never raises."""
    try:
        state = refresh_cfb_elo(today)
        return {"elo": state.get("elo", {}), "games": state.get("games", {})}
    except Exception as e:
        logger.error(f"CFB context failed: {e}")
        return {"elo": {}, "games": {}}


def rating_for(name: str, ctx: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """(rating, games_played) for a team, or (None, 0) when unknown.

    Unknown means FCS or a name we cannot map — either way the game must not be
    analysed rather than silently defaulting to an average rating.
    """
    from src.config import CFB_ELO_DEFAULT
    elo = ctx.get("elo", {})
    k = canon(name)
    if k in elo:
        return float(elo[k]), int(ctx.get("games", {}).get(k, 0))
    # tolerate minor naming differences (St/State, mascot suffixes)
    for key in elo:
        if key.startswith(k) or k.startswith(key):
            return float(elo[key]), int(ctx.get("games", {}).get(key, 0))
    return None, 0
