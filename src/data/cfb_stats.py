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
_SCHEMA = 2   # v2 stores raw results + SRS ratings


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
            "results": [], "srs": {}, "srs_prior": {},
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

        # ── Margin ratings (the ones we actually predict with) ─────────────
        # Keep raw results so SRS can be refitted every run; Elo is retained
        # only as a secondary signal.
        seen_ids = {r.get("id") for r in state.get("results", [])}
        for g in games:
            if g["id"] not in seen_ids:
                state.setdefault("results", []).append(g)
                seen_ids.add(g["id"])
        # Drop anything older than two seasons — keeps the file bounded.
        cutoff = f"{season - 1}-07-01"
        state["results"] = [r for r in state["results"] if r.get("date", "") >= cutoff]

        prior_rows = [r for r in state["results"]
                      if _season_of(date.fromisoformat(r["date"])) < season]
        this_rows = [r for r in state["results"]
                     if _season_of(date.fromisoformat(r["date"])) >= season]

        # Last season's SRS, regressed, becomes this season's prior.
        from src.config import CFB_PRIOR_REGRESSION
        prior_srs = compute_srs(prior_rows) if prior_rows else {}
        priors = {t: v * CFB_PRIOR_REGRESSION for t, v in prior_srs.items()}
        state["srs_prior"] = priors
        state["srs"] = compute_srs(this_rows, priors=priors) if this_rows else dict(priors)

        state["last_scan"] = today.isoformat()
        _save_state(state)
        srs = state.get("srs") or {}
        spread = (max(srs.values()) - min(srs.values())) if srs else 0.0
        logger.info(f"CFB ratings: {len(srs)} teams, {applied} new result(s), "
                    f"SRS spread {spread:.1f} pts "
                    f"({len(this_rows)} games this season, {len(prior_rows)} prior)")
        if srs and spread < 30.0:
            logger.warning(
                f"CFB SRS spread is only {spread:.1f} points — too compressed to "
                f"express real lines (they reach 45). Picks will be gated out; "
                f"this is the Sep 3 2026 failure mode, not a silent one.")
    except Exception as e:
        logger.error(f"CFB Elo refresh failed (non-fatal): {e}")
    return state


def get_cfb_context(today: date) -> Dict[str, Any]:
    """{'elo': {...}, 'games': {...}} for the analyzer. Never raises."""
    try:
        state = refresh_cfb_elo(today)
        return {"elo": state.get("elo", {}), "games": state.get("games", {}),
                "srs": state.get("srs", {})}
    except Exception as e:
        logger.error(f"CFB context failed: {e}")
        return {"elo": {}, "games": {}, "srs": {}}


def rating_for(name: str, ctx: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """(rating IN POINTS, games_played), or (None, 0) when unknown.

    The rating is an SRS margin rating, so it is already in points and the
    caller needs no Elo->points conversion — that conversion is exactly what was
    miscalibrated by ~8x on Sep 3 2026.

    Unknown means FCS or an unmappable name; the game must be SKIPPED rather
    than silently defaulting to average, which would manufacture a huge phantom
    edge against a real opponent.
    """
    srs = ctx.get("srs") or {}
    counts = ctx.get("games", {})
    k = canon(name)
    if k in srs:
        return float(srs[k]), int(counts.get(k, 0))
    for key in srs:
        if key.startswith(k) or k.startswith(key):
            return float(srs[key]), int(counts.get(key, 0))
    return None, 0

# ---------------------------------------------------------------------------
# Margin ratings (SRS) — replaces the Elo scale for prediction
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Win/loss Elo threw away the only signal college football
# actually gives you: HOW MUCH teams win by. Measured Sep 3 2026, one replayed
# season of Elo spanned just 372 points — a maximum expressible margin of 14.9
# — while real lines reach 45. Against the market's own lines the implied
# conversion was ~3 Elo per point versus our constant of 25, i.e. ratings
# compressed roughly 8x. Every pick came out pinned to the credibility cap.
#
# A Simple Rating System solves that structurally rather than by retuning:
#
#     rating_i = mean over i's games of (margin_ij + rating_j)
#
# solved iteratively. Ratings come out IN POINTS, so there is no Elo->points
# conversion left to miscalibrate, and the scale naturally spans the real range
# because it is fitted to actual scoring margins.
#
# Margins are capped before fitting: a 63-0 win says little more than 35-0, and
# uncapped blowouts would let a few cupcake games dominate a rating.

SRS_MARGIN_CAP = 28.0
SRS_ITERATIONS = 25
# Pseudo-games of prior-season rating blended in, so a team with two games is
# not rated purely on two games.
SRS_PRIOR_WEIGHT = 4.0


def compute_srs(games: List[Dict], priors: Optional[Dict[str, float]] = None,
                home_adv: float = None) -> Dict[str, float]:
    """Opponent-adjusted average scoring margin, in POINTS. Never raises."""
    from src.config import CFB_HOME_ADV_POINTS
    hfa = CFB_HOME_ADV_POINTS if home_adv is None else home_adv
    priors = priors or {}
    try:
        # Neutralise home field, then cap, so a rating reflects true strength.
        rows: Dict[str, List[Tuple[str, float]]] = {}
        for g in games:
            hk, ak = canon(g["home"]), canon(g["away"])
            if not hk or not ak:
                continue
            m = float(g["home_score"] - g["away_score"])
            if not g.get("neutral"):
                m -= hfa
            m = max(-SRS_MARGIN_CAP, min(SRS_MARGIN_CAP, m))
            rows.setdefault(hk, []).append((ak, m))
            rows.setdefault(ak, []).append((hk, -m))
        if not rows:
            return {}

        rating = {t: float(priors.get(t, 0.0)) for t in rows}
        for _ in range(SRS_ITERATIONS):
            nxt = {}
            for t, opps in rows.items():
                acc = sum(m + rating.get(o, 0.0) for o, m in opps)
                n = len(opps)
                # Shrink toward the prior when a team has few games.
                p = float(priors.get(t, 0.0))
                nxt[t] = (acc + SRS_PRIOR_WEIGHT * p) / (n + SRS_PRIOR_WEIGHT)
            # Re-centre so the average team is 0 — keeps the scale anchored.
            mean = sum(nxt.values()) / len(nxt)
            rating = {t: v - mean for t, v in nxt.items()}
        return rating
    except Exception as e:
        logger.error(f"SRS computation failed (non-fatal): {e}")
        return {}
