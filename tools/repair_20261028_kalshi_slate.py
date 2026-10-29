#!/usr/bin/env python3
"""One-off repair of the 2026-10-28 slate — the first Kalshi-priced day.

Two defects shipped that day and both are recorded in `tools/analysis/
checkpoints.json` (`kalshi_spread_orientation`, `kalshi_start_times_from_espn`).
The code is fixed, but the state written that morning is still wrong, and two
of the consumers that read it run AFTER the fact:

1. **commence_time was Kalshi's settlement stamp, ~3h after tip.** The nightly
   Kalshi CLV pass samples the "close" at commence_time, so it would have
   sampled every market AFTER the game finished, when Kalshi has already
   resolved to ~0 or ~1. Measured on PHI@IND: the true tip-off close was
   0.295 (CLV −4.75pp, a real market move) while the late stamp gave 0.031
   (CLV −31pp). That turns CLV into a proxy for the RESULT, which destroys the
   entire point of the metric — it exists to measure skill independently of
   outcome on far fewer samples — and it feeds the governor that gates budget
   markets. Repaired from ESPN, which is authoritative. This is a true
   correction, not a reconstruction: the real tip-off time is a known fact.

2. **Spread sign was always "home favoured"**, so in the 9 games where the AWAY
   team was favoured both sides' lines were inverted. Those rows CANNOT be
   repaired faithfully: the morning order book is gone, and the rung the
   selector would have chosen depends on the probability it got wrong, so the
   correct line is not recoverable. They are instead VOIDED with
   `outcome="cancel"`, which every learning consumer already excludes —
   `calibration._outcome_value` returns None for it, `cap_state` filters to
   ("win","loss"), and both shadow-log settlers skip any row whose outcome is
   already set. Voiding beats inventing a line, and it matters now because
   NBA Spread sits at n=92 against the n=100 calibration threshold, so a batch
   of systematically dog-biased rows would arrive exactly as that market
   starts self-tuning.

Idempotent and safe to re-run: it only rewrites a timestamp that is more than
10 minutes off, and only voids a row whose outcome is still unset. Writes a
.bak beside each file it touches.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.game_types import _ESPN_PATHS, _team_match  # noqa: E402

DAY = "2026-10-28"
# ⚠️ Only sports PRICED BY KALSHI that day can have an inverted spread. CFB came
# via the Odds API, which carries a correct posted line, and an away favourite
# there is perfectly ordinary — voiding its rows would discard good data. The
# first pass of this script did exactly that.
KALSHI_PRICED = {"NBA", "NHL", "MLB", "MLS", "LIGAMX", "WNBA", "NFL"}
SHADOW = Path("state/shadow_log/2026-10.json")
DECISION = Path("state/decision_log/2026-10.json")
DRIFT_TOLERANCE_S = 600


def espn_times(slug: str):
    path = _ESPN_PATHS.get(slug)
    if not path:
        return []
    out = []
    for d in ("20261028", "20261029"):
        params = {"dates": d}
        if slug == "cfb":
            params.update({"groups": 80, "limit": 300})
        try:
            r = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard",
                params=params, timeout=20).json()
        except Exception:
            continue
        for e in (r.get("events") or []):
            cs = (e.get("competitions") or [{}])[0].get("competitors") or []
            h = next((c.get("team", {}).get("displayName", "")
                      for c in cs if c.get("homeAway") == "home"), "")
            a = next((c.get("team", {}).get("displayName", "")
                      for c in cs if c.get("homeAway") == "away"), "")
            if h and a and e.get("date"):
                out.append((h, a, e["date"]))
    return out


def load_rows(p: Path):
    raw = json.load(open(p))
    inner = raw.get("entries", raw) if isinstance(raw, dict) else raw
    rows = list(inner.values()) if isinstance(inner, dict) else inner
    return raw, [r for r in rows if isinstance(r, dict)]


def _teams(game: str):
    """(away, home) from a game string, or None."""
    if " @ " not in (game or ""):
        return None
    away, home = [x.strip() for x in game.split(" @ ", 1)]
    return (away, home) if away and home else None


def _same_game(a: str, b: str) -> bool:
    """⚠️ The two logs do NOT agree on team spelling — the shadow log holds
    "Los Angeles Lakers @ LA Clippers" where the decision log holds
    "... @ Los Angeles Clippers". Exact string matching silently missed that
    game on the first pass, leaving an inverted row un-voided."""
    ta, tb = _teams(a), _teams(b)
    if not ta or not tb:
        return False
    return _team_match(ta[0], tb[0]) and _team_match(ta[1], tb[1])


def inverted_games() -> set:
    """Games where the AWAY team was favoured, from the MONEYLINE — the one
    market the orientation bug never touched. The decision log carries both
    sides of every market, so coverage is complete by construction."""
    _raw, rows = load_rows(DECISION)
    fav = {}
    for v in rows:
        if (v.get("date") == DAY and v.get("market_type") == "Moneyline"
                and (v.get("market_prob_at_first_pick") or 0) > 0.5):
            fav[v["game"]] = str(v.get("side") or "")
    out = set()
    for g, side in fav.items():
        if " @ " not in g:
            continue
        away = g.split(" @ ", 1)[0].strip()
        if _team_match(away, side):
            out.add(g)
    return out


def main(apply: bool) -> None:
    cache = {s: espn_times(s) for s in
             ("nba", "nhl", "mlb", "mls", "ligamx", "cfb", "wnba", "wc")}
    bad_games = inverted_games()
    print(f"games with an away favourite (spreads inverted): {len(bad_games)}")

    for path in (SHADOW, DECISION):
        raw, rows = load_rows(path)
        retimed = voided = 0
        for v in rows:
            if v.get("date") != DAY:
                continue
            g = str(v.get("game") or "")
            if " @ " in g:
                away, home = [x.strip() for x in g.split(" @ ", 1)]
                for h, a, dt in cache.get((v.get("sport") or "").lower()) or []:
                    if not (_team_match(home, h) and _team_match(away, a)):
                        continue
                    old = str(v.get("commence_time") or "")
                    try:
                        o = datetime.fromisoformat(old.replace("Z", "+00:00"))
                        n = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                    except Exception:
                        break
                    if abs((n - o).total_seconds()) > DRIFT_TOLERANCE_S:
                        if apply:
                            v["commence_time"] = dt
                            v.setdefault("repair_notes", []).append(
                                "commence_time corrected from ESPN "
                                "(Kalshi reported settlement, not tip-off)")
                        retimed += 1
                    break
            if (v.get("market_type") == "Spread"
                    and (v.get("sport") or "").upper() in KALSHI_PRICED
                    and any(_same_game(g, bg) for bg in bad_games)
                    and v.get("outcome") is None):
                if apply:
                    v["outcome"] = "cancel"
                    v.setdefault("repair_notes", []).append(
                        "VOIDED: spread sign was inverted (always 'home "
                        "favoured'); the correct line is not recoverable")
                voided += 1
        print(f"{path.name}: retime {retimed}, void {voided}"
              f"{'' if apply else '  (dry run)'}")
        if apply and (retimed or voided):
            shutil.copy(path, str(path) + ".bak")
            json.dump(raw, open(path, "w"), indent=2)


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
