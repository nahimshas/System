"""Price a slate from Kalshi — the candidate replacement for the Odds API.

⚠️ WHAT IS GENUINELY NEW HERE, AND WHY A TRIAL IS STILL NEEDED
The decision log already proves Kalshi can PRICE A GAME WE KNOW ABOUT: 9,698
rows carry an Odds API price and a Kalshi mid for the same moment, and
moneylines/spreads agree within a fraction of a point. None of that exercises
DISCOVERY. Every one of those rows began life as a game the Odds API handed us:

    games = module.fetch_games(today_str) if ODDS_API_KEY else []

No sport has any other source. So the untested half is "which games exist today,
and is there a complete line set for each" — and unlike CLV, pricing gets no
second chance. CLV self-heals over a 7-day window; a pricing failure at 9am is
that day's card, permanently.

This module therefore does the part that has never run, and `coverage_report()`
measures ONLY reliability — did we find the games, did we get a full line set —
not price agreement, which the archive has already settled.

Never raises. Kalshi is free and keyless, so there is no credit budget here.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.data.kalshi import (_f, _game_code, _norm, _team_token, enabled,
                             event_date, fetch_open_markets)
from src.data.kalshi_clv import SPORT_SERIES

logger = logging.getLogger(__name__)

# Reverse team lookup is built per sport from the static maps, so a Kalshi token
# ("Los Angeles R") can be turned back into the name our analyzers expect
# ("Los Angeles Rams"). CFB is excluded: it has no static map by design.
_REVERSE_SOURCES = {
    "MLB": "TEAM_TO_KALSHI", "NFL": "NFL_TEAM_TO_KALSHI",
    "NBA": "NBA_TEAM_TO_KALSHI", "NHL": "NHL_TEAM_TO_KALSHI",
    "WNBA": "WNBA_TEAM_TO_KALSHI", "MLS": "MLS_TEAM_TO_KALSHI",
    "LIGAMX": "LIGAMX_TEAM_TO_KALSHI",
}


def _reverse_map(sport: str) -> Dict[str, str]:
    """{normalised Kalshi token -> our full team name}."""
    import src.data.kalshi as K
    name = _REVERSE_SOURCES.get(sport.upper())
    table = getattr(K, name, {}) if name else {}
    # Later keys win on ties, which is harmless: aliases map to the same token.
    return {_norm(v): k for k, v in table.items()}


def _mid(market: Dict) -> Optional[float]:
    """Mid of the two-sided book, or None when it is not quoted both ways.

    Mid rather than ask: the ask carries the ~1% overround we measure
    separately in the execution log, and mixing the two would double-count it.
    """
    b, a = _f(market.get("yes_bid_dollars")), _f(market.get("yes_ask_dollars"))
    if b is None or a is None or not (0 < b < a < 1):
        return None
    return round((a + b) / 2.0, 4)


def _point(title: str) -> Optional[float]:
    m = re.search(r"over\s+(\d+(?:\.\d+)?)", _norm(title))
    return float(m.group(1)) if m else None


def discover(sport: str, game_date: str) -> List[Dict[str, Any]]:
    """Games listed on Kalshi for `game_date`, priced into our game-dict shape.

    Returns the same keys the analyzers read from the Odds API path:
    game_id / sport / home_team / away_team / commence_time /
    moneyline / spread / total — so a switch is a swap of source, not a
    rewrite of every analyzer.

    ⚠️ A game is only returned when BOTH teams map back to our names. An
    unmappable game is dropped and counted, never guessed at: a half-identified
    game would price one side against the wrong team.
    """
    out: List[Dict[str, Any]] = []
    if not enabled():
        return out
    sport = sport.upper()
    series = SPORT_SERIES.get(sport) or {}
    ml_series = series.get("Moneyline")
    if not ml_series:
        return out
    rev = _reverse_map(sport)
    if not rev:
        return out          # CFB and anything without a static map

    # Keyed by SERIES TICKER, not market name — every lookup below is by
    # ticker. Keying by name also silently refetches a series shared by two
    # market types (F5 Moneyline and F5 Tie are both KXMLBF5).
    books: Dict[str, List[Dict]] = {}
    for _tkr in {v for v in series.values() if v}:
        books[_tkr] = fetch_open_markets(_tkr, max_pages=8)

    # ── Discovery: the moneyline book defines the slate ──────────────────────
    # It is the only book whose titles are bare team names, and the ticker's
    # game segment joins it to every other book. Same anchor the CLV resolver
    # uses, for the same reason: prose drifts, tickers do not.
    games: Dict[str, List[Dict]] = {}
    for m in books.get(ml_series, []):
        if event_date(m) != game_date:
            continue
        games.setdefault(_game_code(m), []).append(m)

    for code, sides in games.items():
        named = [(rev.get(_norm(s.get("yes_sub_title"))), s) for s in sides
                 if _norm(s.get("yes_sub_title")) != "tie"]
        named = [(n, s) for n, s in named if n]
        if len(named) != 2:
            continue                       # unmappable or incomplete — skip
        # Ticker segment is <date><away><home>, so the side whose abbreviation
        # ends the segment is the home team.
        def _is_home(mk: Dict) -> bool:
            return code.endswith(str(mk.get("ticker") or "").rsplit("-", 1)[-1])
        home = next((n for n, s in named if _is_home(s)), None)
        away = next((n for n, s in named if n != home), None)
        if not home or not away:
            continue
        hm = next(s for n, s in named if n == home)
        am = next(s for n, s in named if n == away)

        entry: Dict[str, Any] = {
            "game_id": code, "sport": sport,
            "home_team": home, "away_team": away,
            "commence_time": str(hm.get("occurrence_datetime")
                                 or hm.get("expected_expiration_time") or ""),
            "moneyline": None, "spread": None, "total": None,
            "f5_moneyline": None, "f5_spread": None, "f5_total": None,
        }
        h_mid, a_mid = _mid(hm), _mid(am)
        if h_mid is not None and a_mid is not None:
            # Both sides are the SAME binary market, so they already sum to ~1
            # and need no de-vigging. (Contrast the Odds API totals path, which
            # keeps the overround — see the note in odds_client.)
            entry["moneyline"] = {"book": "kalshi", "home_prob": h_mid,
                                  "away_prob": a_mid}

        home_tok = _team_token(home)
        entry["spread"] = _best_spread(books.get(series.get("Spread") or "", []),
                                       code, home_tok)
        entry["total"] = _best_total(books.get(series.get("Total") or "", []), code)
        out.append(entry)
    return out


def _rungs(markets: List[Dict], code: str) -> List[Tuple[float, Dict]]:
    return sorted(((p, m) for m in markets if _game_code(m) == code
                   for p in [_point(m.get("yes_sub_title"))] if p is not None),
                  key=lambda t: t[0])


def _best_spread(markets: List[Dict], code: str, home_token: Optional[str]):
    """The listed rung closest to a coin flip — the market's own view of the line.

    Kalshi quotes a LADDER, not one line, so there is no single "the spread".
    The rung priced nearest 50% is the one the market treats as the handicap,
    which is the closest analogue to a sportsbook's posted line.
    """
    best = None
    for pt, m in _rungs(markets, code):
        mid = _mid(m)
        if mid is None:
            continue
        # The ticker names the favoured side; sign the line from our home team.
        fav_is_home = bool(home_token and str(m.get("ticker") or "")
                           .rsplit("-", 1)[-1].startswith(_abbr(home_token, m)))
        hp = mid if fav_is_home else 1 - mid
        if best is None or abs(hp - 0.5) < abs(best["home_prob"] - 0.5):
            best = {"book": "kalshi",
                    "home_spread": (-pt if fav_is_home else pt),
                    "home_prob": round(hp, 4), "away_prob": round(1 - hp, 4)}
    return best


def _abbr(token: str, market: Dict) -> str:
    """Kalshi's abbreviation for a token, read off this market's own ticker."""
    return str(market.get("ticker") or "").rsplit("-", 1)[-1][:3]


def _best_total(markets: List[Dict], code: str):
    best = None
    for pt, m in _rungs(markets, code):
        mid = _mid(m)
        if mid is None:
            continue
        if best is None or abs(mid - 0.5) < abs(best["over_prob"] - 0.5):
            best = {"book": "kalshi", "line": pt,
                    "over_prob": round(mid, 4), "under_prob": round(1 - mid, 4)}
    return best


def coverage_report(sport: str, game_date: str,
                    odds_games: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Reliability only — NOT price agreement, which the archive already settled.

    `odds_games` is the Odds API slate for the same day when available, so the
    trial can answer the question that matters: would pricing off Kalshi alone
    have produced a complete, sane slate this morning?
    """
    rep: Dict[str, Any] = {"sport": sport.upper(), "date": game_date,
                           "ok": False, "error": None}
    try:
        ks = discover(sport, game_date)
        rep.update(
            kalshi_games=len(ks),
            with_moneyline=sum(1 for g in ks if g.get("moneyline")),
            with_spread=sum(1 for g in ks if g.get("spread")),
            with_total=sum(1 for g in ks if g.get("total")),
            with_commence=sum(1 for g in ks if g.get("commence_time")),
            ok=True,
        )
        if odds_games is not None:
            rep["odds_games"] = len(odds_games)
            ok_keys = {_norm(f"{g.get('away_team')}@{g.get('home_team')}")
                       for g in odds_games}
            ks_keys = {_norm(f"{g['away_team']}@{g['home_team']}") for g in ks}
            rep["matched"] = len(ok_keys & ks_keys)
            rep["missing_from_kalshi"] = sorted(ok_keys - ks_keys)[:10]
            rep["extra_on_kalshi"] = sorted(ks_keys - ok_keys)[:10]
    except Exception as e:                      # never block the card
        rep["error"] = str(e)
        logger.warning(f"Kalshi coverage probe failed ({sport}): {e}")
    return rep


# ── Trial log ────────────────────────────────────────────────────────────────
COVERAGE_DIR = "state/price_coverage"


def record_coverage(run_date, sport: str, odds_games: Optional[List[Dict]] = None) -> bool:
    """Append one reliability row per (date, sport). Never raises.

    Behind the scenes by design: writes a log and nothing else. The card is
    priced by the Odds API exactly as before, so a failure here — or a Kalshi
    outage — cannot change what the user sees. That is the whole point of
    running it alongside rather than switching and watching.
    """
    try:
        import json
        import os
        from datetime import date as _date

        d = run_date.isoformat() if hasattr(run_date, "isoformat") else str(run_date)
        rep = coverage_report(sport, d, odds_games)
        rep["recorded_at"] = datetime.now(timezone.utc).isoformat()

        os.makedirs(COVERAGE_DIR, exist_ok=True)
        path = os.path.join(COVERAGE_DIR, f"{d[:7]}.json")
        shard = {"schema_version": 1, "month": d[:7], "entries": {}}
        if os.path.exists(path):
            try:
                shard = json.load(open(path)) or shard
            except Exception:
                pass
        shard.setdefault("entries", {})[f"{d}|{sport.upper()}"] = rep   # idempotent
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(shard, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.warning(f"Kalshi coverage log failed ({sport}, non-fatal): {e}")
        return False


def map_health(sport: str) -> Dict[str, Any]:
    """Is our team map still in step with Kalshi's live moneyline book?

    ⚠️ THIS IS THE ONE CHECK THAT MATTERS BEFORE TRUSTING KALSHI TO PRICE.
    Everything anchors on the moneyline book: we identify a game by matching its
    titles, then join every other book by ticker. Measured blast radii:

        one team renamed            -> only that team's games
        a spread/total book renamed -> that one market (and the ticker join
                                       now survives it; verified Oct 2026)
        the MONEYLINE book renamed  -> THAT WHOLE SPORT cannot be priced

    Kalshi renamed three books in a month, so this is not hypothetical. Run it
    before pricing a sport and fall back to the Odds API for that sport when it
    fails, rather than silently producing nothing.
    """
    out: Dict[str, Any] = {"sport": sport.upper(), "ok": False}
    try:
        series = SPORT_SERIES.get(sport.upper()) or {}
        ml = series.get("Moneyline")
        rev = _reverse_map(sport)
        if not ml or not rev:
            out["skipped"] = "no moneyline series or no static map"
            out["ok"] = True            # CFB/IPL are not expected to have one
            return out
        live = {_norm(m.get("yes_sub_title"))
                for m in fetch_open_markets(ml, max_pages=8)
                if m.get("yes_sub_title")} - {"tie"}
        unknown = sorted(t for t in live if t not in rev)
        out.update(live_tokens=len(live), unknown=unknown,
                   unknown_pct=round(100 * len(unknown) / len(live), 1) if live else 0.0,
                   ok=not unknown)
    except Exception as e:
        out["error"] = str(e)
    return out
