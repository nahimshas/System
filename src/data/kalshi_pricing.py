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
# Leagues whose ticker segment is <date><HOME><AWAY> rather than
# <date><AWAY><HOME>. Verified against ESPN homeAway flags Oct 2026.
_CODE_HOME_FIRST = {"MLS", "LIGAMX"}

# Leagues whose moneyline is 3-way (home / draw / away).
_THREE_WAY = {"MLS", "LIGAMX", "WC"}

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


# Maximum bid-ask spread, in cents, for a rung to count as PRICED.
# ⚠️ Without this, an untraded placeholder quote (bid 2c / ask 98c, open
# interest 0) yields a "mid" of 0.50 that looks like a real coin-flip price.
# That is exactly how MLS totals produced "Over 0.5 goals at 50%" — a nonsense
# line the model then found a 10% edge against. Measured medians: NFL spread
# and total 2.0c, NBA 2.0/3.0c, NHL spread 3.0c, but NHL total 19.0c and MLS
# spread/total ~40c. A 10c cap keeps 88-100% of the real books and discards
# the empty ones.
MAX_SPREAD_CENTS = 10.0


def _mid(market: Dict) -> Optional[float]:
    """Mid of the two-sided book, or None when it is not quoted both ways.

    Mid rather than ask: the ask carries the ~1% overround we measure
    separately in the execution log, and mixing the two would double-count it.
    """
    b, a = _f(market.get("yes_bid_dollars")), _f(market.get("yes_ask_dollars"))
    if b is None or a is None or not (0 < b < a < 1):
        return None
    if (a - b) * 100.0 > MAX_SPREAD_CENTS:
        return None            # no real market here — see MAX_SPREAD_CENTS
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
        # ⚠️ TICKER ORDER IS NOT THE SAME ACROSS LEAGUES.
        #   US sports : <date><AWAY><HOME>   e.g. 26OCT04 LAR PHI  (LAR @ PHI)
        #   SOCCER    : <date><HOME><AWAY>   e.g. 26OCT06 CHI VAN  (VAN @ CHI)
        # Both verified against ESPN's own homeAway flags. Assuming one order
        # everywhere silently SWAPS home and away for soccer, which is worse
        # than having no price at all — every edge would be computed against the
        # wrong team while looking perfectly healthy. Caught by the coverage
        # trial's `matched` count sitting at 0 for MLS and LigaMX across 15
        # slates while games were being discovered normally.
        _date_len = len(re.match(r"^\d{2}[A-Z]{3}\d{2}", code).group(0)) if re.match(
            r"^\d{2}[A-Z]{3}\d{2}", code) else 7
        _rest = code[_date_len:]

        def _is_home(mk: Dict) -> bool:
            ab = str(mk.get("ticker") or "").rsplit("-", 1)[-1]
            if not ab:
                return False
            return _rest.startswith(ab) if sport in _CODE_HOME_FIRST \
                else _rest.endswith(ab)
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

        # ⚠️ SOCCER MONEYLINES ARE THREE-WAY. analyze_mls_game bails on
        # `if not ml or "draw_prob" not in ml`, so a 2-way moneyline produced
        # ZERO recs for MLS and LigaMX across every slate of the trial while
        # discovery looked perfectly healthy. Kalshi carries the draw as the
        # "Tie" market inside the same book.
        d_mid = None
        if sport in _THREE_WAY:
            tie = next((m for m in sides if _norm(m.get("yes_sub_title")) == "tie"), None)
            d_mid = _mid(tie) if tie is not None else None
            if h_mid is None or a_mid is None or d_mid is None:
                h_mid = a_mid = None          # incomplete 3-way → no moneyline
            else:
                # Kalshi quotes the three independently; normalise so they sum
                # to 1 the way the analyzer expects of a no-vig 3-way price.
                tot = h_mid + d_mid + a_mid
                if tot > 0:
                    h_mid, d_mid, a_mid = (round(h_mid / tot, 4),
                                           round(d_mid / tot, 4),
                                           round(a_mid / tot, 4))

        if h_mid is not None and a_mid is not None:
            # Both sides are the SAME binary market, so they already sum to ~1
            # and need no de-vigging. (Contrast the Odds API totals path, which
            # keeps the overround — see the note in odds_client.)
            entry["moneyline"] = {"book": "kalshi", "home_prob": h_mid,
                                  "away_prob": a_mid}
            if d_mid is not None:
                entry["moneyline"]["draw_prob"] = d_mid

        # ⚠️ Pass the home team's TICKER ABBREVIATION, not its display token.
        # `_team_token` returns "San Antonio", while spread tickers say "SAS",
        # so a display token can never match a ticker suffix. `hm` is the home
        # moneyline market, resolved by the same ticker-order logic above, so
        # its own suffix IS the home abbreviation — exact, with no second map
        # to drift out of sync.
        home_abbr = str(hm.get("ticker") or "").rsplit("-", 1)[-1]
        entry["spread"] = _best_spread(books.get(series.get("Spread") or "", []),
                                       code, home_abbr)
        entry["total"] = _best_total(books.get(series.get("Total") or "", []), code)
        out.append(entry)
    return out


def _rungs(markets: List[Dict], code: str) -> List[Tuple[float, Dict]]:
    return sorted(((p, m) for m in markets if _game_code(m) == code
                   for p in [_point(m.get("yes_sub_title"))] if p is not None),
                  key=lambda t: t[0])


def _best_spread(markets: List[Dict], code: str, home_abbr: Optional[str]):
    """The listed rung closest to a coin flip — the market's own view of the line.

    Kalshi quotes a LADDER, not one line, so there is no single "the spread".
    The rung priced nearest 50% is the one the market treats as the handicap,
    which is the closest analogue to a sportsbook's posted line.

    ⚠️ `home_abbr` MUST be the home team's ticker abbreviation ("UTA"), not its
    display token ("Utah"). Each rung's ticker names the FAVOURED side
    (`...-SAS14` = "San Antonio wins by over 13.5"), so the sign of the line
    depends entirely on whether that side is our home team.

    This was broken on the first Kalshi slate (Oct 28 2026) and it is the
    costliest kind of bug — silently wrong, not absent. The old orientation
    test compared a rung's suffix against the first three characters of THAT
    SAME SUFFIX, so `"SAS14".startswith("SAS")` made `fav_is_home`
    unconditionally True. Every spread was labelled "home favoured by X",
    right by luck when the home team was favoured and inverted otherwise. On
    Oct 28 every one of 8 NBA spread picks came out as a "+X.5" underdog
    priced near 50c, including "San Antonio +12.5" when SAS were in fact ~13
    point FAVOURITES — and the same game's moneyline correctly had them at
    84.5%. Two NHL spreads and one MLB spread were bet real money on inverted
    lines. ⚠️ A game dict whose moneyline and spread disagree about who is
    favoured is the signature; assert on it, do not eyeball it.
    """
    best = None
    for pt, m in _rungs(markets, code):
        mid = _mid(m)
        if mid is None:
            continue
        fav_is_home = bool(home_abbr) and _fav_abbr(m) == home_abbr
        hp = mid if fav_is_home else 1 - mid
        if best is None or abs(hp - 0.5) < abs(best["home_prob"] - 0.5):
            best = {"book": "kalshi",
                    "home_spread": (-pt if fav_is_home else pt),
                    "home_prob": round(hp, 4), "away_prob": round(1 - hp, 4)}
    return best


def _fav_abbr(market: Dict) -> str:
    """The team abbreviation a rung's ticker names, e.g. ...-SAS14 -> "SAS"."""
    suffix = str(market.get("ticker") or "").rsplit("-", 1)[-1]
    m = re.match(r"^([A-Za-z]+)", suffix)
    return m.group(1) if m else ""


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


# ── Card comparison (Oct 2026 trial) ─────────────────────────────────────────
def _pick_key(rec) -> tuple:
    """Identity of a recommendation, independent of price source."""
    return (getattr(rec, "game", ""), getattr(rec, "bet_type", ""),
            str(getattr(rec, "pick", "")))


def compare_picks(module, ctx, odds_recs, sport: str, game_date: str,
                  top_n: int = 5) -> Dict[str, Any]:
    """Would the CARD change if this sport were priced off Kalshi?

    Runs the sport's own analyzer a second time over Kalshi-priced games, using
    the SAME context (stats, injuries, rest) so the only variable is the price
    source, and compares the resulting recommendations.

    ⚠️ WHAT THIS DOES NOT DO: it does not replay budget routing. Slot limits,
    one-bet-per-game, CLV gates and Kelly sizing all sit in main.py and run
    ACROSS sports, so they cannot be reproduced per sport here. Read `top_overlap`
    as "would the same bets be in contention", not "the card would be identical".
    The full-set fields are the honest measure; top_n is the headline.
    """
    out: Dict[str, Any] = {"sport": sport.upper(), "date": game_date, "ok": False}
    try:
        kgames = discover(sport, game_date)
        out["kalshi_games"] = len(kgames)
        if not kgames:
            out["skipped"] = "no Kalshi games discovered"
            out["ok"] = True
            return out

        krecs = module.analyze_games(kgames, ctx) or []
        o_keys = {_pick_key(r) for r in (odds_recs or [])}
        k_keys = {_pick_key(r) for r in krecs}

        def _top(recs):
            return [_pick_key(r) for r in
                    sorted(recs or [], key=lambda r: -float(getattr(r, "edge", 0) or 0))[:top_n]]

        o_top, k_top = _top(odds_recs), _top(krecs)
        out.update(
            odds_recs=len(o_keys), kalshi_recs=len(k_keys),
            both=len(o_keys & k_keys),
            only_odds=sorted(f"{g}|{b}|{p}" for g, b, p in (o_keys - k_keys))[:8],
            only_kalshi=sorted(f"{g}|{b}|{p}" for g, b, p in (k_keys - o_keys))[:8],
            top_overlap=len(set(o_top) & set(k_top)),
            top_n=top_n,
            ok=True,
        )
    except Exception as e:                       # never block the card
        out["error"] = str(e)
        logger.warning(f"Kalshi card comparison failed ({sport}): {e}")
    return out


def record_card_comparison(run_date, sport: str, module, ctx, odds_recs) -> bool:
    """Append the card comparison onto the same (date, sport) coverage row."""
    try:
        import json
        import os
        d = run_date.isoformat() if hasattr(run_date, "isoformat") else str(run_date)
        rep = compare_picks(module, ctx, odds_recs, sport, d)

        os.makedirs(COVERAGE_DIR, exist_ok=True)
        path = os.path.join(COVERAGE_DIR, f"{d[:7]}.json")
        shard = {"schema_version": 1, "month": d[:7], "entries": {}}
        if os.path.exists(path):
            try:
                shard = json.load(open(path)) or shard
            except Exception:
                pass
        key = f"{d}|{sport.upper()}"
        row = shard.setdefault("entries", {}).get(key, {})
        row["card_compare"] = rep
        shard["entries"][key] = row
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(shard, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.warning(f"Kalshi card comparison log failed ({sport}): {e}")
        return False


# ── Primary price source routing (Oct 27 2026) ───────────────────────────────
# Sports Kalshi can price end-to-end: a verified series map AND a team map that
# resolves both sides. Anything absent here keeps the Odds API.
#
#   CFB — no static team map by design (279 tokens that rebrand; matched
#         structurally for CLV, which is not enough to DISCOVER a slate).
#   IPL — Kalshi lists no per-match markets at all.
#   WC  — out of season, series ticker never verified against a live book.
KALSHI_PRICEABLE = {"MLB", "NBA", "NFL", "NHL", "WNBA", "MLS", "LIGAMX"}


def price_source_for(sport: str) -> str:
    """'kalshi' or 'oddsapi' for this sport, honouring the env override."""
    import os
    forced = (os.getenv("PRICE_SOURCE") or "").strip().lower()
    if forced in ("oddsapi", "odds_api"):
        return "oddsapi"
    if forced == "kalshi":
        return "kalshi"
    return "kalshi" if sport.upper() in KALSHI_PRICEABLE else "oddsapi"


def _espn_game_count(sport: str, game_date: str) -> Optional[int]:
    """How many games ESPN lists — the free second opinion on an empty slate.

    Without this the fallback warning fires on every quiet day: NFL on a
    Tuesday, WNBA out of season, a soccer league with no fixtures. An alert
    that cries wolf daily is one the reader learns to ignore, which is exactly
    how the earlier silent failures survived. Returns None when ESPN cannot be
    reached, and an unreachable ESPN must NOT be read as "no games" — that
    would silence the very gap this check exists to find.
    """
    try:
        import requests
        from src.data.outcome_checker import (ESPN_BASE, ESPN_SPORT_PATHS,
                                              ESPN_WATCHLIST_PATHS)
        path = (ESPN_SPORT_PATHS.get(sport.upper())
                or ESPN_WATCHLIST_PATHS.get(sport.upper()))
        if not path:
            return None
        params = {"dates": game_date.replace("-", "")}
        if sport.upper() == "CFB":
            params.update({"groups": 80, "limit": 300})
        r = requests.get(f"{ESPN_BASE}/{path}/scoreboard", params=params, timeout=12)
        r.raise_for_status()
        return len((r.json() or {}).get("events", []) or [])
    except Exception:
        return None


def fetch_games_routed(slug: str, module, today_str: str,
                       odds_key_available: bool = True):
    """Game list for one sport, from whichever source can price it.

    Returns (games, source, fell_back).

    ⚠️ FALLS BACK RATHER THAN FAILING. If Kalshi yields nothing for a sport
    that should be priceable, the Odds API is tried instead when a key exists.
    A silent empty slate is the worst outcome: the sport simply vanishes from
    the card and nothing says why — the exact failure mode that hid the CFB and
    NFL bugs for days. The caller is told so it can be surfaced.
    """
    sport = slug.upper()
    source = price_source_for(sport)

    if source == "kalshi":
        try:
            games = discover(sport, today_str)
        except Exception as e:
            logger.warning(f"{sport}: Kalshi pricing raised ({e}) — falling back")
            games = []
        if games:
            logger.info(f"{sport}: priced {len(games)} game(s) from Kalshi")
            # Props come from Kalshi too. Stamped HERE rather than inside
            # discover() because this is the one place that knows a Kalshi
            # slate was built — the Odds API path stamps them inside its own
            # fetch_games(), which Kalshi bypasses entirely. Isolated: a props
            # failure must never cost us the game card.
            try:
                # Some sports' fetch_games has SIDE EFFECTS the rest of the
                # sport depends on (MLB's probable-pitcher merge and its
                # _last_schedule stash). Kalshi bypasses fetch_games, so a
                # module may expose this hook to re-add them. Registry-driven
                # by duck-typing rather than a per-sport branch here.
                _enrich = getattr(module, "enrich_kalshi_games", None)
                if callable(_enrich):
                    _enrich(games, today_str)
            except Exception as e:
                logger.warning(f"{sport}: Kalshi game enrichment failed: {e}")
            try:
                from src.data.kalshi_props import fetch_props_for_games
                fetch_props_for_games(sport, games, today_str)
            except Exception as e:
                logger.warning(f"{sport}: Kalshi props stamping failed: {e}")
                for _g in games:
                    _g.setdefault("player_props", {})
            return games, "kalshi", False
        # Is this an empty slate, or a broken one? ESPN decides.
        espn_n = _espn_game_count(sport, today_str)
        genuinely_empty = (espn_n == 0)
        if genuinely_empty:
            logger.info(f"{sport}: no games today (ESPN agrees) — nothing to price")
            return [], "kalshi", False
        if odds_key_available:
            logger.warning(
                f"{sport}: Kalshi returned NO games but ESPN lists "
                f"{espn_n if espn_n is not None else '?'} — falling back to the "
                f"Odds API. This is a real gap, not a quiet day.")
            try:
                return (module.fetch_games(today_str) or []), "oddsapi", True
            except Exception as e:
                logger.warning(f"{sport}: Odds API fallback also failed: {e}")
                return [], "kalshi", True
        logger.warning(
            f"{sport}: Kalshi returned no games, ESPN lists "
            f"{espn_n if espn_n is not None else '?'}, and there is no Odds API "
            f"key — THIS SPORT HAS NO PRICES TODAY.")
        return [], "kalshi", True

    if not odds_key_available:
        return [], "oddsapi", False
    try:
        games = module.fetch_games(today_str) or []
    except Exception as e:
        logger.warning(f"{sport}: Odds API fetch failed: {e}")
        return [], "oddsapi", False
    if games:
        return games, "oddsapi", False
    # ⚠️ A dead key is now the likeliest failure on this path — the paid
    # subscription is cancelled and CFB/IPL/WC run on a free-tier key that can
    # expire or exhaust its quota. get_odds() swallows HTTP errors and returns
    # [], which is indistinguishable from a quiet day. Ask ESPN. IPL has no
    # ESPN scoreboard at all (espn_n is None), so only a positive count warns.
    espn_n = _espn_game_count(sport, today_str)
    if espn_n:
        logger.warning(
            f"{sport}: the Odds API returned NO games but ESPN lists {espn_n} — "
            f"check the ODDS_API_KEY secret (expired key? free-tier quota "
            f"exhausted?). This is a real gap, not a quiet day.")
        return [], "oddsapi", True
    return games, "oddsapi", False
