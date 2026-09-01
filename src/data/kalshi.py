"""
Kalshi market-data client — EXECUTION MEASUREMENT ONLY.

Robinhood's sports event contracts are Kalshi-powered, and the price shown in
the Robinhood app is Kalshi's raw ask (verified Aug 24 2026: a Rays ML quoted
0.56 in-app matched Kalshi's 0.56 ask to the cent). Kalshi's market-data API is
public and needs no key, so we can see the exact book we trade against.

Purpose: answer the question that decides whether MLB can be profitable —
"what does our edge look like if we only ever buy at the BID, and how much of
that saving does adverse selection take back?" We pay ask + $0.02 in fees
today, which is ~2pp over fair value on a coin-flip contract, and our measured
edge is ~0. Execution, not prediction, is the biggest single lever left.

⚠️ THIS MUST NEVER BLOCK THE DAILY CARD. Aug 6 2026: F5 markets were appended
to the bulk odds request, the request 422'd, and NO CARD was produced. Every
entry point here is exception-safe and returns empty on failure, and the whole
layer is behind the ENABLE_KALSHI_SNAPSHOT kill switch.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_TIMEOUT = 20

# Series we care about — one per market type we bet.
SERIES = {
    "Moneyline":    "KXMLBGAME",
    "Spread":       "KXMLBSPREAD",
    "Total":        "KXMLBTOTAL",
    "F5 Moneyline": "KXMLBF5",
    "F5 Tie":       "KXMLBF5",
    "F5 Spread":    "KXMLBF5SPREAD",
    "F5 Total":     "KXMLBF5TOTAL",
}

# Our full team names → Kalshi's short token (verified against the live API).
TEAM_TO_KALSHI = {
    "Arizona Diamondbacks": "Arizona", "Atlanta Braves": "Atlanta",
    "Baltimore Orioles": "Baltimore", "Boston Red Sox": "Boston",
    "Chicago Cubs": "Chicago C", "Chicago White Sox": "Chicago WS",
    "Cincinnati Reds": "Cincinnati", "Cleveland Guardians": "Cleveland",
    "Colorado Rockies": "Colorado", "Detroit Tigers": "Detroit",
    "Houston Astros": "Houston", "Kansas City Royals": "Kansas City",
    "Los Angeles Angels": "Los Angeles A", "Los Angeles Dodgers": "Los Angeles D",
    "Miami Marlins": "Miami", "Milwaukee Brewers": "Milwaukee",
    "Minnesota Twins": "Minnesota", "New York Mets": "New York M",
    "New York Yankees": "New York Y", "Athletics": "A's",
    "Oakland Athletics": "A's", "Philadelphia Phillies": "Philadelphia",
    "Pittsburgh Pirates": "Pittsburgh", "San Diego Padres": "San Diego",
    "San Francisco Giants": "San Francisco", "Seattle Mariners": "Seattle",
    "St. Louis Cardinals": "St. Louis", "Tampa Bay Rays": "Tampa Bay",
    "Texas Rangers": "Texas", "Toronto Blue Jays": "Toronto",
    "Washington Nationals": "Washington",
}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
# MLB tickers carry a start time (26AUG242145CINSF); WNBA/MLS/NBA do not
# (26AUG24ATLLA). The time group is therefore OPTIONAL — requiring it made
# event_date() return None for every non-MLB market, which silently
# excluded all watchlist sports from Kalshi CLV.
_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(\d{4})?(?=[A-Z])")


def enabled() -> bool:
    """Kill switch — set ENABLE_KALSHI_SNAPSHOT=false to disable entirely."""
    return os.environ.get("ENABLE_KALSHI_SNAPSHOT", "true").lower() != "false"


# Kalshi rate-limits the public API. A backfill makes one candlestick request
# per pick, which trips 429 almost immediately without pacing, so requests are
# spaced and 429s are retried with backoff rather than dropped (a dropped
# request costs a retry slot and the entry gets marked unmatchable).
_MIN_INTERVAL = 0.12
_RETRY_ON = (429, 500, 502, 503, 504)
_MAX_RETRIES = 4
_last_request_at = 0.0


def _get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    global _last_request_at
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    delay = 0.5
    for attempt in range(_MAX_RETRIES):
        gap = time.monotonic() - _last_request_at
        if gap < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - gap)
        try:
            req = urllib.request.Request(url, headers={"accept": "application/json"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                _last_request_at = time.monotonic()
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            _last_request_at = time.monotonic()
            if e.code in _RETRY_ON and attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            logger.warning(f"Kalshi request failed ({path}): {e}")
            return None
        except Exception as e:
            _last_request_at = time.monotonic()
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            logger.warning(f"Kalshi request failed ({path}): {e}")
            return None
    return None


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def fetch_open_markets(series_ticker: str, max_pages: int = 6) -> List[Dict]:
    """All open markets for a series. Returns [] on any failure."""
    out: List[Dict] = []
    cursor = None
    for _ in range(max_pages):
        params = {"series_ticker": series_ticker, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        if not d:
            break
        out.extend(d.get("markets", []))
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def event_date(market: Dict) -> Optional[str]:
    """ET calendar date of the game, parsed from the event ticker.

    'KXMLBGAME-26AUG241840TBDET' -> '2026-08-24'. Kalshi schedules by Eastern
    time, which is also how MLB lists games, so this is the right key to match
    a pick against (a UTC date would roll night games to the next day).
    """
    m = _EVENT_DATE_RE.search(market.get("event_ticker") or market.get("ticker") or "")
    if not m:
        return None
    yy, mon, dd, _hhmm = m.groups()
    mth = _MONTHS.get(mon)
    if not mth:
        return None
    return f"20{yy}-{mth:02d}-{int(dd):02d}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def _pick_point(pick: str) -> Optional[float]:
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", str(pick or ""))
    return float(m.group(1)) if m else None


# College football has ~134 FBS teams (279 tokens on Kalshi once FCS is
# included) and the two feeds name them differently: Kalshi says "Boise St.",
# the Odds API says "Boise State Broncos". A static map would be 279 rows and
# would rot every time a school rebrands, so CFB matches structurally instead:
# normalise St./State, drop punctuation, and require the Kalshi token to be a
# leading subset of the full name.
# Words that continue a SCHOOL NAME rather than begin a mascot. If one of these
# follows a candidate token, the token is a different school.
_CFB_QUALIFIERS = {
    "state", "tech", "aandm", "am", "southern", "northern", "eastern",
    "western", "central", "international", "atlantic", "pacific", "christian",
    "baptist", "methodist", "military", "polytechnic", "chicago", "dominion",
}


def _cfb_norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())
    s = re.sub(r"\bst\b\.?", "state", s)
    s = re.sub(r"\bu\b|\buniversity\b|\bcollege\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cfb_team_matches(full_name: str, kalshi_token: str) -> bool:
    """True when a Kalshi CFB token names the same school as `full_name`."""
    a, b = _cfb_norm(full_name), _cfb_norm(kalshi_token)
    if not a or not b:
        return False
    if a == b:
        return True
    aw, bw = a.split(), b.split()
    # Kalshi's token is the school; the Odds API adds a mascot. Require every
    # Kalshi word to appear IN ORDER at the start of the full name.
    if not (len(bw) <= len(aw) and aw[:len(bw)] == bw):
        return False
    # ...but a prefix alone is not enough: "Ohio" would swallow "Ohio State
    # Buckeyes", and those are two different FBS programmes. If what remains
    # after the token is itself a school QUALIFIER rather than a mascot, the
    # token names a different school and must not match.
    rest = aw[len(bw):]
    return not (rest and rest[0] in _CFB_QUALIFIERS)


def _cfb_token_from_markets(full_name: str, markets: List[Dict]) -> Optional[str]:
    """Pick the Kalshi CFB token for a team, requiring an UNAMBIGUOUS match.

    Several tokens can prefix-match one school ("Ohio" and "Ohio St."), so the
    longest match wins and a tie means we refuse rather than guess — a wrong
    market is far worse than a missing one.
    """
    if not full_name:
        return None
    toks = {m.get("yes_sub_title") for m in markets if m.get("yes_sub_title")}
    hits = [t for t in toks if cfb_team_matches(full_name, t)]
    if not hits:
        return None
    hits.sort(key=lambda t: len(_cfb_norm(t)), reverse=True)
    if len(hits) > 1 and len(_cfb_norm(hits[0])) == len(_cfb_norm(hits[1])):
        logger.debug(f"CFB token ambiguous for {full_name!r}: {hits[:2]}")
        return None
    return hits[0]


def _team_token(name: str) -> Optional[str]:
    if not name:
        return None
    if name in TEAM_TO_KALSHI:
        return TEAM_TO_KALSHI[name]
    for full, tok in TEAM_TO_KALSHI.items():           # tolerate minor variants
        if _norm(full) == _norm(name):
            return tok
    return None


def _quote(market: Dict, side: str) -> Optional[Dict[str, float]]:
    """Bid/ask for the side we would actually buy.

    Buying NO is buying the complement, so its book is the mirror of YES:
    no_bid = 1 - yes_ask, no_ask = 1 - yes_bid.
    """
    yb, ya = _f(market.get("yes_bid_dollars")), _f(market.get("yes_ask_dollars"))
    if yb is None or ya is None:
        nb, na = _f(market.get("no_bid_dollars")), _f(market.get("no_ask_dollars"))
        if nb is None or na is None:
            return None
        yb, ya = round(1 - na, 4), round(1 - nb, 4)
    bid, ask = (yb, ya) if side == "yes" else (round(1 - ya, 4), round(1 - yb, 4))
    if not (0 < bid <= ask < 1):
        return None
    return {"bid": round(bid, 4), "ask": round(ask, 4), "mid": round((bid + ask) / 2, 4)}


def resolve_pick(pick: Dict, markets: Dict[str, List[Dict]],
                 game_date: str, require_quote: bool = True,
                 series_map: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """Map one of our picks onto the Kalshi market we would buy.

    Returns {ticker, side, bid, ask, mid, open_interest, sub_title} or None
    when no matching market exists (some derivative lines simply aren't listed).

    require_quote=False keeps the match when there is no live two-sided book.
    SETTLED markets have no quotes, so CLV backfill — which only needs the
    ticker and side to replay candlesticks — must pass False or every past game
    silently fails to resolve.
    """
    try:
        bet_type = str(pick.get("bet_type") or "")
        # SERIES is the MLB map. Other sports MUST pass their own, or every
        # lookup lands in the MLB books and silently resolves nothing.
        series = (series_map or SERIES).get(bet_type)
        if not series:
            return None
        pick_txt_early = str(pick.get("pick") or "")
        # A ±0.5 first-five spread is just "wins the first 5" — Kalshi lists no
        # 0.5 line in KXMLBF5SPREAD, so the book to read is the F5 winner one.
        # This has to be decided BEFORE the pool is built, or the empty spread
        # pool short-circuits and the pick looks unmatchable.
        if bet_type == "F5 Spread" and abs(_pick_point(pick_txt_early) or 0) == 0.5:
            series = "KXMLBF5"
        is_cfb = series.startswith("KXNCAAF")
        if is_cfb:
            # CFB has no static token map — derive both tokens from the market
            # list itself using the structural matcher.
            home = _cfb_token_from_markets(pick.get("home_team"), markets.get(series, []))
            away = _cfb_token_from_markets(pick.get("away_team"), markets.get(series, []))
        else:
            home = _team_token(pick.get("home_team"))
            away = _team_token(pick.get("away_team"))
        if not home or not away:
            return None
        pool = [m for m in markets.get(series, []) if event_date(m) == game_date]
        # Both teams must appear in the market's own rules text.
        pool = [m for m in pool
                if _norm(home) in _norm(m.get("rules_primary"))
                and _norm(away) in _norm(m.get("rules_primary"))]
        if not pool:
            return None

        pick_txt = str(pick.get("pick") or "")
        if bet_type in ("Total", "F5 Total"):
            team = None
        elif is_cfb:
            # CFB must use the structural matcher here too — the static map has
            # no college teams, so this silently returned None and every CFB
            # pick failed to resolve even though the tokens were found.
            team = _cfb_token_from_markets(_strip_point(pick_txt), markets.get(series, []))
        else:
            team = _team_token(_strip_point(pick_txt))
        opp = away if team == home else home
        pt = _pick_point(pick_txt)

        want, side = None, "yes"
        if bet_type == "Moneyline":
            want, side = team, "yes"
        elif bet_type == "F5 Moneyline":
            want, side = f"{team} wins first 5 innings", "yes"
        elif bet_type == "F5 Tie":
            want, side = "Tie", "yes"
        elif bet_type == "Spread" and pt is not None:
            if pt < 0:
                want, side = f"{team} wins by over {abs(pt)} runs", "yes"
            else:
                want, side = f"{opp} wins by over {pt} runs", "no"
        elif bet_type == "F5 Spread" and pt is not None:
            if abs(pt) == 0.5:      # ±0.5 in F5 is just "wins the first 5"
                want = f"{team} wins first 5 innings" if pt < 0 else f"{opp} wins first 5 innings"
                side = "yes" if pt < 0 else "no"
            elif pt < 0:
                want, side = f"{team} -{abs(pt)} first 5 innings", "yes"
            else:
                want, side = f"{opp} -{pt} first 5 innings", "no"
        elif bet_type in ("Total", "F5 Total") and pt is not None:
            unit = "runs in the first 5" if bet_type == "F5 Total" else "runs scored"
            want = f"Over {pt} {unit}"
            side = "yes" if "over" in pick_txt.lower() else "no"

        if not want:
            return None
        target = _norm(want)
        m = next((x for x in pool if _norm(x.get("yes_sub_title")) == target), None)
        if m is None:
            m = next((x for x in pool if target in _norm(x.get("yes_sub_title"))), None)
        if m is None:
            return None
        q = _quote(m, side)
        if not q:
            if require_quote:
                return None
            q = {"bid": None, "ask": None, "mid": None}
        return {"ticker": m.get("ticker"), "series": series, "side": side,
                "sub_title": m.get("yes_sub_title"),
                "open_interest": _f(m.get("open_interest_fp")) or 0.0, **q}
    except Exception as e:
        logger.warning(f"Kalshi resolve failed for {pick.get('pick')!r}: {e}")
        return None


def _strip_point(pick: str) -> str:
    return re.sub(r"\s*[+-]?\d+(?:\.\d+)?\s*$", "", str(pick or "")).strip()


def fetch_candles(series: str, ticker: str, start_ts: int, end_ts: int,
                  period_interval: int = 60) -> List[Dict]:
    """Hourly OHLC of price/bid/ask for a market. Used after the game to ask
    'would a resting bid have filled, and where did it close?'"""
    d = _get(f"/series/{series}/markets/{ticker}/candlesticks",
             {"start_ts": start_ts, "end_ts": end_ts,
              "period_interval": period_interval})
    return (d or {}).get("candlesticks", []) or []


def snapshot(picks: List[Dict], game_date: str) -> List[Dict[str, Any]]:
    """Bid/ask snapshot for every pick we can match. Never raises."""
    if not enabled():
        logger.info("Kalshi snapshot disabled via ENABLE_KALSHI_SNAPSHOT")
        return []
    try:
        needed = {SERIES[b] for b in {str(p.get("bet_type")) for p in picks}
                  if b in SERIES}
        if any(str(p.get("bet_type")) == "F5 Spread" for p in picks):
            needed.add("KXMLBF5")           # ±0.5 resolves against the F5 winner book
        markets = {s: fetch_open_markets(s) for s in sorted(needed)}
        total = sum(len(v) for v in markets.values())
        if not total:
            logger.warning("Kalshi snapshot: no markets returned — skipping")
            return []
        out = []
        for p in picks:
            r = resolve_pick(p, markets, game_date)
            if r:
                out.append({"pick": p, "kalshi": r})
        logger.info(f"Kalshi snapshot: matched {len(out)}/{len(picks)} picks "
                    f"across {total} markets")
        return out
    except Exception as e:
        logger.error(f"Kalshi snapshot failed (non-fatal): {e}")
        return []
