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

# NFL (32 teams). NFL is a BUDGET sport — real money — and without this map
# every NFL row was skipped as "unmapped", so NFL had NO CLV at all and the CLV
# governor could never gate it. Verified against the live KXNFLGAME tokens.
NFL_TEAM_TO_KALSHI = {
    "Arizona Cardinals": "Arizona", "Atlanta Falcons": "Atlanta",
    "Baltimore Ravens": "Baltimore", "Buffalo Bills": "Buffalo",
    "Carolina Panthers": "Carolina", "Chicago Bears": "Chicago",
    "Cincinnati Bengals": "Cincinnati", "Cleveland Browns": "Cleveland",
    "Dallas Cowboys": "Dallas", "Denver Broncos": "Denver",
    "Detroit Lions": "Detroit", "Green Bay Packers": "Green Bay",
    "Houston Texans": "Houston", "Indianapolis Colts": "Indianapolis",
    "Jacksonville Jaguars": "Jacksonville", "Kansas City Chiefs": "Kansas City",
    "Las Vegas Raiders": "Las Vegas", "Los Angeles Chargers": "Los Angeles C",
    "Los Angeles Rams": "Los Angeles R", "Miami Dolphins": "Miami",
    "Minnesota Vikings": "Minnesota", "New England Patriots": "New England",
    "New Orleans Saints": "New Orleans", "New York Giants": "New York G",
    "New York Jets": "New York J", "Philadelphia Eagles": "Philadelphia",
    "Pittsburgh Steelers": "Pittsburgh", "San Francisco 49ers": "San Francisco",
    "Seattle Seahawks": "Seattle", "Tampa Bay Buccaneers": "Tampa Bay",
    "Tennessee Titans": "Tennessee", "Washington Commanders": "Washington",
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
    """Kalshi token for a team, across every sport with a static map.

    MLB and NFL are looked up together: a name belongs to at most one league,
    so a merged lookup cannot collide, and keeping them separate is how NFL
    silently ended up with no CLV at all.
    """
    if not name:
        return None
    for table in (TEAM_TO_KALSHI, NFL_TEAM_TO_KALSHI):
        if name in table:
            return table[name]
    for table in (TEAM_TO_KALSHI, NFL_TEAM_TO_KALSHI):
        for full, tok in table.items():                # tolerate minor variants
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


def _game_code(market: Dict) -> str:
    """The game segment of a Kalshi ticker: KXNFLTOTAL-26SEP10SFLAR-49 -> 26SEP10SFLAR.

    Every market for one game carries the SAME segment in every series, so it
    is an exact join key across books — unlike the rules prose, which drifts
    between series for the same game (KXNFLSPREAD says "Los Angeles R", while
    KXNFLTOTAL says plain "Los Angeles" for the identical matchup).
    """
    parts = str(market.get("event_ticker") or market.get("ticker") or "").split("-")
    return parts[1] if len(parts) > 1 else ""


def _anchor_game_code(markets: Dict[str, List[Dict]], anchor_series: Optional[str],
                      home: str, away: str, game_date: str,
                      cfb: bool = False) -> Optional[str]:
    """Find a game's ticker segment via the MONEYLINE book.

    The moneyline series is the reliable anchor because each of its markets is
    titled with exactly one team token ("San Francisco" / "Los Angeles R"), so
    both sides match cleanly. Once the segment is known, every other book for
    that game can be filtered exactly, with no prose involved.
    """
    if not anchor_series:
        return None
    pool = [m for m in markets.get(anchor_series, []) if event_date(m) == game_date]
    by_code: Dict[str, List[Dict]] = {}
    for m in pool:
        by_code.setdefault(_game_code(m), []).append(m)
    for code, ms in by_code.items():
        if not code:
            continue
        titles = [str(m.get("yes_sub_title") or "") for m in ms]
        if cfb:
            ok_home = any(cfb_team_matches(home, t) for t in titles)
            ok_away = any(cfb_team_matches(away, t) for t in titles)
        else:
            norm = {_norm(t) for t in titles}
            ok_home, ok_away = _norm(home) in norm, _norm(away) in norm
        if ok_home and ok_away:
            return code
    return None


def _ladder_points(pool: List[Dict]) -> List[float]:
    """The strike points actually listed in a pool of markets, ascending.

    Kalshi ladders are VARIABLE RESOLUTION: every half point near a coin flip,
    then 2- and 3-point steps as the line widens (NFL has no 8.5/12.5/15.5 at
    all). So "it is a half point" does not imply "it is listed".
    """
    out = set()
    for m in pool:
        hit = re.search(r"(?:over|under)\s+(\d+(?:\.\d+)?)",
                        _norm(m.get("yes_sub_title")))
        if hit:
            out.add(float(hit.group(1)))
    return sorted(out)


def _select(pool: List[Dict], pick: Dict, bet_type: str, series: str,
            is_cfb: bool, home: str, away: str,
            markets: Dict[str, List[Dict]],
            require_quote: bool,
            tok_src: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
    """Choose the exact strike within an already game-filtered pool."""
    pick_txt = str(pick.get("pick") or "")
    if bet_type in ("Total", "F5 Total"):
        team = None
    elif is_cfb:
        # CFB must use the structural matcher here too — the static map has
        # no college teams, so this silently returned None and every CFB
        # pick failed to resolve even though the tokens were found.
        team = _cfb_token_from_markets(_strip_point(pick_txt),
                                       tok_src if tok_src else markets.get(series, []))
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
        # UNIT WORD VARIES BY SPORT: MLB says "runs", NFL/CFB/NBA say
        # "points". Hardcoding "runs" meant spread CLV never resolved for
        # ANY non-MLB sport — silently, since a miss just looks like a
        # market that is not listed. Match on the numeric part instead.
        if pt < 0:
            want, side = f"{team} wins by over {abs(pt)}", "yes"
        else:
            want, side = f"{opp} wins by over {pt}", "no"
    elif bet_type == "F5 Spread" and pt is not None:
        if abs(pt) == 0.5:      # ±0.5 in F5 is just "wins the first 5"
            want = f"{team} wins first 5 innings" if pt < 0 else f"{opp} wins first 5 innings"
            side = "yes" if pt < 0 else "no"
        elif pt < 0:
            want, side = f"{team} -{abs(pt)} first 5 innings", "yes"
        else:
            want, side = f"{opp} -{pt} first 5 innings", "no"
    elif bet_type in ("Total", "F5 Total") and pt is not None:
        # Same unit problem for totals: "Over 8.5 runs scored" (MLB) vs
        # "Over 63.5 points scored" (NFL/CFB). F5 keeps its own suffix
        # because a 5-inning total must not match the full-game one.
        want = (f"Over {pt} runs in the first 5" if bet_type == "F5 Total"
                else f"Over {pt}")
        side = "yes" if "over" in pick_txt.lower() else "no"

    if not want:
        return None
    target = _norm(want)
    m = next((x for x in pool if _norm(x.get("yes_sub_title")) == target), None)
    if m is None:
        # Prefix match, anchored at the START so "over 2.5" cannot match
        # "over 12.5" — that would silently price a completely different line.
        m = next((x for x in pool
                  if _norm(x.get("yes_sub_title")).startswith(target)), None)
    if m is None:
        m = next((x for x in pool if target in _norm(x.get("yes_sub_title"))), None)
    if m is None:
        # UNBUYABLE, not merely unlisted. We reached here with a NON-EMPTY pool,
        # so the game was found and its ladder was read — the specific strike we
        # quoted simply is not on it. Those two cases used to be the same silent
        # None, which is how NFL spread/total stayed broken from the day they
        # shipped: a resolver bug looks exactly like a market Kalshi does not
        # offer. Say which, and name the rungs that DO exist, because a line the
        # exchange will not sell is a line the user cannot actually bet.
        _av = _ladder_points(pool)
        if _av and pt is not None:
            _lo = max([x for x in _av if x < abs(pt)], default=None)
            _hi = min([x for x in _av if x > abs(pt)], default=None)
            logger.warning(
                f"Kalshi has no {abs(pt)} rung for {bet_type} in {series} "
                f"({pick.get('pick')!r}) — nearest listed: {_lo} / {_hi}. "
                f"This line is NOT BUYABLE; the quote cannot be measured or placed.")
        return None
    q = _quote(m, side)
    if not q:
        if require_quote:
            return None
        q = {"bid": None, "ask": None, "mid": None}
    return {"ticker": m.get("ticker"), "series": series, "side": side,
            "sub_title": m.get("yes_sub_title"),
            "open_interest": _f(m.get("open_interest_fp")) or 0.0, **q}

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
        anchor = (series_map or SERIES).get("F5 Moneyline" if bet_type.startswith("F5")
                                            else "Moneyline")
        # CFB tokens are derived from market TITLES, and only the moneyline book
        # titles a market with a bare team name ("Georgia Tech"). The spread and
        # total books title them as sentences ("Georgia Tech wins by over 6.5
        # points"), which the structural matcher cannot read — so deriving from
        # the pick's own series returned None and every CFB spread silently
        # failed to resolve. Always derive from the moneyline book.
        tok_src = markets.get(anchor or "", []) or markets.get(series, [])
        if is_cfb:
            home = _cfb_token_from_markets(pick.get("home_team"), tok_src)
            away = _cfb_token_from_markets(pick.get("away_team"), tok_src)
        else:
            home = _team_token(pick.get("home_team"))
            away = _team_token(pick.get("away_team"))
        if not home or not away:
            return None
        pool = [m for m in markets.get(series, []) if event_date(m) == game_date]
        # PREFERRED: join on the ticker's game segment, resolved off the
        # moneyline book. Exact, and immune to the prose drift below.
        code = _anchor_game_code(markets, anchor, home, away, game_date,
                                 cfb=is_cfb)
        if code:
            by_code = [m for m in pool if _game_code(m) == code]
            if by_code:
                return _select(by_code, pick, bet_type, series, is_cfb,
                               home, away, markets, require_quote, tok_src)
        # FALLBACK: identify the right GAME by event, not by prose. Kalshi's rules text is
        # inconsistent — the same series mixes "Dallas vs New York G" with
        # "NY Giants vs LA Rams" — so a rules-only filter silently dropped half
        # of NFL (16 of 32 moneylines) even with a correct team map.
        #
        # An event's markets collectively name both teams, so pool by
        # event_ticker whose combined text mentions both tokens, and fall back
        # to the per-market rules check when no event grouping is available.
        by_event: Dict[str, List[Dict]] = {}
        for m in pool:
            by_event.setdefault(m.get("event_ticker") or m.get("ticker") or "", []).append(m)
        matched: List[Dict] = []
        for _ev, ms in by_event.items():
            blob = _norm(" ".join(
                [str(x.get("yes_sub_title") or "") for x in ms] +
                [str(ms[0].get("rules_primary") or "")]))
            if _norm(home) in blob and _norm(away) in blob:
                matched.extend(ms)
        if not matched:
            matched = [m for m in pool
                       if _norm(home) in _norm(m.get("rules_primary"))
                       and _norm(away) in _norm(m.get("rules_primary"))]
        pool = matched
        if not pool:
            return None

        return _select(pool, pick, bet_type, series, is_cfb, home, away,
                       markets, require_quote, tok_src)
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
