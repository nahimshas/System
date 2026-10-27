"""Player props from Kalshi order books.

⚠️ **Kalshi's ladder IS the market — this module does not translate it into a
sportsbook's Over/Under.** Kalshi lists a separate binary contract per
threshold ("Jayson Tatum: 25+"), which is exactly `P(stat >= 25)`. A
sportsbook posts one half-point line per player and charges vig around it.
Those are different objects, and pretending the ladder is a book line was the
thing to avoid: there is no single "the line" to pick, so we price EVERY rung
and let edge ranking choose, rather than inventing a line and throwing the
rest of the ladder away.

How a rung reaches the existing analyzer unchanged: `_cover_prob` computes
`P(stat > market_line)`, so a rung at N is passed as `line = N - 0.5`.
That is not a fudge — it is the continuity correction, and it is exact for the
Poisson markets (`int(N-0.5) == N-1`, and `sf(N-1) == P(X >= N)`).

Replaces the Odds API props path, which died with the paid subscription on
Oct 27 2026. The claim that Kalshi has no player markets was WRONG; all
eleven markets we model are listed and two-sided. See CLAUDE.md.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.data.kalshi import (_f, _game_code, _norm, enabled, event_date,
                             fetch_open_markets)

logger = logging.getLogger(__name__)

# Our prop label -> Kalshi series ticker. Verified live Oct 27 2026: every one
# of these returns open, two-sided markets. ⚠️ Keyed by the label the analyzer
# already uses (PROP_MARKET_LABEL's values) so no analyzer code changes.
PROP_SERIES: Dict[str, Dict[str, str]] = {
    "NBA": {
        "Points Over":   "KXNBAPTS",
        "Rebounds Over": "KXNBAREB",
        "Assists Over":  "KXNBAAST",
        "Steals Over":   "KXNBASTL",
        "Blocks Over":   "KXNBABLK",
        "Threes Over":   "KXNBA3PT",
    },
    "MLB": {
        "Strikeouts Over":  "KXMLBKS",
        "Hits Over":        "KXMLBHIT",
        "Total Bases Over": "KXMLBTB",
        "Home Runs Over":   "KXMLBHR",
        "HRR Over":         "KXMLBHRR",
    },
}

# Widest two-sided quote we will call a price, in cents. Same rationale as the
# game books: an untraded contract sits at 2c/98c and a naive midpoint reads as
# a confident 50%. Props are thinner than game markets, so this bites more
# often here — on the Oct 27 sample it rejected 25% of listed contracts, almost
# all of them steals and blocks.
MAX_SPREAD_CENTS = 10.0

# A rung priced at the extremes carries no information we can bet into: the
# model would need an enormous error to disagree usefully, and the quote is
# usually stale rather than tight. Keeps "LeBron 2+ points at 99c" off the card.
MIN_PROB, MAX_PROB = 0.05, 0.95


def _mid(market: Dict) -> Optional[float]:
    """Mid of the two-sided book, or None when it is not quoted both ways."""
    b, a = _f(market.get("yes_bid_dollars")), _f(market.get("yes_ask_dollars"))
    if b is None or a is None or not (0 < b < a < 1):
        return None
    if (a - b) * 100.0 > MAX_SPREAD_CENTS:
        return None
    return round((a + b) / 2.0, 4)


def _parse_subtitle(sub: str) -> Optional[tuple[str, float]]:
    """'Jayson Tatum: 25+' -> ('Jayson Tatum', 25.0).

    ⚠️ Parsed from `yes_sub_title`, never from the ticker's player segment.
    The ticker crushes the name ('BOSJTATUM0', 'ATLNALEXANDERWALKER7') and
    carries a jersey number, so reversing it into 'Jayson Tatum' to match ESPN
    is guesswork. The subtitle is already the display name.
    """
    s = str(sub or "").strip()
    m = re.match(r"^(.+?):\s*(\d+(?:\.\d+)?)\s*\+?$", s)
    if not m:
        return None
    name = m.group(1).strip()
    try:
        thresh = float(m.group(2))
    except ValueError:
        return None
    if not name or thresh <= 0:
        return None
    return name, thresh


def _mlb_game_key(code: str) -> str:
    """MLB prop tickers carry a TIME segment the NBA ones do not.

    NBA:  KXNBAPTS-26OCT27BKNBOS      -> 26OCT27BKNBOS
    MLB:  KXMLBHRR-26OCT272000MILTB   -> 26OCT27 + MILTB, dropping the 2000.

    Without this every MLB prop fails to join its game and props look simply
    absent — the silent-empty failure mode this repo keeps relearning.
    """
    m = re.match(r"^(\d{2}[A-Z]{3}\d{2})(\d{4})?(.*)$", code)
    if not m:
        return code
    return f"{m.group(1)}{m.group(3)}"


def fetch_props_for_games(sport: str, games: List[Dict[str, Any]],
                          game_date: str) -> int:
    """Stamp `game["player_props"]` for each game, from Kalshi. Returns count.

    Shape matches what the Odds API path produced, so `nba_player_props` and
    `mlb_player_props` need no changes:

        {player_name: {prop_label: {line, market_prob, odds_display,
                                    over_price, book}}}

    ⚠️ ONE RUNG PER (player, market) reaches the analyzer — the one closest to
    a coin flip. Every rung is a real tradable contract, but handing the
    analyzer all of them would put six Tatum points picks on one card and let a
    single player's projection error repeat six times. Nearest-50% is the rung
    where the model's disagreement with the market is most likely to be real
    signal rather than tail-modelling error, which is where our sigma
    assumptions are weakest.
    """
    sport = sport.upper()
    for g in games:
        g.setdefault("player_props", {})
    if not enabled():
        return 0
    series = PROP_SERIES.get(sport) or {}
    if not series or not games:
        return 0

    is_mlb = sport == "MLB"

    def _key(code: str) -> str:
        return _mlb_game_key(code) if is_mlb else code

    by_game: Dict[str, Dict[str, Any]] = {_key(str(g.get("game_id") or "")): g
                                          for g in games}
    # {game_key: {player: {label: (dist_from_half, payload)}}}
    best: Dict[str, Dict[str, Dict[str, tuple]]] = {}
    listed = quoted = 0

    for label, ticker in series.items():
        try:
            markets = fetch_open_markets(ticker, max_pages=12)
        except Exception as e:
            logger.warning(f"{sport} props: {ticker} fetch failed: {e}")
            continue
        for m in markets:
            if event_date(m) != game_date:
                continue
            gk = _key(_game_code(m))
            if gk not in by_game:
                continue
            listed += 1
            parsed = _parse_subtitle(m.get("yes_sub_title"))
            if not parsed:
                continue
            player, thresh = parsed
            prob = _mid(m)
            if prob is None or not (MIN_PROB <= prob <= MAX_PROB):
                continue
            quoted += 1
            dist = abs(prob - 0.5)
            slot = best.setdefault(gk, {}).setdefault(player, {})
            if label in slot and slot[label][0] <= dist:
                continue
            slot[label] = (dist, {
                # rung N is P(stat >= N); the analyzer wants P(stat > line)
                "line":         thresh - 0.5,
                "market_prob":  prob,
                "odds_display": f"{int(round(prob * 100))}c",
                "over_price":   prob,
                "book":         "kalshi",
                "kalshi_ticker": m.get("ticker"),
                "kalshi_rung":   thresh,
            })

    total = 0
    for gk, players in best.items():
        g = by_game[gk]
        props: Dict[str, Dict[str, Any]] = {}
        for player, labels in players.items():
            props[player] = {lab: payload for lab, (_d, payload) in labels.items()}
            total += len(labels)
        g["player_props"] = props

    if listed:
        logger.info(f"{sport} props: {listed} contracts on today's games, "
                    f"{quoted} priced → {total} rungs across "
                    f"{sum(len(p) for p in best.values())} players")
    elif games:
        logger.info(f"{sport} props: Kalshi listed none for today's games")
    return total
