"""Fetches game lines from The Odds API."""
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from src.config import ODDS_API_BASE, ODDS_API_KEY, PREFERRED_BOOK, FALLBACK_BOOKS

logger = logging.getLogger(__name__)


def american_to_implied_prob(price: int) -> float:
    """Convert American odds to implied probability including vig."""
    if price >= 0:
        return 100.0 / (price + 100.0)
    else:
        return abs(price) / (abs(price) + 100.0)

PROP_MARKET_LABEL = {
    "player_points":         "Points Over",
    "player_rebounds":       "Rebounds Over",
    "player_assists":        "Assists Over",
    "player_steals":         "Steals Over",
    "player_blocks":         "Blocks Over",
    "player_threes":         "Threes Over",
    "pitcher_strikeouts":    "Strikeouts Over",
    "batter_hits":           "Hits Over",
    "batter_total_bases":    "Total Bases Over",
    "batter_home_runs":      "Home Runs Over",
    "batter_hits_runs_rbis": "HRR Over",
}

_NBA_PROP_MARKETS = [
    "player_points","player_rebounds","player_assists",
    "player_steals","player_blocks","player_threes",
]
_MLB_PROP_MARKETS = [
    "pitcher_strikeouts","batter_hits","batter_total_bases",
    "batter_home_runs","batter_hits_runs_rbis",
]


_last_api_error: Optional[str] = None   # module-level; cleared each call
_credits_used: Optional[int] = None
_credits_remaining: Optional[int] = None


def _get(path: str, params: Dict) -> Optional[Any]:
    global _last_api_error, _credits_used, _credits_remaining
    _last_api_error = None
    params["apiKey"] = ODDS_API_KEY
    try:
        r = requests.get(f"{ODDS_API_BASE}{path}", params=params, timeout=15)
        try:
            _credits_used      = int(r.headers.get("x-requests-used", 0))
            _credits_remaining = int(r.headers.get("x-requests-remaining", 0))
        except (ValueError, TypeError):
            pass
        remaining = _credits_remaining if _credits_remaining is not None else "?"
        used      = _credits_used      if _credits_used      is not None else "?"
        logger.info(f"Odds API credits — used: {used} | remaining: {remaining}")
        if r.status_code == 401:
            _last_api_error = f"Odds API 401 Unauthorized — check ODDS_API_KEY secret in GitHub"
            logger.error(_last_api_error)
            return None
        if r.status_code == 422:
            _last_api_error = f"Odds API 422 — quota exhausted or invalid params ({r.text[:120]})"
            logger.error(_last_api_error)
            return None
        if r.status_code == 429:
            _last_api_error = f"Odds API 429 — rate limited (too many requests per minute)"
            logger.error(_last_api_error)
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        _last_api_error = f"Odds API request failed: {e}"
        logger.error(_last_api_error)
        return None


def get_last_api_error() -> Optional[str]:
    """Returns the last Odds API error message, or None if the last call succeeded."""
    return _last_api_error


def get_api_credits() -> Dict:
    """Returns the most recently seen credit counters from the Odds API response headers."""
    return {"used": _credits_used, "remaining": _credits_remaining}


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(p1: float, p2: float):
    total = p1 + p2
    return p1 / total, p2 / total


def _pick_book_odds(bookmakers: List[Dict], market_key: str) -> Optional[Dict]:
    """Returns the preferred book's outcomes for a given market (used for line value only)."""
    priority = [PREFERRED_BOOK] + FALLBACK_BOOKS
    book_map = {b["key"]: b for b in bookmakers}
    for book_key in priority:
        if book_key in book_map:
            for mkt in book_map[book_key].get("markets", []):
                if mkt["key"] == market_key:
                    return {"book": book_key, "outcomes": mkt["outcomes"]}
    return None


def _consensus_probs(bookmakers: List[Dict], market_key: str) -> Optional[Dict]:
    """
    Returns consensus (averaged no-vig) probabilities across ALL available books.
    More accurate than any single book — reduces noise from outlier lines.
    Returns {name: avg_no_vig_prob, ..., "book_count": n}.
    """
    probs_by_name: Dict[str, List[float]] = {}
    book_count = 0

    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt["key"] != market_key:
                continue
            outcomes = mkt["outcomes"]
            if len(outcomes) < 2:
                continue

            if market_key in ("h2h", "spreads"):
                raw = {o["name"]: american_to_prob(o["price"]) for o in outcomes}
                names = list(raw.keys())
                # Remove vig on each book before averaging
                p0, p1 = remove_vig(raw[names[0]], raw[names[1]])
                probs_by_name.setdefault(names[0], []).append(p0)
                probs_by_name.setdefault(names[1], []).append(p1)
                book_count += 1

            elif market_key == "totals":
                for o in outcomes:
                    p = american_to_prob(o["price"])
                    probs_by_name.setdefault(o["name"], []).append(p)
                book_count += 1
            break  # one market entry per bookmaker

    if not probs_by_name or book_count == 0:
        return None
    avg = {n: sum(ps) / len(ps) for n, ps in probs_by_name.items()}
    avg["book_count"] = book_count
    return avg


def _consensus_probs_for_spread(
    bookmakers: List[Dict], home: str, home_point: float
) -> Optional[Dict]:
    """
    Spread-specific consensus that only averages books offering the SAME spread
    direction for the home team as the preferred book's line.

    Problem this solves: in MLB some books list the favourite at -1.5 while
    DraftKings (preferred) may list them at +1.5 (alternate/reverse line).
    Mixing -1.5 probability (~40%) with +1.5 probability (~60%) produces a
    nonsensical consensus that matches neither bet.  By filtering to books with
    a matching sign we always compare apples to apples.
    """
    probs_by_name: Dict[str, List[float]] = {}
    book_count = 0

    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt["key"] != "spreads":
                continue
            outcomes = mkt["outcomes"]
            if len(outcomes) < 2:
                continue

            # Only include this book if it offers the home team on the same
            # side of zero as the preferred book.
            home_o = next((o for o in outcomes if o["name"] == home), None)
            if home_o is None:
                break
            book_home_point = home_o.get("point", 0)
            if home_point > 0 and book_home_point <= 0:
                break  # this book has home as favourite; preferred has them as underdog
            if home_point < 0 and book_home_point >= 0:
                break  # this book has home as underdog; preferred has them as favourite

            raw = {o["name"]: american_to_prob(o["price"]) for o in outcomes}
            names = list(raw.keys())
            p0, p1 = remove_vig(raw[names[0]], raw[names[1]])
            probs_by_name.setdefault(names[0], []).append(p0)
            probs_by_name.setdefault(names[1], []).append(p1)
            book_count += 1
            break

    if not probs_by_name or book_count == 0:
        return None
    avg = {n: sum(ps) / len(ps) for n, ps in probs_by_name.items()}
    avg["book_count"] = book_count
    return avg


def _pacific_offset() -> int:
    month = datetime.now(timezone.utc).month
    return -7 if 3 <= month <= 10 else -8


def _today_pacific():
    now_utc = datetime.now(timezone.utc)
    return (now_utc + timedelta(hours=_pacific_offset())).date()


def _pacific_today_utc_window():
    """Returns UTC timestamps covering today in Pacific time."""
    offset = _pacific_offset()
    now_utc = datetime.now(timezone.utc)
    now_pacific = now_utc + timedelta(hours=offset)
    pac_midnight = now_pacific.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = pac_midnight - timedelta(hours=offset)
    end_utc = start_utc + timedelta(days=1)
    return start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def get_game_odds(sport: str) -> List[Dict]:
    """Returns list of TODAY's games with parsed moneyline, spread, and total odds."""
    commence_from, commence_to = _pacific_today_utc_window()
    data = _get(f"/sports/{sport}/odds", {
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american",
        "commenceTimeFrom": commence_from,
        "commenceTimeTo": commence_to,
    })
    if not data:
        return []

    now_utc = datetime.now(timezone.utc)
    today_pacific = _today_pacific()
    games = []
    for game in data:
        commence_str = game.get("commence_time", "")
        try:
            commence_dt = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
            # Skip games already started
            if commence_dt <= now_utc:
                logger.info(f"Skipping started: {game.get('home_team')} vs {game.get('away_team')}")
                continue
            # Skip games not on today's Pacific date
            game_pacific_date = (commence_dt + timedelta(hours=_pacific_offset())).date()
            if game_pacific_date != today_pacific:
                logger.info(f"Skipping non-today game ({game_pacific_date}): {game.get('home_team')} vs {game.get('away_team')}")
                continue
        except (ValueError, AttributeError):
            pass

        home = game["home_team"]
        away = game["away_team"]
        bookmakers = game.get("bookmakers", [])

        entry = {
            "game_id": game["id"],
            "sport": sport,
            "home_team": home,
            "away_team": away,
            "commence_time": game["commence_time"],
            "moneyline": None,
            "spread": None,
            "total": None,
        }

        # --- Moneyline ---
        ml      = _pick_book_odds(bookmakers, "h2h")    # for reference odds
        ml_cons = _consensus_probs(bookmakers, "h2h")   # for probability
        if ml and ml_cons and home in ml_cons and away in ml_cons:
            hp = ml_cons[home]
            ap = ml_cons[away]
            entry["moneyline"] = {
                "book": f"consensus({ml_cons.get('book_count', 1)})",
                "home_prob": hp,
                "away_prob": ap,
                "home_odds": next((o["price"] for o in ml["outcomes"] if o["name"] == home), None),
                "away_odds": next((o["price"] for o in ml["outcomes"] if o["name"] == away), None),
            }
        elif ml:
            probs = {o["name"]: american_to_prob(o["price"]) for o in ml["outcomes"]}
            if home in probs and away in probs:
                hp, ap = remove_vig(probs[home], probs[away])
                entry["moneyline"] = {
                    "book": ml["book"],
                    "home_prob": hp,
                    "away_prob": ap,
                    "home_odds": next(o["price"] for o in ml["outcomes"] if o["name"] == home),
                    "away_odds": next(o["price"] for o in ml["outcomes"] if o["name"] == away),
                }

        # --- Spread ---
        sp = _pick_book_odds(bookmakers, "spreads")
        if sp:
            for o in sp["outcomes"]:
                if o["name"] == home:
                    home_point = o.get("point", 0)
                    # Build consensus ONLY from books that offer the home team on
                    # the same side of the line (same sign) as the preferred book.
                    # This prevents mixing -1.5 (~40%) and +1.5 (~60%) probabilities
                    # when books disagree on which team is the run-line favourite.
                    sp_cons = _consensus_probs_for_spread(bookmakers, home, home_point)
                    hp_sp = sp_cons.get(home) if sp_cons else None
                    entry["spread"] = {
                        "book": sp["book"],
                        "home_spread": home_point,
                        "home_prob": hp_sp if hp_sp else american_to_prob(o["price"]),
                        "away_prob": 1 - (hp_sp if hp_sp else american_to_prob(o["price"])),
                    }
                    break

        # --- Total ---
        tot      = _pick_book_odds(bookmakers, "totals")   # preferred book for line value
        tot_cons = _consensus_probs(bookmakers, "totals")  # consensus for probability
        if tot:
            for o in tot["outcomes"]:
                if o["name"] == "Over":
                    line = o.get("point", 0)
                    over_p  = tot_cons.get("Over")  if tot_cons else None
                    under_p = tot_cons.get("Under") if tot_cons else None
                    entry["total"] = {
                        "book": f"consensus({tot_cons.get('book_count',1)})" if tot_cons else tot["book"],
                        "line": line,
                        "over_prob":  over_p  if over_p  else american_to_prob(o["price"]),
                        "under_prob": under_p if under_p else 1 - american_to_prob(o["price"]),
                    }
                    break

        games.append(entry)

    logger.info(f"Fetched {len(games)} {sport} games from odds API")
    return games


def fetch_player_props(game_id: str, sport: str) -> Dict[str, Dict]:
    """
    Fetch player prop lines from Odds API for a specific game event.
    Returns {player_name: {prop_label: {line, over_price, odds_display, market_prob, book}}}

    - Uses preferred book (DraftKings) line as the reference line.
    - market_prob = consensus no-vig Over probability across books with matching line.
    - odds_display = with-vig implied % from preferred book (e.g. "-110" → "52.4%").
    """
    markets = ",".join(_NBA_PROP_MARKETS if "basketball" in sport else _MLB_PROP_MARKETS)
    data = _get(f"/sports/{sport}/events/{game_id}/odds", {
        "regions": "us", "markets": markets, "oddsFormat": "american",
    })
    if not data:
        return {}

    bookmakers = data.get("bookmakers", [])

    # ── Diagnostic: log raw structure of first bookmaker on first call ──────
    if bookmakers:
        bk0 = bookmakers[0]
        mkts0 = bk0.get("markets", [])
        logger.info(
            f"[props-diag] game={game_id[:8]} sport={sport} "
            f"bookmakers={len(bookmakers)} first_book={bk0.get('key')} "
            f"markets={[m.get('key') for m in mkts0]}"
        )
        for m in mkts0[:2]:   # log first two markets' first outcome
            outs = m.get("outcomes", [])
            if outs:
                o0 = outs[0]
                logger.info(
                    f"[props-diag] market={m.get('key')} first_outcome: "
                    f"name={o0.get('name')!r} description={o0.get('description')!r} "
                    f"point={o0.get('point')} price={o0.get('price')}"
                )
    else:
        logger.warning(
            f"[props-diag] game={game_id[:8]} sport={sport} → "
            f"response OK but NO bookmakers returned"
        )
    # ────────────────────────────────────────────────────────────────────────

    book_map   = {b["key"]: b for b in bookmakers}
    priority   = [PREFERRED_BOOK] + FALLBACK_BOOKS
    result: Dict[str, Dict] = {}
    all_market_keys = _NBA_PROP_MARKETS if "basketball" in sport else _MLB_PROP_MARKETS

    for market_key in all_market_keys:
        prop_label = PROP_MARKET_LABEL.get(market_key, market_key)

        # Step 1: find preferred book's Over line per player.
        # Odds API v4 player props may use either of two outcome formats:
        #   Format A: name=PlayerName, description="Over"/"Under"  (newer)
        #   Format B: name="Over"/"Under", description=PlayerName  (older)
        # We detect which format is in use from the first outcome we inspect.
        pref_lines:  Dict[str, float] = {}
        pref_prices: Dict[str, int]   = {}
        pref_book = None
        for bk in priority:
            if bk not in book_map:
                continue
            for mkt in book_map[bk].get("markets", []):
                if mkt.get("key") != market_key:
                    continue
                pref_book = bk
                outcomes = mkt.get("outcomes", [])
                if not outcomes:
                    break

                # Detect format from first outcome
                first = outcomes[0]
                desc_lower = first.get("description", "").lower()
                name_lower = first.get("name", "").lower()
                # Format A: description is "over" or "under"
                if desc_lower in ("over", "under"):
                    for o in outcomes:
                        if o.get("description", "").lower() == "over":
                            name  = o.get("name", "")
                            line  = o.get("point")
                            price = o.get("price")
                            if name and line is not None and price is not None:
                                pref_lines[name]  = float(line)
                                pref_prices[name] = int(price)
                # Format B: name is "over" or "under", description is player name
                elif name_lower in ("over", "under"):
                    for o in outcomes:
                        if o.get("name", "").lower() == "over":
                            name  = o.get("description", "")
                            line  = o.get("point")
                            price = o.get("price")
                            if name and line is not None and price is not None:
                                pref_lines[name]  = float(line)
                                pref_prices[name] = int(price)
                else:
                    # Unknown format — log and skip
                    logger.debug(
                        f"Unknown prop outcome format for {market_key}: "
                        f"name='{first.get('name')}' description='{first.get('description')}'"
                    )
                break
            if pref_book:
                break

        if not pref_lines:
            continue

        # Step 2: consensus no-vig probability across books with same line.
        # Must handle both formats A and B (detected per-book below).
        for player_name, pref_line in pref_lines.items():
            over_probs = []
            for book in bookmakers:
                for mkt in book.get("markets", []):
                    if mkt.get("key") != market_key:
                        continue
                    outcomes = mkt.get("outcomes", [])
                    if not outcomes:
                        break
                    # Detect format for this book
                    f = outcomes[0]
                    f_name = f.get("name", "").lower()
                    f_desc = f.get("description", "").lower()
                    if f_desc in ("over", "under"):
                        # Format A
                        over_o  = next((o for o in outcomes
                                        if o.get("name") == player_name
                                        and o.get("description", "").lower() == "over"
                                        and o.get("point") == pref_line), None)
                        under_o = next((o for o in outcomes
                                        if o.get("name") == player_name
                                        and o.get("description", "").lower() == "under"
                                        and o.get("point") == pref_line), None)
                    elif f_name in ("over", "under"):
                        # Format B
                        over_o  = next((o for o in outcomes
                                        if o.get("name", "").lower() == "over"
                                        and o.get("description") == player_name
                                        and o.get("point") == pref_line), None)
                        under_o = next((o for o in outcomes
                                        if o.get("name", "").lower() == "under"
                                        and o.get("description") == player_name
                                        and o.get("point") == pref_line), None)
                    else:
                        break
                    if over_o and under_o:
                        p_o = american_to_implied_prob(int(over_o["price"]))
                        p_u = american_to_implied_prob(int(under_o["price"]))
                        total = p_o + p_u
                        if total > 0:
                            over_probs.append(p_o / total)
                    break

            if not over_probs:
                continue

            market_prob  = sum(over_probs) / len(over_probs)
            over_price   = pref_prices.get(player_name, -110)
            odds_display = f"{american_to_implied_prob(over_price) * 100:.1f}%"

            result.setdefault(player_name, {})[prop_label] = {
                "line":         pref_line,
                "over_price":   over_price,
                "odds_display": odds_display,
                "market_prob":  round(market_prob, 4),
                "book":         pref_book or PREFERRED_BOOK,
            }

    if result:
        logger.info(f"Player props for {game_id}: {len(result)} players, "
                    f"{sum(len(v) for v in result.values())} lines")
    else:
        logger.warning(
            f"Player props EMPTY for {game_id} (sport={sport}) — "
            f"bookmakers={len(bookmakers)}, "
            f"credits_remaining={_credits_remaining}"
        )
    return result
