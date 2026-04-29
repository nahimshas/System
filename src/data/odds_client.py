"""Fetches game lines from The Odds API."""
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from src.config import ODDS_API_BASE, ODDS_API_KEY, PREFERRED_BOOK, FALLBACK_BOOKS

logger = logging.getLogger(__name__)


_last_api_error: Optional[str] = None   # module-level; cleared each call


def _get(path: str, params: Dict) -> Optional[Any]:
    global _last_api_error
    _last_api_error = None
    params["apiKey"] = ODDS_API_KEY
    try:
        r = requests.get(f"{ODDS_API_BASE}{path}", params=params, timeout=15)
        remaining = r.headers.get("x-requests-remaining", "?")
        used      = r.headers.get("x-requests-used", "?")
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
        sp      = _pick_book_odds(bookmakers, "spreads")
        sp_cons = _consensus_probs(bookmakers, "spreads")
        if sp:
            for o in sp["outcomes"]:
                if o["name"] == home:
                    hp_sp = sp_cons.get(home) if sp_cons else None
                    entry["spread"] = {
                        "book": sp["book"],
                        "home_spread": o.get("point", 0),
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
