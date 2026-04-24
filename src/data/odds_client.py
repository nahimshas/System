"""Fetches game lines from The Odds API (free tier: 500 credits/month)."""
import requests
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from src.config import ODDS_API_BASE, ODDS_API_KEY, PREFERRED_BOOK, FALLBACK_BOOKS

logger = logging.getLogger(__name__)


def _get(path: str, params: Dict) -> Optional[Any]:
    params["apiKey"] = ODDS_API_KEY
    try:
        r = requests.get(f"{ODDS_API_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API credits remaining: {remaining}")
        return r.json()
    except requests.RequestException as e:
        logger.error(f"Odds API error: {e}")
        return None


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(p1: float, p2: float):
    total = p1 + p2
    return p1 / total, p2 / total


def _pick_book_odds(bookmakers: List[Dict], market_key: str) -> Optional[Dict]:
    """Returns the best available bookmaker's outcomes for a given market."""
    priority = [PREFERRED_BOOK] + FALLBACK_BOOKS
    book_map = {b["key"]: b for b in bookmakers}
    for book_key in priority:
        if book_key in book_map:
            for mkt in book_map[book_key].get("markets", []):
                if mkt["key"] == market_key:
                    return {"book": book_key, "outcomes": mkt["outcomes"]}
    return None


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
        ml = _pick_book_odds(bookmakers, "h2h")
        if ml:
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
                    entry["spread"] = {
                        "book": sp["book"],
                        "home_spread": o.get("point", 0),
                        "home_prob": american_to_prob(o["price"]),
                        "away_prob": 1 - american_to_prob(o["price"]),
                    }
                    break

        # --- Total ---
        tot = _pick_book_odds(bookmakers, "totals")
        if tot:
            for o in tot["outcomes"]:
                if o["name"] == "Over":
                    line = o.get("point", 0)
                    op = american_to_prob(o["price"])
                    entry["total"] = {
                        "book": tot["book"],
                        "line": line,
                        "over_prob": op,
                        "under_prob": 1 - op,
                    }
                    break

        games.append(entry)

    logger.info(f"Fetched {len(games)} {sport} games from odds API")
    return games
