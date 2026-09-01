"""
College football (FBS) sport module — implements the Sport protocol.

WATCHLIST ONLY, with its own display tile (in_main_display_pool=False). Picks
go to state/watchlist_history.json; no money is ever allocated.

Why watchlist and not budget, beyond the usual probation: Kalshi's CFB books
are mostly untradeable. Measured Aug 29 2026 — moneyline median spread 8 cents
(MLB is 1), and the spread/total books show median open interest of ZERO with
16-18 cent spreads. Only marquee games have real depth. Even a working model
would be actionable on a handful of games a week, so this exists to answer
"does the model have any signal", not "can we bet it".

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["cfb"]


class CFBModule:
    """Sport adapter for college football."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch NCAAF odds from the Odds API (one bulk call for the slate)."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import CFB_SPORT

        games = get_game_odds(CFB_SPORT)
        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"CFB odds unavailable: {err}")
            else:
                logger.info("No CFB games today or odds unavailable")
        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Elo ratings — bootstraps from ESPN on first run, self-updates after."""
        from datetime import date as _date
        from src.data.cfb_stats import get_cfb_context
        try:
            return get_cfb_context(_date.fromisoformat(today))
        except Exception as e:
            logger.error(f"CFB context fetch failed: {e}")
            return {"elo": {}, "games": {}}

    def analyze_games(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """Run the Elo-margin model over every rated matchup."""
        from src.models.edge_finder import analyze_cfb_game

        results: list[Any] = []
        if not context.get("elo"):
            logger.warning("CFB: no Elo ratings available — skipping analysis")
            return results
        for game in games:
            try:
                results.extend(analyze_cfb_game(game, context, min_edge=min_edge))
            except Exception as e:
                logger.error(f"CFB game analysis error ({game.get('home_team')}): {e}")
        return results

    def fetch_props(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """No CFB props."""
        return []

    def settle(self, today: str) -> int:
        """Settled date-based via check_and_settle_watchlist()."""
        return 0


# Module-level singleton
cfb = CFBModule()
