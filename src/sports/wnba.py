"""
WNBA sport module — implements the Sport protocol.

WNBA is watchlist-only with its own display tile (in_main_display_pool=False).
Picks are written to state/watchlist_history.json and settled manually or
via the date-based watchlist settler (check_and_settle_watchlist).

Write-once morning backup (morning_wnba_display) protects picks from being
erased on re-runs once games have started and the Odds API drops them.

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["wnba"]


class WNBAModule:
    """Sport adapter for the WNBA."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch WNBA odds from the Odds API."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import WNBA_SPORT

        games = get_game_odds(WNBA_SPORT)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"WNBA odds unavailable: {err}")
            else:
                logger.info("No WNBA games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch WNBA season stats, recent form, rest days, and injuries."""
        from datetime import date as _date
        from src.data.wnba_stats import get_wnba_context, get_wnba_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_wnba_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"WNBA stats fetch failed: {e}")
            ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}

        try:
            ctx["injuries"] = get_wnba_injuries()
        except Exception as e:
            logger.warning(f"WNBA injuries unavailable: {e}")
            ctx["injuries"] = {}

        return ctx

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_games(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """Run the WNBA edge-finding model over every game."""
        from src.models.edge_finder import analyze_wnba_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_wnba_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"WNBA game analysis error ({game.get('home_team')}): {e}")
        return results

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle yesterday's WNBA watchlist picks.

        Uses the date-based watchlist settler.  Games that were in-progress
        during the previous run are carried forward via the morning_wnba_display
        backup and settled here once final.
        """
        from src.data.outcome_checker import check_and_settle_watchlist
        try:
            settled = check_and_settle_watchlist(today)
            if settled:
                logger.info(f"WNBA watchlist settlement: {settled} pick(s) closed")
            return settled
        except Exception as e:
            logger.warning(f"WNBA watchlist settlement failed (non-fatal): {e}")
            return 0


# Module-level singleton
wnba = WNBAModule()
