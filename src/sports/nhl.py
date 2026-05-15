"""
NHL sport module — implements the Sport protocol.

NHL is display-only in the main card section (in_main_display_pool=True)
but never enters budget allocation or parlays.  Results settle into
watchlist_history.json via check_and_settle_watchlist().

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["nhl"]


class NHLModule:
    """Sport adapter for the NHL."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch NHL odds from the Odds API."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import NHL_SPORT

        games = get_game_odds(NHL_SPORT)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"NHL odds unavailable: {err}")
            else:
                logger.info("No NHL games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch season stats, recent form, rest days, and injuries."""
        from datetime import date as _date
        from src.data.nhl_stats import get_nhl_context
        from src.data.injuries import get_nhl_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_nhl_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"NHL stats fetch failed: {e}")
            ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}

        try:
            ctx["injuries"] = get_nhl_injuries()
        except Exception as e:
            logger.warning(f"NHL injuries unavailable: {e}")
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
        """Run the NHL edge-finding model over every game."""
        from src.models.edge_finder import analyze_nhl_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_nhl_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"NHL game analysis error ({game.get('home_team')}): {e}")
        return results

    # ------------------------------------------------------------------
    # Props
    # ------------------------------------------------------------------

    def fetch_props(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """No NHL props currently. Returns [] until has_props=True is set."""
        return []

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle yesterday's NHL watchlist picks.

        Uses the date-based watchlist settler — NHL games finish overnight,
        well before the 9 am run, so results are deterministic by morning.
        """
        from src.data.outcome_checker import check_and_settle_watchlist
        try:
            settled = check_and_settle_watchlist(today)
            if settled:
                logger.info(f"NHL watchlist settlement: {settled} pick(s) closed")
            return settled
        except Exception as e:
            logger.warning(f"NHL watchlist settlement failed (non-fatal): {e}")
            return 0


# Module-level singleton
nhl = NHLModule()
