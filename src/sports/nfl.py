"""
NFL sport module — implements the Sport protocol.

Thin adapter that wires together the existing data-fetching and analysis
functions from src/data/ and src/models/ behind the unified Sport interface.
No analysis logic lives here; all math stays in its original module.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["nfl"]


class NFLModule:
    """Sport adapter for the NFL."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch NFL odds from the Odds API."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import NFL_SPORT

        games = get_game_odds(NFL_SPORT)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"NFL odds unavailable: {err}")
            else:
                logger.info("No NFL games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch season stats, recent form, rest days, and injuries."""
        from datetime import date as _date
        from src.data.nfl_stats import get_nfl_context
        from src.data.injuries import get_nfl_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_nfl_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"NFL stats fetch failed: {e}")
            ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}

        try:
            ctx["injuries"] = get_nfl_injuries()
        except Exception as e:
            logger.warning(f"NFL injuries unavailable: {e}")
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
        """Run the NFL edge-finding model over every game."""
        from src.models.edge_finder import analyze_nfl_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_nfl_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"NFL game analysis error ({game.get('home_team')}): {e}")
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
        """
        No NFL props yet.  To enable: set has_props=True in the registry
        and implement the body here (follow the NBA/MLB pattern).
        """
        return []

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle yesterday's NFL picks.

        See NBAModule.settle() — centralised settler handles all budget
        sports together.  Phase 3 will break this apart per-sport.
        """
        return 0


# Module-level singleton
nfl = NFLModule()
