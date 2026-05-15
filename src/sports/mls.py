"""
MLS sport module — implements the Sport protocol.

MLS is watchlist-only with its own display tile (in_main_display_pool=False).
Picks are written to state/watchlist_history.json.
Write-once morning backup (morning_mls_display) protects picks from erasure
on re-runs once games have started.

Active February through November.

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["mls"]


class MLSModule:
    """Sport adapter for MLS."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch MLS odds from the Odds API."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import MLS_SPORT

        games = get_game_odds(MLS_SPORT)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"MLS odds unavailable: {err}")
            else:
                logger.info("No MLS games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch MLS season stats, recent form, rest days, and injuries."""
        from datetime import date as _date
        from src.data.mls_stats import get_mls_context, get_mls_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_mls_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"MLS stats fetch failed: {e}")
            ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}, "injuries": {}}

        try:
            ctx["injuries"] = get_mls_injuries()
        except Exception as e:
            logger.warning(f"MLS injuries unavailable: {e}")
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
        """Run the MLS edge-finding model over every game."""
        from src.models.edge_finder import analyze_mls_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_mls_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"MLS game analysis error ({game.get('home_team')}): {e}")
        return results

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        MLS picks are settled manually into watchlist_history.json.

        The date-based watchlist settler (check_and_settle_watchlist) handles
        any MLS entries that have ESPN game IDs.  For now returns 0 —
        Phase 3 will wire up a dedicated MLS settler if needed.
        """
        return 0


# Module-level singleton
mls = MLSModule()
