"""
Liga MX sport module — implements the Sport protocol.

Liga MX (Mexican Primera División) is watchlist-only with its own display tile
(in_main_display_pool=False). Elo-driven model (no club xG for Mexico), like the
World Cup. Picks are written to state/watchlist_history.json.

Robinhood added Liga MX in July 2026. Active roughly year-round (Apertura
Jul–Dec, Clausura Jan–May).

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["ligamx"]


class LigaMXModule:
    """Sport adapter for Liga MX."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch Liga MX odds from the Odds API."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import LIGAMX_SPORT

        games = get_game_odds(LIGAMX_SPORT)
        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"Liga MX odds unavailable: {err}")
            else:
                logger.info("No Liga MX games today or odds unavailable")
        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch Liga MX Elo ratings (bootstraps on first run, self-updates after)."""
        from datetime import date as _date
        from src.data.ligamx_stats import get_ligamx_context, get_ligamx_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})
        try:
            ctx = get_ligamx_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"Liga MX context fetch failed: {e}")
            ctx = {"elo": {}, "last_played": {}}
        try:
            ctx["injuries"] = get_ligamx_injuries()
        except Exception:
            ctx["injuries"] = {}
        return ctx

    def analyze_games(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """Run the Liga MX Elo model over every game."""
        from src.models.edge_finder import analyze_ligamx_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_ligamx_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"Liga MX game analysis error ({game.get('home_team')}): {e}")
        return results

    def fetch_props(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """No Liga MX props."""
        return []

    def settle(self, today: str) -> int:
        """Settled date-based via check_and_settle_watchlist() (soccer 90' aware)."""
        return 0


# Module-level singleton
ligamx = LigaMXModule()
