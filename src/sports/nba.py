"""
NBA sport module — implements the Sport protocol.

Thin adapter that wires together the existing data-fetching and analysis
functions from src/data/ and src/models/ behind the unified Sport interface.
No analysis logic lives here; all math stays in its original module.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401 (satisfies Protocol)
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["nba"]


class NBAModule:
    """Sport adapter for the NBA."""

    key:   str              = _ENTRY.key
    label: str              = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch NBA odds + player prop lines for each game."""
        from src.data.odds_client import get_game_odds, fetch_player_props, get_last_api_error
        from src.config import NBA_SPORT

        games = get_game_odds(NBA_SPORT)
        for g in games:
            try:
                g["player_props"] = fetch_player_props(g["game_id"], NBA_SPORT)
            except Exception as e:
                logger.warning(f"NBA player props fetch failed ({g.get('home_team')}): {e}")
                g["player_props"] = {}

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"NBA odds unavailable: {err}")
            else:
                logger.info("No NBA games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch season stats, recent form, rest days, and injuries."""
        from datetime import date as _date
        from src.data.nba_stats import get_nba_context
        from src.data.injuries import get_nba_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_nba_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"NBA stats fetch failed: {e}")
            ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}, "team_leaders": {}}

        try:
            ctx["injuries"] = get_nba_injuries()
        except Exception as e:
            logger.warning(f"NBA injuries unavailable: {e}")
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
        """Run the NBA edge-finding model over every game."""
        from src.models.edge_finder import analyze_nba_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_nba_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"NBA game analysis error ({game.get('home_team')}): {e}")
        return results

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle yesterday's NBA picks.

        Note: check_and_settle() handles all budget-sport picks together
        (NBA + MLB + NFL) since they share history.json.  In Phase 3 this
        will be broken into per-sport settlers; for now we return 0 here
        and let main.py call the centralised settler once.
        """
        return 0


# Module-level singleton — import and use directly.
nba = NBAModule()
