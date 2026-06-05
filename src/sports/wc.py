"""
FIFA World Cup 2026 sport module — implements the Sport protocol.

World Cup is watchlist-only with its own display tile (in_main_display_pool=False).
Picks are written to state/watchlist_history.json. Write-once morning backup
(morning_wc_display) protects picks from erasure on re-runs once games start.

Active June–July 2026. Driven by an Elo strength model (see wc_stats.py /
edge_finder.analyze_wc_game) rather than club xG, since international teams have
little usable group-stage form.

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["wc"]


class WorldCupModule:
    """Sport adapter for the FIFA World Cup."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """Fetch World Cup odds from the Odds API (3-way soccer markets)."""
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import WC_SPORT

        games = get_game_odds(WC_SPORT)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"World Cup odds unavailable: {err}")
            else:
                logger.info("No World Cup games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """Fetch current Elo ratings (seed + learned + updated from results)."""
        from datetime import date as _date
        from src.data.wc_stats import get_wc_context, get_wc_injuries

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})

        try:
            ctx = get_wc_context(today_date, team_names=team_names)
        except Exception as e:
            logger.error(f"World Cup Elo context fetch failed: {e}")
            ctx = {"elo": {}, "league_avg": 1620.0}

        try:
            ctx["injuries"] = get_wc_injuries()
        except Exception as e:
            logger.warning(f"World Cup injuries unavailable: {e}")
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
        """Run the World Cup Elo edge-finding model over every game."""
        from src.models.edge_finder import analyze_wc_game

        results: list[Any] = []
        injuries = context.get("injuries", {})
        for game in games:
            try:
                recs = analyze_wc_game(game, context, injuries, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"World Cup game analysis error ({game.get('home_team')}): {e}")
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
        """No World Cup props currently. Returns [] until has_props=True is set."""
        return []

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        World Cup picks settle date-based via check_and_settle_watchlist()
        (ESPN soccer/fifa.world final scores), which runs unconditionally in
        main.py. Returns 0 here — no module-local settlement needed.
        """
        return 0


# Module-level singleton
wc = WorldCupModule()
