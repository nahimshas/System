"""
IPL sport module — implements the Sport protocol.

IPL is watchlist-only with its own display tile (in_main_display_pool=False).
It uses the rolling pending-file pattern: picks are staged in
state/watchlist_pending.json when discovered and settled asynchronously
once the game is final (games often start at ~7 am PST and finish at ~11 am,
straddling the 9 am run window).

A 36-hour lookahead is used so tomorrow's match is captured the evening before.

Thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["ipl"]


class IPLModule:
    """Sport adapter for the IPL."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """
        Fetch IPL odds with a 36-hour lookahead.

        The extended window ensures tomorrow's match (which may already have
        odds posted) is captured during the evening run before the game starts.
        """
        from src.data.odds_client import get_game_odds, get_last_api_error
        from src.config import IPL_SPORT

        games = get_game_odds(IPL_SPORT, hours_lookahead=self.caps.hours_lookahead)

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"IPL odds unavailable: {err}")
            else:
                logger.info("No IPL games today or odds unavailable")

        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Fetch IPL season form, rest days, venue stats, head-to-head records,
        and player unavailabilities.
        """
        from datetime import date as _date
        from src.data.ipl_stats import get_ipl_context

        today_date = _date.fromisoformat(today)
        team_names = list({t for g in games for t in [g["home_team"], g["away_team"]]})
        matchups   = [(g["home_team"], g["away_team"]) for g in games]

        try:
            ctx = get_ipl_context(today_date, team_names=team_names, matchups=matchups)
        except Exception as e:
            logger.error(f"IPL stats fetch failed: {e}")
            ctx = {
                "season_form": {}, "rest_days": {}, "venue_stats": {},
                "venue_config": {}, "match_venues": {}, "h2h": {},
                "match_flags": {}, "unavailabilities": {},
            }

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
        """Run the IPL edge-finding model over every game."""
        from src.models.edge_finder import analyze_ipl_game

        results: list[Any] = []
        for game in games:
            try:
                recs = analyze_ipl_game(game, context, min_edge=min_edge)
                results.extend(recs)
            except Exception as e:
                logger.error(f"IPL game analysis error ({game.get('home_team')}): {e}")
        return results

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle IPL picks from the rolling pending file.

        Picks whose games are now final (ESPN stateId == "3") are moved from
        watchlist_pending.json to watchlist_history.json.  Safe to call
        multiple times — already-settled picks are skipped.
        """
        from src.data.outcome_checker import settle_watchlist_pending

        now_utc = datetime.now(timezone.utc)
        try:
            settled = settle_watchlist_pending(now_utc)
            if settled:
                logger.info(f"IPL rolling pending settlement: {settled} pick(s) settled")
            return settled
        except Exception as e:
            logger.warning(f"IPL rolling pending settlement failed (non-fatal): {e}")
            return 0


# Module-level singleton
ipl = IPLModule()
