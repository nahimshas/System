"""
MLB sport module — implements the Sport protocol.

MLB has more per-game data enrichment than other sports:
  • Schedule data (pitcher IDs, venue, team IDs) is stamped onto Odds API
    game dicts inside fetch_games().
  • Umpire assignment and weather are resolved per-game inside analyze_games().
  • Pitcher stats are fetched inside analyze_games() and collected in
    context["pitcher_stats_map"] so the props analyzer can use them afterward
    without a second network round-trip.

This is a thin adapter — no analysis logic lives here.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sports.base import Sport, SportCapabilities  # noqa: F401
from src.sports.registry import REGISTRY

logger = logging.getLogger(__name__)

_ENTRY = REGISTRY["mlb"]


class MLBModule:
    """Sport adapter for the MLB."""

    key:   str               = _ENTRY.key
    label: str               = _ENTRY.label
    caps:  SportCapabilities = _ENTRY.caps

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """
        Fetch MLB odds + player prop lines, then enrich each game dict with
        schedule data (pitcher IDs, venue, team IDs, game_pk).

        Returns enriched game dicts ready for analyze_games().
        """
        from datetime import date as _date
        from src.data.odds_client import get_game_odds, fetch_player_props, get_last_api_error
        from src.data.mlb_stats import get_todays_games
        from src.config import MLB_SPORT

        today_date = _date.fromisoformat(today)   # get_todays_games needs a date object
        games = get_game_odds(MLB_SPORT)

        for g in games:
            try:
                g["player_props"] = fetch_player_props(g["game_id"], MLB_SPORT)
            except Exception as e:
                logger.warning(f"MLB player props fetch failed ({g.get('home_team')}): {e}")
                g["player_props"] = {}

        if not games:
            err = get_last_api_error()
            if err:
                logger.error(f"MLB odds unavailable: {err}")
            else:
                logger.info(f"No MLB odds available today")

        # Enrich odds games with schedule data (pitcher IDs, venue, team IDs)
        schedule = get_todays_games(today_date)
        schedule_map = {g["home_team"]: g for g in schedule}
        schedule_map.update({g["away_team"]: g for g in schedule})

        for game in games:
            sched = schedule_map.get(game["home_team"]) or schedule_map.get(game["away_team"])
            if sched:
                game.update({
                    "home_pitcher_id":   sched.get("home_pitcher_id"),
                    "home_pitcher_name": sched.get("home_pitcher_name", "TBD"),
                    "away_pitcher_id":   sched.get("away_pitcher_id"),
                    "away_pitcher_name": sched.get("away_pitcher_name", "TBD"),
                    "venue":             sched.get("venue", ""),
                    "home_team_id":      sched.get("home_team_id"),
                    "away_team_id":      sched.get("away_team_id"),
                    "game_pk":           sched.get("game_pk"),
                })
                # Back-stamp commence_time onto schedule game (used by props lock badge)
                sched["commence_time"] = game.get("commence_time", "")
                sched["player_props"]  = game.get("player_props", {})

        # Stash schedule on the module for fetch_context() to pick up
        self._last_schedule = schedule
        return games

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Fetch MLB injuries and umpire assignments.

        Also carries the schedule (set by fetch_games) so props_analyzer can
        use it after analyze_games().  Initialises pitcher_stats_map as an
        empty dict; analyze_games() populates it as a side-effect.
        """
        from datetime import date as _date
        from src.data.injuries import get_mlb_injuries
        from src.data.umpire import get_home_plate_umpires, get_umpire_tendency

        today_date = _date.fromisoformat(today)   # get_home_plate_umpires needs a date object

        ctx: dict[str, Any] = {
            "injuries":          {},
            "umpire_map":        {},
            "schedule":          getattr(self, "_last_schedule", []),
            "pitcher_stats_map": {},   # populated by analyze_games()
            "today_date":        today_date,  # date object needed by get_team_schedule_load
            "team_records":      {},   # {team_id: {wins, losses}} — populated below
        }

        try:
            from src.data.mlb_stats import get_mlb_team_records
            ctx["team_records"] = get_mlb_team_records()
        except Exception as e:
            logger.warning(f"MLB standings unavailable: {e}")

        try:
            ctx["injuries"] = get_mlb_injuries()
        except Exception as e:
            logger.warning(f"MLB injuries unavailable: {e}")

        try:
            ctx["umpire_map"] = get_home_plate_umpires(today_date)
        except Exception as e:
            logger.warning(f"MLB umpire fetch failed: {e}")

        # Stamp umpire k_factor onto every odds game dict now so edge_finder has it
        umpire_map    = ctx["umpire_map"]
        get_tendency  = get_umpire_tendency
        for game in games:
            game_pk              = game.get("game_pk")
            ump_name             = umpire_map.get(game_pk, "")
            ump_tendency         = get_tendency(ump_name)
            game["umpire_name"]  = ump_name
            game["umpire_k_factor"] = ump_tendency.get("k_factor", 1.0)

        # Stamp umpire data onto schedule games too (used by props_analyzer)
        for sg in ctx["schedule"]:
            gpk = sg.get("game_pk")
            ump = umpire_map.get(gpk, "")
            sg["umpire_name"]     = ump
            sg["umpire_k_factor"] = get_tendency(ump).get("k_factor", 1.0)

        ctx["get_umpire_tendency"] = get_tendency   # pass through for analyze_games
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
        """
        Run the MLB edge-finding model over every game.

        Fetches pitcher stats, batting, bullpen, and weather per-game.
        Populates context["pitcher_stats_map"] as a side-effect so the
        props analyzer can use pitcher stats without a second API round-trip.
        """
        from src.data.mlb_stats import (
            get_pitcher_stats, get_pitcher_recent_stats,
            get_team_batting_stats, get_bullpen_stats, get_team_schedule_load,
            get_pitcher_hand, get_team_splits_vs_hand,
        )
        from src.data.umpire import get_umpire_tendency
        from src.data.weather import get_game_weather
        from src.models.edge_finder import analyze_mlb_game

        from datetime import date as _date
        injuries          = context.get("injuries", {})
        pitcher_stats_map = context.setdefault("pitcher_stats_map", {})
        get_tendency      = context.get("get_umpire_tendency", get_umpire_tendency)
        today_date        = context.get("today_date", _date.today())
        team_records      = context.get("team_records", {})

        results: list[Any] = []

        for game in games:
            home = game["home_team"]
            away = game["away_team"]

            # Weather (cached by city — no duplicate API calls for same park)
            try:
                wx = get_game_weather(game.get("venue", ""), game.get("commence_time", ""))
            except Exception as e:
                logger.debug(f"Weather fetch failed ({game.get('venue', '')}): {e}")
                wx = {}

            try:
                hp_stats = get_pitcher_stats(game.get("home_pitcher_id"))
                ap_stats = get_pitcher_stats(game.get("away_pitcher_id"))

                hp_recent = get_pitcher_recent_stats(game.get("home_pitcher_id"))
                ap_recent = get_pitcher_recent_stats(game.get("away_pitcher_id"))
                if hp_recent:
                    hp_stats = {**hp_stats, **hp_recent}
                if ap_recent:
                    ap_stats = {**ap_stats, **ap_recent}

                home_bat  = get_team_batting_stats(game.get("home_team_id"))
                away_bat  = get_team_batting_stats(game.get("away_team_id"))
                home_bp   = get_bullpen_stats(game.get("home_team_id"))
                away_bp   = get_bullpen_stats(game.get("away_team_id"))

                # Platoon splits: each lineup's OPS vs the OPPOSING starter's hand.
                # home_off_split = home batters vs the away starter's throwing hand.
                home_hand = get_pitcher_hand(game.get("home_pitcher_id"))
                away_hand = get_pitcher_hand(game.get("away_pitcher_id"))
                home_splits = get_team_splits_vs_hand(game.get("home_team_id"))
                away_splits = get_team_splits_vs_hand(game.get("away_team_id"))
                home_off_split = (
                    {**home_splits[away_hand], "vs_hand": away_hand}
                    if away_hand and away_hand in home_splits else None
                )
                away_off_split = (
                    {**away_splits[home_hand], "vs_hand": home_hand}
                    if home_hand and home_hand in away_splits else None
                )
                home_load = get_team_schedule_load(game.get("home_team_id"), today_date)
                away_load = get_team_schedule_load(game.get("away_team_id"), today_date)

                # Collect pitcher stats for props_analyzer
                if game.get("home_pitcher_name"):
                    pitcher_stats_map[game["home_pitcher_name"]] = hp_stats
                if game.get("away_pitcher_name"):
                    pitcher_stats_map[game["away_pitcher_name"]] = ap_stats

                ump_tendency = get_tendency(game.get("umpire_name", ""))

                home_record = team_records.get(game.get("home_team_id")) or {}
                away_record = team_records.get(game.get("away_team_id")) or {}

                recs = analyze_mlb_game(
                    game, hp_stats, ap_stats, home_bat, away_bat,
                    home_bp, away_bp, injuries,
                    home_schedule_load=home_load,
                    away_schedule_load=away_load,
                    umpire_tendency=ump_tendency,
                    weather=wx,
                    home_season_stats=home_record,
                    away_season_stats=away_record,
                    home_off_split=home_off_split,
                    away_off_split=away_off_split,
                    min_edge=min_edge,
                )
                results.extend(recs)

            except Exception as e:
                logger.error(f"MLB game analysis error ({home} vs {away}): {e}")

        # Propagate model totals to schedule games (used by props_analyzer)
        odds_by_matchup = {(g["home_team"], g["away_team"]): g for g in games}
        for sg in context.get("schedule", []):
            og = odds_by_matchup.get((sg.get("home_team"), sg.get("away_team")))
            if og:
                sg["model_total"]  = og.get("model_total")
                sg["market_total"] = og.get("market_total")

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
        Return MLB player-prop picks.

        Uses context["schedule"] and context["pitcher_stats_map"] which are
        populated by fetch_games() and analyze_games() respectively.
        Call analyze_games() before fetch_props() to ensure pitcher stats
        are available.
        """
        from src.models.props_analyzer import mlb_player_props
        schedule          = context.get("schedule", [])
        pitcher_stats_map = context.get("pitcher_stats_map", {})
        return mlb_player_props(schedule, pitcher_stats_map, min_edge=min_edge)

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Settle yesterday's MLB picks.

        See NBAModule.settle() — centralised settler handles all budget
        sports together.  Phase 3 will break this apart per-sport.
        """
        return 0


# Module-level singleton
mlb = MLBModule()
