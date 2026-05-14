"""
Entry point. Can be run directly or via GitHub Actions.

Usage:
  python -m src.main                  # full run (NBA + MLB)
  python -m src.main --league nba     # NBA only
  python -m src.main --league mlb     # MLB only
  python -m src.main --no-email       # skip email delivery
"""
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.config import (
    ODDS_API_KEY, REPORT_DIR, REPORT_FILE,
    NBA_SPORT, MLB_SPORT, NFL_SPORT, NHL_SPORT, IPL_SPORT, WNBA_SPORT,
    MAX_SINGLE_BETS, SPORT_ACTIVE_MONTHS, MIN_EDGE,
)
from src.data.odds_client import get_game_odds, get_last_api_error, get_api_credits, fetch_player_props
from src.data.nba_stats import get_nba_context
from src.data.mlb_stats import (
    get_todays_games, get_pitcher_stats, get_pitcher_recent_stats,
    get_team_batting_stats, get_bullpen_stats, get_team_schedule_load,
)
from src.data.nfl_stats import get_nfl_context
from src.data.nhl_stats import get_nhl_context
from src.data.ipl_stats import get_ipl_context
from src.data.wnba_stats import get_wnba_context, get_wnba_injuries
from src.data.injuries import get_nba_injuries, get_mlb_injuries, get_nfl_injuries, get_nhl_injuries
from src.data.umpire import get_home_plate_umpires, get_umpire_tendency
from src.data.weather import get_game_weather
from src.models.edge_finder import analyze_nba_game, analyze_mlb_game, analyze_nfl_game, analyze_nhl_game, analyze_ipl_game, analyze_wnba_game
from src.models.parlay_builder import build_parlays
from src.models.props_analyzer import nba_player_props, mlb_player_props
from src.state.manager import (
    load_state, save_state, merge_picks,
    bet_to_dict, parlay_to_dict, prop_to_dict,
    _game_started,
)
from src.report.card_context import build_card_context, build_prop_context
from src.data.outcome_checker import (
    check_and_settle, check_and_settle_props,
    check_and_settle_watchlist,      # NHL date-based settlement
    settle_watchlist_pending,        # IPL (+ future leagues) rolling pending settlement
    load_watchlist_pending, save_watchlist_pending,
    load_watchlist_today_settled,
)
from src.report.generator import build_report
from src.report.email_sender import send_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Narrative/context hydration helpers
# Recompute display fields for pick dicts saved before the card context feature
# was deployed (those dicts have signals/research but no narrative/context key).
# ---------------------------------------------------------------------------

def _hydrate_bet(d: dict) -> dict:
    if "narrative" in d:
        return d
    narrative, context = build_card_context(
        d.get("sport", ""), d.get("pick", ""), d.get("bet_type", ""),
        d.get("signals", []), d.get("research", []),
        d.get("model_prob_pct", 50) / 100,
        d.get("market_prob_pct", 50) / 100,
        d.get("edge_pct", 0) / 100,
    )
    return {**d, "narrative": narrative, "context": context}


def _hydrate_prop(d: dict) -> dict:
    if "narrative" in d:
        return d
    narrative, context = build_prop_context(
        d.get("sport", ""), d.get("prop_type", ""), d.get("player", ""),
        d.get("team", ""), d.get("opponent", ""),
        d.get("signals", []), d.get("research", []),
        d.get("model_line", d.get("market_line", 0)),
        d.get("market_line", 0),
        d.get("edge_pct", 0) / 100,
    )
    return {**d, "narrative": narrative, "context": context}


def run(leagues: list[str], send_email: bool = True, reevaluate: bool = False,
        code_only: bool = False) -> int:
    from src.data.odds_client import _today_pacific
    today = _today_pacific()
    errors: list[str] = []

    # ------------------------------------------------------------------ #
    #  Code-only mode: re-render from saved state, zero API calls
    # ------------------------------------------------------------------ #
    if code_only:
        state = load_state(today)
        if state is not None:
            logger.info("Code-only mode — re-rendering report from saved state (no API calls)")
            final_singles   = state.get("singles", [])
            final_parlays   = state.get("parlays", [])
            final_props     = state.get("props",   [])
            change_warnings = state.get("warnings", [])
            saved_credits   = state.get("odds_api_credits", {})
            final_singles_display = state.get("singles_display") or final_singles
            final_props_display   = state.get("props_display")   or final_props
            final_ipl_display     = state.get("ipl_display", [])
            final_wnba_display    = state.get("wnba_display", [])

            # Hydrate narrative/context for picks saved before the card context feature
            final_singles_display = [_hydrate_bet(d) for d in final_singles_display]
            final_singles         = [_hydrate_bet(d) for d in final_singles]
            final_props_display   = [_hydrate_prop(d) for d in final_props_display]
            final_props           = [_hydrate_prop(d) for d in final_props]

            report_data = build_report(
                run_date=today,
                singles=final_singles,
                singles_display=final_singles_display,
                parlays=final_parlays,
                props=final_props,
                props_display=final_props_display,
                nba_game_count=state.get("nba_game_count", 0),
                mlb_game_count=state.get("mlb_game_count", 0),
                nfl_game_count=state.get("nfl_game_count", 0),
                nhl_game_count=state.get("nhl_game_count", 0),
                ipl_game_count=state.get("ipl_game_count", 0),
                ipl_display=final_ipl_display,
                wnba_game_count=state.get("wnba_game_count", 0),
                wnba_display=final_wnba_display,
                errors=errors,
                change_warnings=change_warnings,
                odds_api_credits=saved_credits,
            )
            template_dir = Path(__file__).parent / "report" / "templates"
            jinja_env    = Environment(loader=FileSystemLoader(str(template_dir)))
            template     = jinja_env.get_template("report.html")
            html         = template.render(report=report_data)

            out_dir  = Path(REPORT_DIR)
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / REPORT_FILE
            out_path.write_text(html, encoding="utf-8")
            logger.info(f"Report written to {out_path} (code-only)")

            bet_count = len(final_singles) + len(final_parlays)
            if send_email:
                send_report(html, today, bet_count)
            return bet_count
        else:
            logger.warning("Code-only mode requested but no state file found for today — running full analysis")

    # ------------------------------------------------------------------ #
    #  Settle yesterday's picks (idempotent — skips already-settled bets)
    # ------------------------------------------------------------------ #
    try:
        settled = check_and_settle(today)
        if settled:
            logger.info(f"Outcome settlement: {settled} pick(s) closed from yesterday")
    except Exception as e:
        logger.warning(f"Outcome settlement failed (non-fatal): {e}")

    try:
        settled_props = check_and_settle_props(today)
        if settled_props:
            logger.info(f"Prop settlement: {settled_props} prop(s) closed from yesterday")
    except Exception as e:
        logger.warning(f"Prop settlement failed (non-fatal): {e}")

    # NHL: date-based settlement (games finish overnight, well before 9am run)
    try:
        settled_nhl = check_and_settle_watchlist(today)
        if settled_nhl:
            logger.info(f"NHL watchlist settlement: {settled_nhl} pick(s) closed from yesterday")
    except Exception as e:
        logger.warning(f"NHL watchlist settlement failed (non-fatal): {e}")

    # IPL (+ future rolling-pending leagues): settle any picks whose games are now final
    _now_utc = datetime.now(timezone.utc)
    try:
        settled_pending = settle_watchlist_pending(_now_utc)
        if settled_pending:
            logger.info(f"Rolling pending settlement: {settled_pending} pick(s) settled")
    except Exception as e:
        logger.warning(f"Rolling pending settlement failed (non-fatal): {e}")

    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY is not set.")
        errors.append("ODDS_API_KEY missing — odds data unavailable. Set the secret in GitHub.")

    # ------------------------------------------------------------------ #
    #  NBA
    # ------------------------------------------------------------------ #
    nba_display_raw = []
    nba_singles_raw = []
    nba_game_count  = 0
    nba_props_raw   = []
    nba_props_display_raw = []

    if "nba" in leagues:
        logger.info("=== NBA Analysis ===")
        nba_odds_games = []
        if ODDS_API_KEY:
            nba_odds_games = get_game_odds(NBA_SPORT)
        # Fetch Odds API player prop lines for each NBA game
        for _g in nba_odds_games:
            try:
                _g["player_props"] = fetch_player_props(_g["game_id"], NBA_SPORT)
            except Exception as _e:
                logger.warning(f"NBA player props fetch failed ({_g.get('home_team')}): {_e}")
                _g["player_props"] = {}
        # Surface any prop API error so it appears in the report
        _prop_err = get_last_api_error()
        if _prop_err and not any(_g.get("player_props") for _g in nba_odds_games):
            errors.append(f"NBA player props unavailable: {_prop_err}")

        nba_game_count = len(nba_odds_games)

        if nba_game_count == 0:
            api_err = get_last_api_error()
            if api_err:
                errors.append(f"NBA odds unavailable: {api_err}")
                logger.error(f"NBA odds unavailable: {api_err}")
            else:
                logger.info("No NBA games today or odds unavailable")
        else:
            team_names_today = list({
                t for g in nba_odds_games
                for t in [g["home_team"], g["away_team"]]
            })
            try:
                nba_ctx = get_nba_context(today, team_names=team_names_today)
            except Exception as e:
                logger.error(f"NBA stats fetch failed: {e}")
                nba_ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}, "team_leaders": {}}
                errors.append(f"NBA stats partially unavailable: {e}")

            try:
                nba_injuries = get_nba_injuries()
            except Exception as e:
                logger.warning(f"NBA injuries unavailable: {e}")
                nba_injuries = {}

            for game in nba_odds_games:
                try:
                    recs = analyze_nba_game(game, nba_ctx, nba_injuries, min_edge=0.0)
                    nba_display_raw.extend(recs)
                except Exception as e:
                    logger.error(f"NBA game analysis error ({game.get('home_team')}): {e}")

            nba_singles_raw = [r for r in nba_display_raw if r.edge >= MIN_EDGE]
            nba_props_raw = nba_player_props(nba_odds_games, nba_ctx)
            nba_props_display_raw = nba_player_props(nba_odds_games, nba_ctx, min_edge=0.0)
            logger.info(f"NBA: {len(nba_singles_raw)} qualifying edge(s) ({len(nba_display_raw)} total) "
                        f"across {nba_game_count} games | {len(nba_props_raw)} prop pick(s)")

    # ------------------------------------------------------------------ #
    #  MLB
    # ------------------------------------------------------------------ #
    mlb_display_raw = []
    mlb_singles_raw = []
    mlb_game_count  = 0
    mlb_props_display_raw = []
    mlb_props_raw   = []

    if "mlb" in leagues:
        logger.info("=== MLB Analysis ===")
        mlb_odds_games = []
        if ODDS_API_KEY:
            mlb_odds_games = get_game_odds(MLB_SPORT)
        # Fetch Odds API player prop lines for each MLB game
        for _g in mlb_odds_games:
            try:
                _g["player_props"] = fetch_player_props(_g["game_id"], MLB_SPORT)
            except Exception as _e:
                logger.warning(f"MLB player props fetch failed ({_g.get('home_team')}): {_e}")
                _g["player_props"] = {}
        # Surface any prop API error so it appears in the report
        _prop_err = get_last_api_error()
        if _prop_err and not any(_g.get("player_props") for _g in mlb_odds_games):
            errors.append(f"MLB player props unavailable: {_prop_err}")

        mlb_schedule = get_todays_games(today)
        mlb_game_count = len(mlb_odds_games)

        if mlb_game_count == 0 and ODDS_API_KEY:
            api_err = get_last_api_error()
            if api_err:
                errors.append(f"MLB odds unavailable: {api_err}")
                logger.error(f"MLB odds unavailable: {api_err}")
            else:
                logger.info(f"No MLB odds available today (schedule has {len(mlb_schedule)} games)")

        schedule_map = {g["home_team"]: g for g in mlb_schedule}
        schedule_map.update({g["away_team"]: g for g in mlb_schedule})

        try:
            mlb_injuries = get_mlb_injuries()
        except Exception as e:
            logger.warning(f"MLB injuries unavailable: {e}")
            mlb_injuries = {}

        # Fetch umpires once for all today's games (single API call)
        try:
            umpire_map = get_home_plate_umpires(today)  # {game_pk: umpire_name}
        except Exception as e:
            logger.warning(f"Umpire fetch failed: {e}")
            umpire_map = {}

        pitcher_stats_map: dict = {}

        for game in mlb_odds_games:
            home = game["home_team"]
            away = game["away_team"]

            sched_game = schedule_map.get(home) or schedule_map.get(away)
            if sched_game:
                game.update({
                    "home_pitcher_id":   sched_game.get("home_pitcher_id"),
                    "home_pitcher_name": sched_game.get("home_pitcher_name", "TBD"),
                    "away_pitcher_id":   sched_game.get("away_pitcher_id"),
                    "away_pitcher_name": sched_game.get("away_pitcher_name", "TBD"),
                    "venue":             sched_game.get("venue", ""),
                    "home_team_id":      sched_game.get("home_team_id"),
                    "away_team_id":      sched_game.get("away_team_id"),
                    "game_pk":           sched_game.get("game_pk"),
                })
                # Stamp commence_time back onto schedule game so props
                # (generated from mlb_schedule) get the correct game start time
                # for the client-side lock badge check.
                sched_game["commence_time"] = game.get("commence_time", "")
                sched_game["player_props"] = game.get("player_props", {})

            # Stamp umpire name + k_factor onto game dict (used by edge_finder + props)
            game_pk = game.get("game_pk")
            ump_name = umpire_map.get(game_pk, "")
            ump_tendency = get_umpire_tendency(ump_name)
            game["umpire_name"]     = ump_name
            game["umpire_k_factor"] = ump_tendency.get("k_factor", 1.0)

            # Fetch weather for this venue (cached by city across games at the same park)
            try:
                wx = get_game_weather(game.get("venue", ""), game.get("commence_time", ""))
            except Exception as e:
                logger.debug(f"Weather fetch failed ({game.get('venue', '')}): {e}")
                wx = {}

            try:
                hp_stats  = get_pitcher_stats(game.get("home_pitcher_id"))
                ap_stats  = get_pitcher_stats(game.get("away_pitcher_id"))

                # Merge recent form (last 4 starts) into season stats dicts.
                # edge_finder will blend these with season xFIP for the quality score.
                # Falls back gracefully — if the game log fetch fails, season stats are used as-is.
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
                home_load = get_team_schedule_load(game.get("home_team_id"), today)
                away_load = get_team_schedule_load(game.get("away_team_id"), today)

                if game.get("home_pitcher_name"):
                    pitcher_stats_map[game["home_pitcher_name"]] = hp_stats
                if game.get("away_pitcher_name"):
                    pitcher_stats_map[game["away_pitcher_name"]] = ap_stats

                recs = analyze_mlb_game(game, hp_stats, ap_stats, home_bat, away_bat,
                                        home_bp, away_bp, mlb_injuries,
                                        home_schedule_load=home_load,
                                        away_schedule_load=away_load,
                                        umpire_tendency=ump_tendency,
                                        weather=wx,
                                        min_edge=0.0)
                mlb_display_raw.extend(recs)
            except Exception as e:
                logger.error(f"MLB game analysis error ({home} vs {away}): {e}")

        # Stamp umpire k_factor onto schedule games too (used by props_analyzer)
        for sg in mlb_schedule:
            gpk = sg.get("game_pk")
            ump = umpire_map.get(gpk, "")
            sg["umpire_name"]     = ump
            sg["umpire_k_factor"] = get_umpire_tendency(ump).get("k_factor", 1.0)

        # Propagate singles model projections to schedule games so MLB props can use them.
        # analyze_mlb_game stamps model_total/market_total onto the odds game dict;
        # mlb_player_props uses schedule games, so we copy them across here.
        _odds_by_matchup = {(g["home_team"], g["away_team"]): g for g in mlb_odds_games}
        for _sg in mlb_schedule:
            _og = _odds_by_matchup.get((_sg.get("home_team"), _sg.get("away_team")))
            if _og:
                _sg["model_total"]  = _og.get("model_total")
                _sg["market_total"] = _og.get("market_total")

        mlb_singles_raw = [r for r in mlb_display_raw if r.edge >= MIN_EDGE]
        mlb_props_raw = mlb_player_props(mlb_schedule, pitcher_stats_map)
        mlb_props_display_raw = mlb_player_props(mlb_schedule, pitcher_stats_map, min_edge=0.0)
        logger.info(f"MLB: {len(mlb_singles_raw)} qualifying edge(s) ({len(mlb_display_raw)} total) "
                    f"across {mlb_game_count} games | {len(mlb_props_raw)} prop pick(s)")

    # ------------------------------------------------------------------ #
    #  NFL
    # ------------------------------------------------------------------ #
    nfl_display_raw = []
    nfl_singles_raw = []
    nfl_game_count  = 0

    if "nfl" in leagues:
        if today.month not in SPORT_ACTIVE_MONTHS.get("nfl", []):
            logger.info("NFL is out of season — skipping")
        else:
            logger.info("=== NFL Analysis ===")
            nfl_odds_games = []
            if ODDS_API_KEY:
                nfl_odds_games = get_game_odds(NFL_SPORT)

            nfl_game_count = len(nfl_odds_games)

            if nfl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"NFL odds unavailable: {api_err}")
                    logger.error(f"NFL odds unavailable: {api_err}")
                else:
                    logger.info("No NFL games today or odds unavailable")
            else:
                nfl_team_names = list({
                    t for g in nfl_odds_games
                    for t in [g["home_team"], g["away_team"]]
                })
                try:
                    nfl_ctx = get_nfl_context(today, team_names=nfl_team_names)
                except Exception as e:
                    logger.error(f"NFL stats fetch failed: {e}")
                    nfl_ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}
                    errors.append(f"NFL stats partially unavailable: {e}")

                try:
                    nfl_injuries = get_nfl_injuries()
                except Exception as e:
                    logger.warning(f"NFL injuries unavailable: {e}")
                    nfl_injuries = {}

                for game in nfl_odds_games:
                    try:
                        recs = analyze_nfl_game(game, nfl_ctx, nfl_injuries, min_edge=0.0)
                        nfl_display_raw.extend(recs)
                    except Exception as e:
                        logger.error(f"NFL game analysis error ({game.get('home_team')}): {e}")

                nfl_singles_raw = [r for r in nfl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"NFL: {len(nfl_singles_raw)} qualifying edge(s) ({len(nfl_display_raw)} total) "
                            f"across {nfl_game_count} games")

    # ------------------------------------------------------------------ #
    #  NHL
    # ------------------------------------------------------------------ #
    nhl_display_raw = []
    nhl_singles_raw = []
    nhl_game_count  = 0

    if "nhl" in leagues:
        if today.month not in SPORT_ACTIVE_MONTHS.get("nhl", []):
            logger.info("NHL is out of season — skipping")
        else:
            logger.info("=== NHL Analysis ===")
            nhl_odds_games = []
            if ODDS_API_KEY:
                nhl_odds_games = get_game_odds(NHL_SPORT)

            nhl_game_count = len(nhl_odds_games)

            if nhl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"NHL odds unavailable: {api_err}")
                    logger.error(f"NHL odds unavailable: {api_err}")
                else:
                    logger.info("No NHL games today or odds unavailable")
            else:
                nhl_team_names = list({
                    t for g in nhl_odds_games
                    for t in [g["home_team"], g["away_team"]]
                })
                try:
                    nhl_ctx = get_nhl_context(today, team_names=nhl_team_names)
                except Exception as e:
                    logger.error(f"NHL stats fetch failed: {e}")
                    nhl_ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}
                    errors.append(f"NHL stats partially unavailable: {e}")

                try:
                    nhl_injuries = get_nhl_injuries()
                except Exception as e:
                    logger.warning(f"NHL injuries unavailable: {e}")
                    nhl_injuries = {}

                for game in nhl_odds_games:
                    try:
                        recs = analyze_nhl_game(game, nhl_ctx, nhl_injuries, min_edge=0.0)
                        nhl_display_raw.extend(recs)
                    except Exception as e:
                        logger.error(f"NHL game analysis error ({game.get('home_team')}): {e}")

                nhl_singles_raw = [r for r in nhl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"NHL: {len(nhl_singles_raw)} qualifying edge(s) ({len(nhl_display_raw)} total) "
                            f"across {nhl_game_count} games")

    # ------------------------------------------------------------------ #
    #  IPL (watchlist only — never enters budget allocation or parlays)
    # ------------------------------------------------------------------ #
    ipl_display_raw = []
    ipl_game_count  = 0

    if "ipl" in leagues:
        if today.month not in SPORT_ACTIVE_MONTHS.get("ipl", []):
            logger.info("IPL is out of season — skipping")
        else:
            logger.info("=== IPL Analysis ===")
            ipl_odds_games = []
            if ODDS_API_KEY:
                # IPL games start at ~7am PST and finish ~11am — after the 9am run.
                # Use a 36-hour lookahead so tomorrow's match is always captured.
                ipl_odds_games = get_game_odds(IPL_SPORT, hours_lookahead=36)

            ipl_game_count = len(ipl_odds_games)

            if ipl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"IPL odds unavailable: {api_err}")
                    logger.error(f"IPL odds unavailable: {api_err}")
                else:
                    logger.info("No IPL games today or odds unavailable")
            else:
                ipl_team_names = list({
                    t for g in ipl_odds_games
                    for t in [g["home_team"], g["away_team"]]
                })
                ipl_matchups = [
                    (g["home_team"], g["away_team"]) for g in ipl_odds_games
                ]
                try:
                    ipl_ctx = get_ipl_context(
                        today,
                        team_names=ipl_team_names,
                        matchups=ipl_matchups,
                    )
                except Exception as e:
                    logger.error(f"IPL stats fetch failed: {e}")
                    ipl_ctx = {"season_form": {}, "rest_days": {}, "venue_stats": {}, "venue_config": {}, "match_venues": {}, "h2h": {}, "match_flags": {}, "unavailabilities": {}}
                    errors.append(f"IPL stats partially unavailable: {e}")

                for game in ipl_odds_games:
                    try:
                        recs = analyze_ipl_game(game, ipl_ctx, min_edge=0.0)
                        ipl_display_raw.extend(recs)
                    except Exception as e:
                        logger.error(f"IPL game analysis error ({game.get('home_team')}): {e}")

                ipl_qualifying = [r for r in ipl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"IPL: {len(ipl_qualifying)} qualifying edge(s) ({len(ipl_display_raw)} total) "
                            f"across {ipl_game_count} games")

    # ------------------------------------------------------------------ #
    #  WNBA (watchlist only — same as NHL, never enters budget)
    # ------------------------------------------------------------------ #
    wnba_display_raw = []
    wnba_game_count  = 0

    if "wnba" in leagues:
        if today.month not in SPORT_ACTIVE_MONTHS.get("wnba", []):
            logger.info("WNBA is out of season — skipping")
        else:
            logger.info("=== WNBA Analysis ===")
            wnba_odds_games = []
            if ODDS_API_KEY:
                wnba_odds_games = get_game_odds(WNBA_SPORT)

            wnba_game_count = len(wnba_odds_games)

            if wnba_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"WNBA odds unavailable: {api_err}")
                else:
                    logger.info("No WNBA games today or odds unavailable")
            else:
                wnba_team_names = list({
                    t for g in wnba_odds_games
                    for t in [g["home_team"], g["away_team"]]
                })
                try:
                    wnba_ctx = get_wnba_context(today, team_names=wnba_team_names)
                except Exception as e:
                    logger.error(f"WNBA stats fetch failed: {e}")
                    wnba_ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}
                    errors.append(f"WNBA stats partially unavailable: {e}")

                try:
                    wnba_injuries = get_wnba_injuries()
                except Exception as e:
                    logger.warning(f"WNBA injuries unavailable: {e}")
                    wnba_injuries = {}

                for game in wnba_odds_games:
                    try:
                        recs = analyze_wnba_game(game, wnba_ctx, wnba_injuries, min_edge=0.0)
                        wnba_display_raw.extend(recs)
                    except Exception as e:
                        logger.error(f"WNBA game analysis error ({game.get('home_team')}): {e}")

                wnba_qualifying = [r for r in wnba_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"WNBA: {len(wnba_qualifying)} qualifying edge(s) ({len(wnba_display_raw)} total) across {wnba_game_count} games")

    # ------------------------------------------------------------------ #
    #  Build parlays from raw BetRecommendation objects (before serialising)
    # ------------------------------------------------------------------ #
    # Budget-qualifying picks only (edge >= MIN_EDGE) — used for allocation table + parlays
    # NHL, IPL, and WNBA are watchlist-only and never enter budget allocation or parlays
    all_singles_raw = nba_singles_raw + mlb_singles_raw + nfl_singles_raw
    # All positive-EV picks (no min-edge threshold) — used for display in league sections
    # NHL display picks are handled separately via nhl_watchlist in generator.py
    all_display_raw = nba_display_raw + mlb_display_raw + nfl_display_raw + nhl_display_raw
    parlays_raw          = build_parlays(all_singles_raw)
    props_raw            = nba_props_raw + mlb_props_raw
    props_display_raw    = nba_props_display_raw + mlb_props_display_raw

    # ------------------------------------------------------------------ #
    #  Serialise to dicts (template-ready + state-storable)
    # ------------------------------------------------------------------ #
    _sorted_raw   = sorted(all_singles_raw,
                           key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge))

    # Dedup: one bet per (game, team/direction) — keep only the higher-edge one
    # when the same underlying bet appears twice (e.g. after a line move).
    # • ML/Spread: keyed by (game, team-we're-backing)
    # • Total:     keyed by (game, "over"/"under") — line moves create a new pick
    #              label but it's still the same over/under bet on the same game.
    _seen_game_bet: set = set()
    _deduped_raw = []
    for _r in _sorted_raw:
        if _r.bet_type in ("Moneyline", "Spread"):
            _team = next(
                (t for t in [_r.home_team, _r.away_team] if _r.pick.startswith(t)),
                _r.pick,
            )
            _key = (_r.game, _team)
        elif _r.bet_type == "Total":
            _direction = "over" if "Over" in _r.pick or "over" in _r.pick else "under"
            _key = (_r.game, _direction)
        else:
            _key = (_r.game, _r.pick)   # fallback — exact match
        if _key in _seen_game_bet:
            continue
        _seen_game_bet.add(_key)
        _deduped_raw.append(_r)

    fresh_singles = [bet_to_dict(r) for r in _deduped_raw[:MAX_SINGLE_BETS]]
    # Full uncapped list — passed to merge_picks so signal refresh works even for
    # bets that dropped out of the top-5 since the morning run.
    fresh_singles_all = [bet_to_dict(r) for r in _deduped_raw]
    fresh_parlays = [parlay_to_dict(p) for p in parlays_raw]
    fresh_props   = [prop_to_dict(p)   for p in props_raw]

    # Display picks — all positive-EV bets (no MIN_EDGE gate), deduplicated per sport.
    # Used for per-league section cards; budget allocation still uses fresh_singles.
    _display_sorted = sorted(all_display_raw,
                             key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge))
    _display_seen: set = set()
    _display_deduped = []
    for _r in _display_sorted:
        if _r.bet_type in ("Moneyline", "Spread"):
            _team = next(
                (t for t in [_r.home_team, _r.away_team] if _r.pick.startswith(t)),
                _r.pick,
            )
            _dkey = (_r.sport, _r.game, _team)
        elif _r.bet_type == "Total":
            _direction = "over" if "Over" in _r.pick or "over" in _r.pick else "under"
            _dkey = (_r.sport, _r.game, _direction)
        else:
            _dkey = (_r.sport, _r.game, _r.pick)
        if _dkey not in _display_seen:
            _display_seen.add(_dkey)
            _display_deduped.append(_r)
    fresh_singles_display = [bet_to_dict(r) for r in _display_deduped]

    # Display props — all positive-EV props (no MIN_PROP_EDGE gate).
    fresh_props_display = [prop_to_dict(p) for p in props_display_raw]

    # IPL display picks — serialised separately (never enter budget pool).
    fresh_ipl_display = [bet_to_dict(r) for r in sorted(
        ipl_display_raw, key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge)
    )]

    # WNBA display picks — serialised separately (watchlist only, never in budget pool).
    fresh_wnba_display = [bet_to_dict(r) for r in sorted(
        wnba_display_raw, key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge)
    )]

    # ------------------------------------------------------------------ #
    #  IPL rolling pending management
    #  New upcoming picks from today's odds run are merged into the
    #  persistent pending list (state/watchlist_pending.json).  The full
    #  pending list — which may contain both an in-play pick (started but
    #  not yet settled) and a fresh upcoming pick — becomes fresh_ipl_display
    #  so the report always shows the correct live picture.
    # ------------------------------------------------------------------ #
    if "ipl" in leagues and today.month in SPORT_ACTIVE_MONTHS.get("ipl", []):
        _wl_pending  = load_watchlist_pending()
        _pending_keys = {p.get("game_key", "") for p in _wl_pending}

        # Add any new upcoming picks the odds run found today (dedup by game_key)
        for _bet in fresh_ipl_display:
            _commence_str = _bet.get("commence_time", "")
            _game_date_str = _commence_str[:10] if len(_commence_str) >= 10 else today.isoformat()
            _game_key = f"{_game_date_str}|{_bet.get('game', '')}"

            if _game_key in _pending_keys:
                continue  # Already being tracked

            # Only add games that haven't started yet — in-progress games would
            # have been added on a prior run; if they're missing from pending it
            # means they were already settled or came from a different API window.
            try:
                _cdt = datetime.fromisoformat(_commence_str.replace("Z", "+00:00"))
                if _cdt <= _now_utc:
                    continue
            except Exception:
                pass

            _wl_pending.append({**_bet, "game_key": _game_key, "sport": "IPL"})
            _pending_keys.add(_game_key)
            logger.info(f"IPL pending: added upcoming pick → {_bet.get('pick')} ({_game_key})")

        # Annotate each pending pick with its current status so the template
        # and JS can show the right badge without extra computation.
        for _p in _wl_pending:
            try:
                _cdt = datetime.fromisoformat(
                    _p.get("commence_time", "").replace("Z", "+00:00")
                )
                _p["status"] = "upcoming" if _cdt > _now_utc else "in_progress"
            except Exception:
                _p["status"] = "in_progress"

        save_watchlist_pending(_wl_pending)

        # Include today's settled picks so the card shows "Match Ended + WON/LOST"
        _today_settled = [
            {**r, "status": "settled"}
            for r in load_watchlist_today_settled("IPL", today)
        ]

        # The report shows all picks: settled (with result) + in-play + upcoming
        fresh_ipl_display = _today_settled + _wl_pending
        ipl_game_count    = len(fresh_ipl_display)
        logger.info(
            f"IPL pending: {len(_wl_pending)} unsettled pick(s) "
            f"({sum(1 for p in _wl_pending if p.get('status')=='in_progress')} in-play, "
            f"{sum(1 for p in _wl_pending if p.get('status')=='upcoming')} upcoming), "
            f"{len(_today_settled)} settled today"
        )

    # ------------------------------------------------------------------ #
    #  State management — lock morning picks, merge on subsequent runs
    # ------------------------------------------------------------------ #
    state = load_state(today)

    # Default: display picks = fresh picks (first run, or subsequent full run)
    final_props_display = fresh_props_display

    if state is None:
        # ── First run of the day ─────────────────────────────────────────
        logger.info("First run today — saving picks as morning baseline")
        final_singles         = fresh_singles
        final_parlays         = fresh_parlays
        final_props           = fresh_props
        final_singles_display = fresh_singles_display
        change_warnings       = []

        save_state(today, {
            "date":          today.isoformat(),
            "first_run_at":  datetime.now(timezone.utc).isoformat(),
            "singles":       final_singles,
            "singles_display": final_singles_display,
            # Write-once morning backup — never overwritten by afternoon runs.
            # Subsequent runs use this as the preservation baseline so that
            # watchlist picks (NHL, etc.) survive even if afternoon analysis
            # produces zero picks due to context failures.
            "morning_singles_display": final_singles_display,
            "parlays":       final_parlays,
            "props":         final_props,
            "props_display": fresh_props_display,
            "warnings":      [],
            "odds_api_credits": get_api_credits(),
            "nba_game_count": nba_game_count,
            "mlb_game_count": mlb_game_count,
            "nfl_game_count": nfl_game_count,
            "nhl_game_count": nhl_game_count,
            "ipl_game_count":  ipl_game_count,
            "ipl_display":     fresh_ipl_display,
            "wnba_game_count": wnba_game_count,
            "wnba_display":    fresh_wnba_display,
        })
    else:
        # ── Subsequent run — merge locked picks with new analysis ────────
        mode = "re-evaluate + replace unlocked" if reevaluate else "refresh signals only"
        logger.info(f"Subsequent run — merging with morning baseline ({mode})")
        final_singles, final_parlays, final_props, change_warnings = merge_picks(
            state, fresh_singles, fresh_parlays, fresh_props,
            all_fresh_singles=fresh_singles_all,
            allow_replace=reevaluate,
        )

        # ── Preserve morning display picks for sports where the re-run produced
        # no display picks despite having live odds (game_count > 0).
        # Two common causes:
        #   • Analysis context fetch failed (nhl_ctx errors) → zero picks returned
        #   • The Odds API dropped NHL lines after games started → game_count = 0
        # Rule: if a sport had morning display picks AND games exist in the odds
        # (game_count > 0) but the fresh analysis produced nothing, keep morning.
        # If game_count = 0 (no games today / off-season), do NOT preserve.
        _sports_with_games = {
            sp for sp, cnt in [
                ("NBA", nba_game_count), ("MLB", mlb_game_count),
                ("NFL", nfl_game_count), ("NHL", nhl_game_count),
                ("WNBA", wnba_game_count),
            ] if cnt > 0
        }
        # Use the write-once morning backup as the preservation source.
        # Falls back to singles_display for states written before this key existed.
        _morning_display = (
            state.get("morning_singles_display")
            or state.get("singles_display")
            or []
        )
        _morning_sports  = {r.get("sport") for r in _morning_display if r.get("sport")}
        _fresh_sports    = {r.get("sport") for r in fresh_singles_display if r.get("sport")}

        # Sports skipped by a league filter this run — always preserve their morning picks.
        _not_analyzed = {
            sp for sp, flag in [
                ("NBA", "nba"), ("MLB", "mlb"), ("NFL", "nfl"), ("NHL", "nhl"),
            ] if flag not in leagues
        }

        # Preserve: (1) sport not in this run's league filter, OR
        #           (2) sport was analyzed, had games in odds, but produced no display picks
        #               (typically a context-fetch failure).
        _preserve_sports = _not_analyzed | ((_morning_sports & _sports_with_games) - _fresh_sports)
        if _preserve_sports:
            logger.info(
                f"Subsequent run: preserving morning display picks for {_preserve_sports}"
            )
        _preserved_display = [r for r in _morning_display if r.get("sport") in _preserve_sports]

        # Always carry forward locked picks from analyzed sports too — once a game
        # starts, that pick must never disappear from display regardless of fresh output.
        _morning_locked = [
            {**r, "locked": True}
            for r in _morning_display
            if r.get("sport") not in _preserve_sports
            and _game_started(r.get("commence_time", ""))
        ]

        final_singles_display = fresh_singles_display + _preserved_display + _morning_locked

        # Preserve morning game counts for sports whose odds vanish once games start.
        # The odds API stops listing a game the moment it begins, so a subsequent run
        # sees 0 games even though picks exist — falling back to the morning count keeps
        # the header summary and has_* flags correct.
        nba_game_count  = nba_game_count  or state.get("nba_game_count", 0)
        mlb_game_count  = mlb_game_count  or state.get("mlb_game_count", 0)
        nfl_game_count  = nfl_game_count  or state.get("nfl_game_count", 0)
        nhl_game_count  = nhl_game_count  or state.get("nhl_game_count", 0)
        wnba_game_count = wnba_game_count or state.get("wnba_game_count", 0)

        # Props display: preserve morning props for sports not analyzed this run,
        # and any locked props (game already started) from analyzed sports.
        _morning_props_display = state.get("props_display") or []
        _preserved_props = [
            p for p in _morning_props_display
            if p.get("sport") in _not_analyzed
            or _game_started(p.get("commence_time", ""))
        ]
        final_props_display = fresh_props_display + _preserved_props

        # Persist updated state — morning_singles_display is intentionally NOT
        # updated here; it stays locked to the first run's value all day.
        save_state(today, {
            "date":          today.isoformat(),
            "first_run_at":  state.get("first_run_at"),
            "singles":       final_singles,
            "singles_display": final_singles_display,
            "morning_singles_display": state.get("morning_singles_display"),
            "parlays":       final_parlays,
            "props":         final_props,
            "props_display": final_props_display,
            "warnings":      change_warnings,
            "odds_api_credits": get_api_credits(),
            "nba_game_count": nba_game_count,
            "mlb_game_count": mlb_game_count,
            "nfl_game_count": nfl_game_count,
            "nhl_game_count":  nhl_game_count,
            "ipl_game_count":  ipl_game_count,
            "ipl_display":     fresh_ipl_display,
            "wnba_game_count": wnba_game_count,
            "wnba_display":    fresh_wnba_display,
        })

        if change_warnings:
            logger.warning(f"{len(change_warnings)} pick change(s) since morning run")
            for w in change_warnings:
                logger.warning(f"  {w.get('reason', w)}")

    # Hydrate narrative/context for any state-loaded picks missing those keys
    final_singles_display = [_hydrate_bet(d) for d in final_singles_display]
    final_singles         = [_hydrate_bet(d) for d in final_singles]
    final_props_display   = [_hydrate_prop(d) for d in final_props_display]
    final_props           = [_hydrate_prop(d) for d in final_props]

    # ------------------------------------------------------------------ #
    #  Build report & render HTML
    # ------------------------------------------------------------------ #
    report_data = build_report(
        run_date=today,
        singles=final_singles,
        singles_display=final_singles_display,
        parlays=final_parlays,
        props=final_props,
        props_display=final_props_display,
        nba_game_count=nba_game_count,
        mlb_game_count=mlb_game_count,
        nfl_game_count=nfl_game_count,
        nhl_game_count=nhl_game_count,
        ipl_game_count=ipl_game_count,
        ipl_display=fresh_ipl_display,
        wnba_game_count=wnba_game_count,
        wnba_display=fresh_wnba_display,
        errors=errors,
        change_warnings=change_warnings,
        odds_api_credits=get_api_credits(),
    )

    template_dir = Path(__file__).parent / "report" / "templates"
    env          = Environment(loader=FileSystemLoader(str(template_dir)))
    template     = env.get_template("report.html")
    html         = template.render(report=report_data)

    out_dir  = Path(REPORT_DIR)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / REPORT_FILE
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {out_path}")

    bet_count = len(report_data["all_singles"]) + len(final_parlays)
    if send_email:
        send_report(html, today, bet_count)

    return bet_count


def main():
    parser = argparse.ArgumentParser(description="Sports Betting Analysis System")
    parser.add_argument("--league", choices=["nba", "mlb", "nfl", "nhl", "ipl", "wnba"], help="Run for one league only")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-evaluate unlocked picks and replace any no longer in the top options")
    parser.add_argument("--code-only", action="store_true",
                        help="Re-render report from saved state — zero Odds API calls (for visual/template deploys)")
    args = parser.parse_args()

    leagues   = [args.league] if args.league else ["nba", "mlb", "nfl", "nhl", "ipl", "wnba"]
    bet_count = run(leagues=leagues, send_email=not args.no_email,
                    reevaluate=args.reevaluate, code_only=args.code_only)
    logger.info(f"Done. {bet_count} bet recommendation(s) generated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
