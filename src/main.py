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

from src.config import ODDS_API_KEY, REPORT_DIR, REPORT_FILE, NBA_SPORT, MLB_SPORT, MAX_SINGLE_BETS
from src.data.odds_client import get_game_odds, get_last_api_error, get_api_credits, fetch_player_props
from src.data.nba_stats import get_nba_context
from src.data.mlb_stats import (
    get_todays_games, get_pitcher_stats, get_team_batting_stats,
    get_bullpen_stats, get_team_schedule_load,
)
from src.data.injuries import get_nba_injuries, get_mlb_injuries
from src.data.umpire import get_home_plate_umpires, get_umpire_tendency
from src.data.weather import get_game_weather
from src.models.edge_finder import analyze_nba_game, analyze_mlb_game
from src.models.parlay_builder import build_parlays
from src.models.props_analyzer import nba_player_props, mlb_player_props
from src.state.manager import (
    load_state, save_state, merge_picks,
    bet_to_dict, parlay_to_dict, prop_to_dict,
)
from src.data.outcome_checker import check_and_settle, check_and_settle_props
from src.report.generator import build_report
from src.report.email_sender import send_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


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

            report_data = build_report(
                run_date=today,
                singles=final_singles,
                parlays=final_parlays,
                props=final_props,
                nba_game_count=0,
                mlb_game_count=0,
                errors=errors,
                change_warnings=change_warnings,
                odds_api_credits=None,
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

    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY is not set.")
        errors.append("ODDS_API_KEY missing — odds data unavailable. Set the secret in GitHub.")

    # ------------------------------------------------------------------ #
    #  NBA
    # ------------------------------------------------------------------ #
    nba_singles_raw = []
    nba_game_count  = 0
    nba_props_raw   = []

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
                    recs = analyze_nba_game(game, nba_ctx, nba_injuries)
                    nba_singles_raw.extend(recs)
                except Exception as e:
                    logger.error(f"NBA game analysis error ({game.get('home_team')}): {e}")

            nba_props_raw = nba_player_props(nba_odds_games, nba_ctx)
            logger.info(f"NBA: {len(nba_singles_raw)} edge(s) found across {nba_game_count} games | "
                        f"{len(nba_props_raw)} prop pick(s)")

    # ------------------------------------------------------------------ #
    #  MLB
    # ------------------------------------------------------------------ #
    mlb_singles_raw = []
    mlb_game_count  = 0
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
                                        weather=wx)
                mlb_singles_raw.extend(recs)
            except Exception as e:
                logger.error(f"MLB game analysis error ({home} vs {away}): {e}")

        # Stamp umpire k_factor onto schedule games too (used by props_analyzer)
        for sg in mlb_schedule:
            gpk = sg.get("game_pk")
            ump = umpire_map.get(gpk, "")
            sg["umpire_name"]     = ump
            sg["umpire_k_factor"] = get_umpire_tendency(ump).get("k_factor", 1.0)

        mlb_props_raw = mlb_player_props(mlb_schedule, pitcher_stats_map)
        logger.info(f"MLB: {len(mlb_singles_raw)} edge(s) found across {mlb_game_count} games | "
                    f"{len(mlb_props_raw)} prop pick(s)")

    # ------------------------------------------------------------------ #
    #  Build parlays from raw BetRecommendation objects (before serialising)
    # ------------------------------------------------------------------ #
    all_singles_raw = nba_singles_raw + mlb_singles_raw
    parlays_raw     = build_parlays(all_singles_raw)
    props_raw       = nba_props_raw + mlb_props_raw

    # ------------------------------------------------------------------ #
    #  Serialise to dicts (template-ready + state-storable)
    # ------------------------------------------------------------------ #
    _sorted_raw   = sorted(all_singles_raw,
                           key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge))

    # Dedup: one bet per (game, team) — if both ML and Spread for the same
    # team make the rankings, keep only the higher-edge one.
    # Totals are exempt (no single team to deduplicate against).
    _seen_game_team: set = set()
    _deduped_raw = []
    for _r in _sorted_raw:
        if _r.bet_type in ("Moneyline", "Spread"):
            # Identify which team we're backing
            _team = next(
                (t for t in [_r.home_team, _r.away_team] if _r.pick.startswith(t)),
                _r.pick,
            )
            _key = (_r.game, _team)
            if _key in _seen_game_team:
                continue
            _seen_game_team.add(_key)
        _deduped_raw.append(_r)

    fresh_singles = [bet_to_dict(r) for r in _deduped_raw[:MAX_SINGLE_BETS]]
    # Full uncapped list — passed to merge_picks so signal refresh works even for
    # bets that dropped out of the top-5 since the morning run.
    fresh_singles_all = [bet_to_dict(r) for r in _deduped_raw]
    fresh_parlays = [parlay_to_dict(p) for p in parlays_raw]
    fresh_props   = [prop_to_dict(p)   for p in props_raw]

    # ------------------------------------------------------------------ #
    #  State management — lock morning picks, merge on subsequent runs
    # ------------------------------------------------------------------ #
    state = load_state(today)

    if state is None:
        # ── First run of the day ─────────────────────────────────────────
        logger.info("First run today — saving picks as morning baseline")
        final_singles    = fresh_singles
        final_parlays    = fresh_parlays
        final_props      = fresh_props
        change_warnings  = []

        save_state(today, {
            "date":          today.isoformat(),
            "first_run_at":  datetime.now(timezone.utc).isoformat(),
            "singles":       final_singles,
            "parlays":       final_parlays,
            "props":         final_props,
            "warnings":      [],
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

        # Persist updated state (locked flags, any substitutions)
        save_state(today, {
            "date":          today.isoformat(),
            "first_run_at":  state.get("first_run_at"),
            "singles":       final_singles,
            "parlays":       final_parlays,
            "props":         final_props,
            "warnings":      change_warnings,
        })

        if change_warnings:
            logger.warning(f"{len(change_warnings)} pick change(s) since morning run")
            for w in change_warnings:
                logger.warning(f"  {w.get('reason', w)}")

    # ------------------------------------------------------------------ #
    #  Build report & render HTML
    # ------------------------------------------------------------------ #
    report_data = build_report(
        run_date=today,
        singles=final_singles,
        parlays=final_parlays,
        props=final_props,
        nba_game_count=nba_game_count,
        mlb_game_count=mlb_game_count,
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
    parser.add_argument("--league", choices=["nba", "mlb"], help="Run for one league only")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-evaluate unlocked picks and replace any no longer in the top options")
    parser.add_argument("--code-only", action="store_true",
                        help="Re-render report from saved state — zero Odds API calls (for visual/template deploys)")
    args = parser.parse_args()

    leagues   = [args.league] if args.league else ["nba", "mlb"]
    bet_count = run(leagues=leagues, send_email=not args.no_email,
                    reevaluate=args.reevaluate, code_only=args.code_only)
    logger.info(f"Done. {bet_count} bet recommendation(s) generated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
