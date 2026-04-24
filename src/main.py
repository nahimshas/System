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
import os
import sys
from datetime import date
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from src.config import ODDS_API_KEY, REPORT_DIR, REPORT_FILE, NBA_SPORT, MLB_SPORT
from src.data.odds_client import get_game_odds
from src.data.nba_stats import get_nba_context
from src.data.mlb_stats import (
    get_todays_games, get_pitcher_stats, get_team_batting_stats,
    get_bullpen_stats,
)
from src.data.injuries import get_nba_injuries, get_mlb_injuries
from src.models.edge_finder import analyze_nba_game, analyze_mlb_game
from src.models.parlay_builder import build_parlays
from src.models.props_analyzer import nba_player_props, mlb_player_props
from src.report.generator import build_report
from src.report.email_sender import send_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run(leagues: list[str], send_email: bool = True) -> int:
    today = date.today()
    errors: list[str] = []

    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY is not set. Set it as an environment variable or GitHub Secret.")
        errors.append("ODDS_API_KEY missing — odds data unavailable. Set the secret in GitHub.")

    # ------------------------------------------------------------------ #
    #  NBA
    # ------------------------------------------------------------------ #
    nba_singles = []
    nba_game_count = 0
    nba_props = []

    if "nba" in leagues:
        logger.info("=== NBA Analysis ===")
        nba_odds_games = []
        if ODDS_API_KEY:
            nba_odds_games = get_game_odds(NBA_SPORT)

        nba_game_count = len(nba_odds_games)

        if nba_game_count == 0:
            logger.info("No NBA games today or odds unavailable")
        else:
            try:
                nba_ctx = get_nba_context(today)
            except Exception as e:
                logger.error(f"NBA stats fetch failed: {e}")
                nba_ctx = {"season_stats": {}, "recent_form": {}, "rest_days": {}}
                errors.append(f"NBA stats partially unavailable: {e}")

            try:
                nba_injuries = get_nba_injuries()
            except Exception as e:
                logger.warning(f"NBA injuries unavailable: {e}")
                nba_injuries = {}

            for game in nba_odds_games:
                try:
                    recs = analyze_nba_game(game, nba_ctx, nba_injuries)
                    nba_singles.extend(recs)
                except Exception as e:
                    logger.error(f"NBA game analysis error ({game.get('home_team')}): {e}")

            nba_props = nba_player_props(nba_odds_games, nba_ctx)
            logger.info(f"NBA: {len(nba_singles)} edge(s) found across {nba_game_count} games")

    # ------------------------------------------------------------------ #
    #  MLB
    # ------------------------------------------------------------------ #
    mlb_singles = []
    mlb_game_count = 0
    mlb_props = []

    if "mlb" in leagues:
        logger.info("=== MLB Analysis ===")
        mlb_odds_games = []
        if ODDS_API_KEY:
            mlb_odds_games = get_game_odds(MLB_SPORT)

        mlb_schedule = get_todays_games(today)
        mlb_game_count = len(mlb_odds_games)

        # Build a lookup of schedule data (pitcher info) keyed by team name
        schedule_map = {g["home_team"]: g for g in mlb_schedule}
        schedule_map.update({g["away_team"]: g for g in mlb_schedule})

        try:
            mlb_injuries = get_mlb_injuries()
        except Exception as e:
            logger.warning(f"MLB injuries unavailable: {e}")
            mlb_injuries = {}

        pitcher_stats_map: dict = {}

        for game in mlb_odds_games:
            home = game["home_team"]
            away = game["away_team"]

            # Match to schedule for pitcher IDs
            sched_game = schedule_map.get(home) or schedule_map.get(away)
            if sched_game:
                game.update({
                    "home_pitcher_id": sched_game.get("home_pitcher_id"),
                    "home_pitcher_name": sched_game.get("home_pitcher_name", "TBD"),
                    "away_pitcher_id": sched_game.get("away_pitcher_id"),
                    "away_pitcher_name": sched_game.get("away_pitcher_name", "TBD"),
                    "venue": sched_game.get("venue", ""),
                    "home_team_id": sched_game.get("home_team_id"),
                    "away_team_id": sched_game.get("away_team_id"),
                })

            try:
                hp_stats = get_pitcher_stats(game.get("home_pitcher_id"))
                ap_stats = get_pitcher_stats(game.get("away_pitcher_id"))
                home_bat = get_team_batting_stats(game.get("home_team_id"))
                away_bat = get_team_batting_stats(game.get("away_team_id"))
                home_bp = get_bullpen_stats(game.get("home_team_id"))
                away_bp = get_bullpen_stats(game.get("away_team_id"))

                if game.get("home_pitcher_name"):
                    pitcher_stats_map[game["home_pitcher_name"]] = hp_stats
                if game.get("away_pitcher_name"):
                    pitcher_stats_map[game["away_pitcher_name"]] = ap_stats

                recs = analyze_mlb_game(game, hp_stats, ap_stats, home_bat, away_bat, home_bp, away_bp, mlb_injuries)
                mlb_singles.extend(recs)
            except Exception as e:
                logger.error(f"MLB game analysis error ({home} vs {away}): {e}")

        mlb_props = mlb_player_props(mlb_schedule, pitcher_stats_map)
        logger.info(f"MLB: {len(mlb_singles)} edge(s) found across {mlb_game_count} games")

    # ------------------------------------------------------------------ #
    #  Parlays & Report
    # ------------------------------------------------------------------ #
    all_singles = nba_singles + mlb_singles
    parlays = build_parlays(all_singles)
    props = nba_props + mlb_props

    report_data = build_report(
        run_date=today,
        nba_singles=nba_singles,
        mlb_singles=mlb_singles,
        parlays=parlays,
        props=props,
        nba_game_count=nba_game_count,
        mlb_game_count=mlb_game_count,
        errors=errors,
    )

    # Render HTML
    template_dir = Path(__file__).parent / "report" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html")
    html = template.render(report=report_data)

    # Save to docs/ for GitHub Pages
    out_dir = Path(REPORT_DIR)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / REPORT_FILE
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {out_path}")

    # Email
    bet_count = len(report_data["all_singles"]) + len(parlays)
    if send_email:
        send_report(html, today, bet_count)

    return bet_count


def main():
    parser = argparse.ArgumentParser(description="Sports Betting Analysis System")
    parser.add_argument("--league", choices=["nba", "mlb"], help="Run for one league only")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    args = parser.parse_args()

    leagues = [args.league] if args.league else ["nba", "mlb"]
    bet_count = run(leagues=leagues, send_email=not args.no_email)
    logger.info(f"Done. {bet_count} bet recommendation(s) generated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
