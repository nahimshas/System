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
    MAX_SINGLE_BETS, MIN_EDGE,
)
from src.data.odds_client import get_last_api_error, get_api_credits
from src.models.parlay_builder import build_parlays
# Sport modules — each encapsulates fetch / context / analyze / props for its sport
from src.sports.nba  import nba
from src.sports.mlb  import mlb
from src.sports.nfl  import nfl
from src.sports.nhl  import nhl
from src.sports.ipl  import ipl
from src.sports.wnba import wnba
from src.sports.mls  import mls
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
    # Re-compute if narrative key is missing OR empty (e.g. saved before a sport's
    # narrative was implemented — WNBA/IPL added May 2026).
    if d.get("narrative"):
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
    today     = _today_pacific()
    today_str = today.isoformat()   # string form used by sport module APIs
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
            final_mls_display     = state.get("mls_display", [])

            # Hydrate narrative/context for picks saved before the card context feature
            final_singles_display = [_hydrate_bet(d) for d in final_singles_display]
            final_singles         = [_hydrate_bet(d) for d in final_singles]
            final_props_display   = [_hydrate_prop(d) for d in final_props_display]
            final_props           = [_hydrate_prop(d) for d in final_props]
            final_mls_display     = [_hydrate_bet(d) for d in final_mls_display]

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
                mls_display=final_mls_display,
                mls_game_count=state.get("mls_game_count", 0),
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
    #  MLS / WNBA / IPL / NHL watchlist init
    # ------------------------------------------------------------------ #
    mls_display_raw = []
    mls_game_count  = 0

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
        nba_games = nba.fetch_games(today_str) if ODDS_API_KEY else []
        nba_game_count = len(nba_games)

        _prop_err = get_last_api_error()
        if _prop_err and not any(g.get("player_props") for g in nba_games):
            errors.append(f"NBA player props unavailable: {_prop_err}")

        if nba_game_count == 0:
            api_err = get_last_api_error()
            if api_err:
                errors.append(f"NBA odds unavailable: {api_err}")
        else:
            nba_ctx = nba.fetch_context(today_str, nba_games)
            if not nba_ctx.get("season_stats") and not nba_ctx.get("recent_form"):
                errors.append("NBA stats partially unavailable")
            nba_display_raw = nba.analyze_games(nba_games, nba_ctx)
            nba_singles_raw = [r for r in nba_display_raw if r.edge >= MIN_EDGE]
            nba_props_raw         = nba.fetch_props(nba_games, nba_ctx, min_edge=MIN_EDGE)
            nba_props_display_raw = nba.fetch_props(nba_games, nba_ctx, min_edge=0.0)
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
        mlb_games = mlb.fetch_games(today_str) if ODDS_API_KEY else []
        mlb_game_count = len(mlb_games)

        _prop_err = get_last_api_error()
        if _prop_err and not any(g.get("player_props") for g in mlb_games):
            errors.append(f"MLB player props unavailable: {_prop_err}")

        if mlb_game_count == 0 and ODDS_API_KEY:
            api_err = get_last_api_error()
            if api_err:
                errors.append(f"MLB odds unavailable: {api_err}")
        else:
            mlb_ctx = mlb.fetch_context(today_str, mlb_games)
            # analyze_games fetches per-game pitcher/weather/bullpen stats and
            # populates mlb_ctx["pitcher_stats_map"] + propagates model totals
            # onto mlb_ctx["schedule"] so the props analyzer can use them.
            mlb_display_raw = mlb.analyze_games(mlb_games, mlb_ctx)
            mlb_singles_raw = [r for r in mlb_display_raw if r.edge >= MIN_EDGE]
            mlb_props_raw         = mlb.fetch_props(mlb_games, mlb_ctx, min_edge=MIN_EDGE)
            mlb_props_display_raw = mlb.fetch_props(mlb_games, mlb_ctx, min_edge=0.0)
            logger.info(f"MLB: {len(mlb_singles_raw)} qualifying edge(s) ({len(mlb_display_raw)} total) "
                        f"across {mlb_game_count} games | {len(mlb_props_raw)} prop pick(s)")

    # ------------------------------------------------------------------ #
    #  NFL
    # ------------------------------------------------------------------ #
    nfl_display_raw = []
    nfl_singles_raw = []
    nfl_game_count  = 0

    if "nfl" in leagues:
        if today.month not in nfl.caps.active_months:
            logger.info("NFL is out of season — skipping")
        else:
            logger.info("=== NFL Analysis ===")
            nfl_games = nfl.fetch_games(today_str) if ODDS_API_KEY else []
            nfl_game_count = len(nfl_games)

            if nfl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"NFL odds unavailable: {api_err}")
            else:
                nfl_ctx = nfl.fetch_context(today_str, nfl_games)
                if not nfl_ctx.get("season_stats") and not nfl_ctx.get("recent_form"):
                    errors.append("NFL stats partially unavailable")
                nfl_display_raw = nfl.analyze_games(nfl_games, nfl_ctx)
                nfl_singles_raw = [r for r in nfl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"NFL: {len(nfl_singles_raw)} qualifying edge(s) ({len(nfl_display_raw)} total) "
                            f"across {nfl_game_count} games")

    # ------------------------------------------------------------------ #
    #  NHL (display-only in main card — never enters budget or parlays)
    # ------------------------------------------------------------------ #
    nhl_display_raw = []
    nhl_singles_raw = []
    nhl_game_count  = 0

    if "nhl" in leagues:
        if today.month not in nhl.caps.active_months:
            logger.info("NHL is out of season — skipping")
        else:
            logger.info("=== NHL Analysis ===")
            nhl_games = nhl.fetch_games(today_str) if ODDS_API_KEY else []
            nhl_game_count = len(nhl_games)

            if nhl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"NHL odds unavailable: {api_err}")
            else:
                nhl_ctx = nhl.fetch_context(today_str, nhl_games)
                if not nhl_ctx.get("season_stats") and not nhl_ctx.get("recent_form"):
                    errors.append("NHL stats partially unavailable")
                nhl_display_raw = nhl.analyze_games(nhl_games, nhl_ctx)
                nhl_singles_raw = [r for r in nhl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"NHL: {len(nhl_singles_raw)} qualifying edge(s) ({len(nhl_display_raw)} total) "
                            f"across {nhl_game_count} games")

    # ------------------------------------------------------------------ #
    #  IPL (watchlist only — never enters budget allocation or parlays)
    # ------------------------------------------------------------------ #
    ipl_display_raw = []
    ipl_game_count  = 0

    if "ipl" in leagues:
        if today.month not in ipl.caps.active_months:
            logger.info("IPL is out of season — skipping")
        else:
            logger.info("=== IPL Analysis ===")
            ipl_games = ipl.fetch_games(today_str) if ODDS_API_KEY else []
            ipl_game_count = len(ipl_games)

            if ipl_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"IPL odds unavailable: {api_err}")
            else:
                ipl_ctx = ipl.fetch_context(today_str, ipl_games)
                if not ipl_ctx.get("season_form"):
                    errors.append("IPL stats partially unavailable")
                ipl_display_raw = ipl.analyze_games(ipl_games, ipl_ctx)
                ipl_qualifying = [r for r in ipl_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"IPL: {len(ipl_qualifying)} qualifying edge(s) ({len(ipl_display_raw)} total) "
                            f"across {ipl_game_count} games")

    # ------------------------------------------------------------------ #
    #  WNBA (watchlist only — never enters budget)
    # ------------------------------------------------------------------ #
    wnba_display_raw = []
    wnba_game_count  = 0

    if "wnba" in leagues:
        if today.month not in wnba.caps.active_months:
            logger.info("WNBA is out of season — skipping")
        else:
            logger.info("=== WNBA Analysis ===")
            wnba_games = wnba.fetch_games(today_str) if ODDS_API_KEY else []
            wnba_game_count = len(wnba_games)

            if wnba_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"WNBA odds unavailable: {api_err}")
            else:
                wnba_ctx = wnba.fetch_context(today_str, wnba_games)
                wnba_display_raw = wnba.analyze_games(wnba_games, wnba_ctx)
                wnba_qualifying = [r for r in wnba_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"WNBA: {len(wnba_qualifying)} qualifying edge(s) ({len(wnba_display_raw)} total) "
                            f"across {wnba_game_count} games")

    # ------------------------------------------------------------------ #
    #  MLS (watchlist only — never enters budget)
    # ------------------------------------------------------------------ #

    if "mls" in leagues:
        if today.month not in mls.caps.active_months:
            logger.info("MLS is out of season — skipping")
        else:
            logger.info("=== MLS Analysis ===")
            mls_games = mls.fetch_games(today_str) if ODDS_API_KEY else []
            mls_game_count = len(mls_games)

            if mls_game_count == 0:
                api_err = get_last_api_error()
                if api_err:
                    errors.append(f"MLS odds unavailable: {api_err}")
            else:
                mls_ctx = mls.fetch_context(today_str, mls_games)
                mls_display_raw = mls.analyze_games(mls_games, mls_ctx)
                mls_qualifying = [r for r in mls_display_raw if r.edge >= MIN_EDGE]
                logger.info(f"MLS: {len(mls_qualifying)} qualifying edge(s) ({len(mls_display_raw)} total) "
                            f"across {mls_game_count} games")

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

    # MLS display picks — serialised separately (watchlist only, never in budget pool).
    fresh_mls_display = [bet_to_dict(r) for r in sorted(
        mls_display_raw, key=lambda r: (0 if r.confidence == "HIGH" else 1, -r.edge)
    )]

    # ------------------------------------------------------------------ #
    #  IPL rolling pending management
    #  New upcoming picks from today's odds run are merged into the
    #  persistent pending list (state/watchlist_pending.json).  The full
    #  pending list — which may contain both an in-play pick (started but
    #  not yet settled) and a fresh upcoming pick — becomes fresh_ipl_display
    #  so the report always shows the correct live picture.
    # ------------------------------------------------------------------ #
    if "ipl" in leagues and today.month in ipl.caps.active_months:
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
            # Write-once morning backup — same pattern as morning_singles_display.
            # Subsequent runs read from this so locked picks survive even when
            # the Odds API drops games after they start.
            "morning_wnba_display": fresh_wnba_display,
            "mls_game_count":  mls_game_count,
            "mls_display":     fresh_mls_display,
            "morning_mls_display": fresh_mls_display,
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
                ("WNBA", wnba_game_count), ("MLS", mls_game_count),
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
                ("NBA", "nba"), ("MLB", "mlb"), ("NFL", "nfl"), ("NHL", "nhl"), ("MLS", "mls"),
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
        mls_game_count  = mls_game_count  or state.get("mls_game_count", 0)

        # Carry forward locked WNBA picks (odds API drops games once they start,
        # leaving fresh_wnba_display empty even though picks are in progress).
        # Reads from the write-once morning backup (same pattern as budget singles)
        # so a previous run that wiped wnba_display doesn't break carry-forward.
        _morning_wnba = (
            state.get("morning_wnba_display")
            or state.get("wnba_display", [])
        )
        if _morning_wnba:
            _fresh_wnba_keys = {
                (r.get("home_team"), r.get("away_team")) for r in fresh_wnba_display
            }
            for _wr in _morning_wnba:
                if _game_started(_wr.get("commence_time", "")):
                    _wkey = (_wr.get("home_team"), _wr.get("away_team"))
                    if _wkey not in _fresh_wnba_keys:
                        fresh_wnba_display.append({**_wr, "locked": True})
                        _fresh_wnba_keys.add(_wkey)
                        logger.info(
                            f"WNBA: carrying forward locked pick "
                            f"'{_wr.get('pick')}' ({_wr.get('game')})"
                        )

        # Same for MLS — reads from write-once morning backup
        _morning_mls = (
            state.get("morning_mls_display")
            or state.get("mls_display", [])
        )
        if _morning_mls:
            _fresh_mls_keys = {
                (r.get("home_team"), r.get("away_team")) for r in fresh_mls_display
            }
            for _mr in _morning_mls:
                if _game_started(_mr.get("commence_time", "")):
                    _mkey = (_mr.get("home_team"), _mr.get("away_team"))
                    if _mkey not in _fresh_mls_keys:
                        fresh_mls_display.append({**_mr, "locked": True})
                        _fresh_mls_keys.add(_mkey)
                        logger.info(
                            f"MLS: carrying forward locked pick "
                            f"'{_mr.get('pick')}' ({_mr.get('game')})"
                        )

        # Props display: preserve morning props for sports not analyzed this run,
        # and any locked props (game already started) from analyzed sports.
        _morning_props_display = state.get("props_display") or []
        _preserved_props = [
            p for p in _morning_props_display
            if p.get("sport") in _not_analyzed
            or _game_started(p.get("commence_time", ""))
        ]
        final_props_display = fresh_props_display + _preserved_props

        # Persist updated state — morning_*_display keys are intentionally NOT
        # updated here; they stay locked to the first run's values all day.
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
            "morning_wnba_display": state.get("morning_wnba_display"),
            "mls_game_count":  mls_game_count,
            "mls_display":     fresh_mls_display,
            "morning_mls_display": state.get("morning_mls_display"),
        })

        if change_warnings:
            logger.warning(f"{len(change_warnings)} pick change(s) since morning run")
            for w in change_warnings:
                logger.warning(f"  {w.get('reason', w)}")

    # Hydrate narrative/context for any state-loaded picks missing those keys
    # (also catches WNBA/IPL picks saved before those sports had narrative support)
    final_singles_display = [_hydrate_bet(d) for d in final_singles_display]
    final_singles         = [_hydrate_bet(d) for d in final_singles]
    final_props_display   = [_hydrate_prop(d) for d in final_props_display]
    final_props           = [_hydrate_prop(d) for d in final_props]
    fresh_ipl_display     = [_hydrate_bet(d) for d in fresh_ipl_display]
    fresh_wnba_display    = [_hydrate_bet(d) for d in fresh_wnba_display]
    fresh_mls_display     = [_hydrate_bet(d) for d in fresh_mls_display]

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
        mls_display=fresh_mls_display,
        mls_game_count=mls_game_count,
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
    parser.add_argument("--league", choices=["nba", "mlb", "nfl", "nhl", "ipl", "wnba", "mls"], help="Run for one league only")
    parser.add_argument("--no-email", action="store_true", help="Skip email delivery")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-evaluate unlocked picks and replace any no longer in the top options")
    parser.add_argument("--code-only", action="store_true",
                        help="Re-render report from saved state — zero Odds API calls (for visual/template deploys)")
    args = parser.parse_args()

    leagues   = [args.league] if args.league else ["nba", "mlb", "nfl", "nhl", "ipl", "wnba", "mls"]
    bet_count = run(leagues=leagues, send_email=not args.no_email,
                    reevaluate=args.reevaluate, code_only=args.code_only)
    logger.info(f"Done. {bet_count} bet recommendation(s) generated.")
    sys.exit(0)


if __name__ == "__main__":
    main()
