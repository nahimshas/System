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
    MAX_SINGLE_BETS, MAX_PROPS_PER_SPORT, MIN_EDGE,
)
from src.data.odds_client import get_last_api_error, get_api_credits
from src.models.parlay_builder import build_parlays
# Sport modules — each encapsulates fetch / context / analyze / props for its sport
from src.sports.nba      import nba
from src.sports.mlb      import mlb
from src.sports.nfl      import nfl
from src.sports.nhl      import nhl
from src.sports.ipl      import ipl
from src.sports.wnba     import wnba
from src.sports.mls      import mls
from src.sports.registry import REGISTRY

# Maps every registry slug to its module singleton — used by the analysis loop.
# Add a new entry here when a new sport module is created.
SPORT_MODULES: dict = {
    "nba": nba, "mlb": mlb, "nfl": nfl, "nhl": nhl,
    "ipl": ipl, "wnba": wnba, "mls": mls,
}
from src.state.manager import (
    load_state, save_state, merge_picks,
    bet_to_dict, parlay_to_dict, prop_to_dict,
    _game_started, _update_lock_flags,
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


def _slot_sort_key(rec):
    """
    Canonical sort key for slot-filling decisions across budget singles, props,
    display singles, and own-tile sports.

    Confidence-first: HIGH-confidence picks always rank above MEDIUM regardless
    of edge magnitude. Within each confidence tier, higher *effective* edge
    wins — where effective_edge = raw_edge × calibration_ratio.

    During Phase 0 (no calibration data yet), calibration_ratio is 1.0 for
    every sport × market, so this behaves identically to ranking by raw edge.
    As shadow log data accumulates and sports auto-promote to Phase A/B, the
    same chokepoint progressively factors in realised hit rates without any
    other code changes.
    """
    try:
        from src.state.calibration import effective_edge
        eff = effective_edge(rec)
    except Exception:
        eff = float(getattr(rec, "edge", 0.0) or 0.0)
    return (0 if rec.confidence == "HIGH" else 1, -eff)


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
    from src.state.manager import _prop_stat_avg
    out = d
    if "narrative" not in d:
        narrative, context = build_prop_context(
            d.get("sport", ""), d.get("prop_type", ""), d.get("player", ""),
            d.get("team", ""), d.get("opponent", ""),
            d.get("signals", []), d.get("research", []),
            d.get("model_line", d.get("market_line", 0)),
            d.get("market_line", 0),
            d.get("edge_pct", 0) / 100,
        )
        out = {**out, "narrative": narrative, "context": context}
    if "prop_stat_avg" not in d:
        out = {**out, "prop_stat_avg": _prop_stat_avg(d.get("signals", []), d.get("prop_type", ""))}
    return out


def _write_sw(out_dir: Path, run_date) -> None:
    """Rewrite sw.js with a date-stamped cache name so browsers detect a new SW on each report."""
    sw_path = out_dir / "sw.js"
    if not sw_path.exists():
        return
    try:
        content = sw_path.read_text(encoding="utf-8")
        import re as _re
        new_content = _re.sub(
            r"const CACHE = 'picks-[^']*';",
            f"const CACHE = 'picks-{run_date.isoformat()}';",
            content,
        )
        if new_content != content:
            sw_path.write_text(new_content, encoding="utf-8")
            logger.info(f"sw.js updated: cache key → picks-{run_date.isoformat()}")
    except Exception as e:
        logger.warning(f"sw.js update skipped: {e}")


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
            # Use the state file's own date so TODAY_DATE in the JS matches the picks'
            # game dates. Without this, ESPN scoreboard calls use today's date and
            # can't match picks from a prior day (e.g. morning hasn't run yet).
            state_date_str = state.get("date", "")
            if state_date_str:
                try:
                    from datetime import date as _date
                    today = _date.fromisoformat(state_date_str)
                except ValueError:
                    pass
            final_singles   = state.get("singles", [])
            final_parlays   = state.get("parlays", [])
            final_props     = state.get("props",   [])
            change_warnings = state.get("warnings", [])
            saved_credits   = state.get("odds_api_credits", {})
            final_singles_display = state.get("singles_display") or final_singles
            final_props_display   = state.get("props_display")   or final_props
            # Own-tile display picks — loaded per registered own-tile sport.
            _own_displays_loaded = {
                _s: state.get(f"{_s}_display", [])
                for _s in REGISTRY
                if not REGISTRY[_s].caps.in_main_display_pool
            }
            final_ipl_display  = _own_displays_loaded.get("ipl",  [])
            final_wnba_display = _own_displays_loaded.get("wnba", [])
            final_mls_display  = _own_displays_loaded.get("mls",  [])

            # Apply current lock flags — state was written before some games started,
            # so re-computing here ensures the re-rendered HTML shows correct badges.
            _update_lock_flags(final_singles)
            _update_lock_flags(final_singles_display)
            _update_lock_flags(final_props)
            _update_lock_flags(final_props_display)

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

            try:
                spa_template = jinja_env.get_template("report_spa.html")
                spa_html     = spa_template.render(report=report_data)
                spa_path     = out_dir / "index_spa.html"
                spa_path.write_text(spa_html, encoding="utf-8")
                logger.info(f"SPA report written to {spa_path} (code-only)")
            except Exception as e:
                logger.warning(f"SPA render skipped: {e}")

            _write_sw(out_dir, today)
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

    # Shadow log: propagate outcomes from the now-updated history files into
    # matching shadow log entries (calibration data). Runs AFTER all existing
    # settlers so it picks up the freshest outcomes. Non-fatal — never blocks
    # the report if it fails.
    try:
        from src.state.shadow_log import settle_from_history
        _settled_shadow = settle_from_history()
        if _settled_shadow:
            logger.info(f"Shadow log settlement: {_settled_shadow} entries closed from yesterday")
    except Exception as _e_shadow_settle:
        logger.warning(f"Shadow log settlement failed (non-fatal): {_e_shadow_settle}")

    # Cap auto-relaxation: counterfactual analysis on cap-fired picks decides
    # whether each credibility cap should widen, tighten, or stay. Throttled
    # to once per 30 days per cap with hard safety bounds (±5% to ±30%).
    # Runs after shadow log settlement so it has the freshest outcomes.
    # Non-fatal — falls back to constant cap values if anything fails.
    try:
        from src.state.cap_state import evaluate_and_adjust_caps
        _cap_adjusts = evaluate_and_adjust_caps()
        if _cap_adjusts:
            logger.info(f"Cap auto-adjustment: {_cap_adjusts} cap(s) updated")
    except Exception as _e_cap_eval:
        logger.warning(f"Cap auto-adjustment failed (non-fatal): {_e_cap_eval}")

    if not ODDS_API_KEY:
        logger.error("ODDS_API_KEY is not set.")
        errors.append("ODDS_API_KEY missing — odds data unavailable. Set the secret in GitHub.")

    # ------------------------------------------------------------------ #
    #  Sport analysis — registry-driven loop
    #
    #  Capability flags (SportCapabilities) route each sport's output:
    #    enters_budget      → qualifying picks enter all_singles_raw (budget pool)
    #    in_main_display_pool → display picks enter all_display_raw (main card)
    #    has_props          → sport.fetch_props() called; results enter props pools
    #    own-tile sports    → picks stored in own_display[slug] instead
    #
    #  Adding a new sport: create the module, register it in REGISTRY,
    #  add it to SPORT_MODULES at the top of this file. No other edits needed.
    # ------------------------------------------------------------------ #
    all_singles_raw:   list = []   # budget picks (edge >= MIN_EDGE) → budget pool + parlays
    all_display_raw:   list = []   # main display pool (budget sports + NHL)
    props_raw:         list = []   # qualified props for P&L tracking
    props_display_raw: list = []   # all positive-EV props for display
    game_counts:       dict = {}   # slug → int  (0 if skipped / out of season)
    own_display:       dict = {}   # slug → list[BetRecommendation]  (own-tile sports)

    for slug, entry in REGISTRY.items():
        if slug not in leagues:
            continue
        caps = entry.caps
        if caps.active_months and today.month not in caps.active_months:
            logger.info(f"{entry.label} is out of season — skipping")
            continue

        module = SPORT_MODULES[slug]
        logger.info(f"=== {entry.label} Analysis ===")

        games = module.fetch_games(today_str) if ODDS_API_KEY else []
        game_counts[slug] = len(games)

        # Surface prop-line API errors before checking game count
        if caps.has_props and games and not any(g.get("player_props") for g in games):
            _prop_err = get_last_api_error()
            if _prop_err:
                errors.append(f"{entry.label} player props unavailable: {_prop_err}")

        if not games:
            _api_err = get_last_api_error()
            if _api_err:
                errors.append(f"{entry.label} odds unavailable: {_api_err}")
            continue

        ctx         = module.fetch_context(today_str, games)
        display_raw = module.analyze_games(games, ctx)
        qualifying  = [r for r in display_raw if r.edge >= MIN_EDGE]

        # Route display picks based on capability flags
        if caps.in_main_display_pool:
            all_display_raw.extend(display_raw)
            if caps.enters_budget:
                all_singles_raw.extend(qualifying)
        else:
            own_display[slug] = display_raw   # IPL / WNBA / MLS → own tile

        # Collect props (only if sport has them)
        _sport_props_raw = []
        if caps.has_props:
            _sport_props_raw      = module.fetch_props(games, ctx, min_edge=MIN_EDGE)
            _sport_props_display  = module.fetch_props(games, ctx, min_edge=0.0)
            props_raw.extend(_sport_props_raw)
            props_display_raw.extend(_sport_props_display)

        _prop_str = f" | {len(_sport_props_raw)} prop pick(s)" if caps.has_props else ""
        logger.info(
            f"{entry.label}: {len(qualifying)} qualifying edge(s) ({len(display_raw)} total) "
            f"across {game_counts[slug]} games{_prop_str}"
        )

    # Extract named game counts — used by state management and build_report.
    # Phase 5 will generalise these into registry-driven dicts.
    nba_game_count  = game_counts.get("nba", 0)
    mlb_game_count  = game_counts.get("mlb", 0)
    nfl_game_count  = game_counts.get("nfl", 0)
    nhl_game_count  = game_counts.get("nhl", 0)
    ipl_game_count  = game_counts.get("ipl", 0)
    wnba_game_count = game_counts.get("wnba", 0)
    mls_game_count  = game_counts.get("mls", 0)

    # Extract named display lists for own-tile sports (IPL / WNBA / MLS).
    ipl_display_raw  = own_display.get("ipl",  [])
    wnba_display_raw = own_display.get("wnba", [])
    mls_display_raw  = own_display.get("mls",  [])

    # ------------------------------------------------------------------ #
    #  Build parlays — budget-qualifying picks only (edge >= MIN_EDGE).
    #  all_singles_raw and props pools are already assembled by the loop.
    # ------------------------------------------------------------------ #
    parlays_raw = build_parlays(all_singles_raw)

    # ------------------------------------------------------------------ #
    #  Serialise to dicts (template-ready + state-storable)
    # ------------------------------------------------------------------ #
    _sorted_raw   = sorted(all_singles_raw, key=_slot_sort_key)

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
    # Cap props at MAX_PROPS_PER_SPORT per sport before saving to state.
    # Without this cap, when min_edge is explicitly passed to fetch_props() the
    # analyzer's internal [:6] guard is bypassed and all qualifying props are
    # returned — which can be 100+ for MLB — bloating the state and causing
    # mass re-settlement on subsequent runs.
    _props_sorted = sorted(props_raw, key=_slot_sort_key)
    _props_sport_counts: dict = {}
    _props_capped: list = []
    for _pr in _props_sorted:
        _ps = getattr(_pr, "sport", "")
        if _props_sport_counts.get(_ps, 0) < MAX_PROPS_PER_SPORT:
            _props_capped.append(_pr)
            _props_sport_counts[_ps] = _props_sport_counts.get(_ps, 0) + 1
    fresh_props   = [prop_to_dict(p) for p in _props_capped]

    # Display picks — all positive-EV bets (no MIN_EDGE gate), deduplicated per sport.
    # Used for per-league section cards; budget allocation still uses fresh_singles.
    _display_sorted = sorted(all_display_raw, key=_slot_sort_key)
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

    # Own-tile display picks — serialised per sport (IPL / WNBA / MLS / future sports).
    # IPL pending section will overwrite fresh_own_displays["ipl"] below.
    # Cap non-IPL own-tile sports at MAX_SINGLE_BETS (top 5) — without this cap
    # WNBA/MLS can produce 40+ picks that clutter their tile sections.
    fresh_own_displays: dict[str, list] = {
        slug: [bet_to_dict(r) for r in sorted(raw, key=_slot_sort_key)]
              [:MAX_SINGLE_BETS if slug != "ipl" else len(raw)]
        for slug, raw in own_display.items()
    }
    # Named aliases — kept for backward-compat with the IPL pending section,
    # carry-forward logic, hydration, and build_report parameters.
    fresh_ipl_display  = fresh_own_displays.get("ipl",  [])
    fresh_wnba_display = fresh_own_displays.get("wnba", [])
    fresh_mls_display  = fresh_own_displays.get("mls",  [])

    # ------------------------------------------------------------------ #
    #  Shadow log — calibration data foundation.
    #  Records every pick the model produced (not just the top-5 displayed)
    #  so the calibration engine can later learn per-sport realised hit
    #  rates and auto-adjust edges. Idempotent across re-runs (stable keys);
    #  entries are frozen once commence_time passes. Failure here is
    #  non-fatal and never blocks the report.
    # ------------------------------------------------------------------ #
    try:
        from src.state.shadow_log import record_picks, compute_top_keys

        # Picks that made the top-5 display slot (singles + per-sport own tiles).
        # Used to mark `displayed_in_top` so we can later analyse whether
        # displayed picks outperform shadow-only picks.
        _top_displayed_recs = list(_deduped_raw[:MAX_SINGLE_BETS])
        for _slug, _raw in own_display.items():
            _own_sorted = sorted(_raw, key=_slot_sort_key)
            _cap = len(_raw) if _slug == "ipl" else MAX_SINGLE_BETS
            _top_displayed_recs.extend(_own_sorted[:_cap])
        _top_keys = compute_top_keys(_top_displayed_recs, today)

        # Pool of ALL picks the model produced (top-5 + the rest).
        # We dedup by stable key inside record_picks(), so passing the union
        # of every source is safe and ensures we don't miss anything.
        _shadow_pool = []
        _shadow_pool.extend(all_display_raw)        # NBA/MLB/NFL/NHL display
        for _raw in own_display.values():           # IPL/WNBA/MLS
            _shadow_pool.extend(_raw)
        _shadow_pool.extend(props_display_raw)      # all positive-EV props

        record_picks(_shadow_pool, today, displayed_top_keys=_top_keys)
    except Exception as _shadow_err:
        # Defence in depth — record_picks() already swallows internals,
        # but catch anything that escapes (e.g. import errors).
        logger.error(f"Shadow log integration failed (non-fatal): {_shadow_err}")

    # Persist calibration state for the panel layer. Computed lazily on first
    # call (already used by _slot_sort_key above when applicable), so this
    # mainly writes the snapshot to disk for external inspection / display.
    try:
        from src.state.calibration import persist_state as _persist_calib
        _persist_calib()
    except Exception as _calib_err:
        logger.error(f"Calibration persist failed (non-fatal): {_calib_err}")

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

        # Include today's settled picks so the card shows "Match Ended + WON/LOST".
        # Deduplicate by game in case history has multiple records for the same
        # game (e.g. from a buggy settlement run followed by a manual correction).
        # If both WON and LOST exist for the same game, the LOST record wins
        # (manual corrections always change WON→LOST, never the reverse).
        _today_settled_raw = load_watchlist_today_settled("IPL", today)
        _settled_by_game: dict = {}
        for _sr in _today_settled_raw:
            _sg = (_sr.get("game", ""), _sr.get("pick", ""))
            if _sg not in _settled_by_game or _sr.get("result") == "LOST":
                _settled_by_game[_sg] = _sr
        _today_settled = [
            {**r, "status": "settled"}
            for r in _settled_by_game.values()
        ]

        # The report shows all picks: settled (with result) + in-play + upcoming
        fresh_ipl_display = _today_settled + _wl_pending
        ipl_game_count    = len(fresh_ipl_display)
        # Sync so registry-driven save_state picks up the pending-managed values.
        fresh_own_displays["ipl"] = fresh_ipl_display
        game_counts["ipl"]        = ipl_game_count
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
            # Per-sport game counts — one key per registry sport.
            **{f"{_s}_game_count": game_counts.get(_s, 0) for _s in REGISTRY},
            # Own-tile display picks — one key per own-tile sport.
            # For IPL the pending section has already updated fresh_own_displays["ipl"].
            **{f"{_s}_display": fresh_own_displays.get(_s, [])
               for _s in REGISTRY if not REGISTRY[_s].caps.in_main_display_pool},
            # Write-once morning backups for carry-forward.
            # IPL is excluded — its display is managed by the rolling pending section.
            **{f"morning_{_s}_display": fresh_own_displays.get(_s, [])
               for _s in REGISTRY
               if not REGISTRY[_s].caps.in_main_display_pool
               and not REGISTRY[_s].caps.uses_pending_file},
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
            REGISTRY[slug].label
            for slug, cnt in game_counts.items()
            if cnt > 0
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
            entry.label
            for slug, entry in REGISTRY.items()
            if entry.caps.in_main_display_pool and slug not in leagues
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

        # Preserve morning game counts for sports whose odds vanish once games start,
        # AND for sports not analyzed on this run (e.g. --league ipl only).
        # The odds API stops listing a game the moment it begins, so a subsequent run
        # sees 0 games even though picks exist — falling back to the morning count keeps
        # the header summary and has_* flags correct.
        # Skips uses_pending_file sports (IPL): their count is managed by the pending
        # section above and already reflected in game_counts["ipl"].
        for _slug in REGISTRY:
            if not REGISTRY[_slug].caps.uses_pending_file and not game_counts.get(_slug, 0):
                game_counts[_slug] = state.get(f"{_slug}_game_count", 0)
        # Sync named variables after the fallback pass.
        nba_game_count  = game_counts.get("nba",  nba_game_count)
        mlb_game_count  = game_counts.get("mlb",  mlb_game_count)
        nfl_game_count  = game_counts.get("nfl",  nfl_game_count)
        nhl_game_count  = game_counts.get("nhl",  nhl_game_count)
        ipl_game_count  = game_counts.get("ipl",  ipl_game_count)
        wnba_game_count = game_counts.get("wnba", wnba_game_count)
        mls_game_count  = game_counts.get("mls",  mls_game_count)

        # Carry forward locked picks for own-tile sports (WNBA, MLS, any future sport).
        # Reads from write-once morning backup — same pattern as budget singles.
        # Skips IPL: its display is fully managed by the rolling pending section above.
        for _own_slug, _own_entry in REGISTRY.items():
            if _own_entry.caps.in_main_display_pool or _own_entry.caps.uses_pending_file:
                continue
            _morning_own = (
                state.get(f"morning_{_own_slug}_display")
                or state.get(f"{_own_slug}_display", [])
            )
            # Cap the morning backup to MAX_SINGLE_BETS so that state files
            # written before the cap was introduced don't flood the carry-forward.
            _morning_own = _morning_own[:MAX_SINGLE_BETS]
            if not _morning_own:
                continue
            _fresh_own      = fresh_own_displays.setdefault(_own_slug, [])
            _fresh_own_keys = {(r.get("home_team"), r.get("away_team")) for r in _fresh_own}
            for _or in _morning_own:
                if _game_started(_or.get("commence_time", "")):
                    _okey = (_or.get("home_team"), _or.get("away_team"))
                    if _okey not in _fresh_own_keys:
                        _fresh_own.append({**_or, "locked": True})
                        _fresh_own_keys.add(_okey)
                        logger.info(
                            f"{_own_entry.label}: carrying forward locked pick "
                            f"'{_or.get('pick')}' ({_or.get('game')})"
                        )
        # Refresh named aliases after the carry-forward loop.
        fresh_wnba_display = fresh_own_displays.get("wnba", fresh_wnba_display)
        fresh_mls_display  = fresh_own_displays.get("mls",  fresh_mls_display)

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
            # Per-sport game counts — falls back to morning state for sports not
            # analyzed this run (e.g. filtered out by --league).
            **{f"{_s}_game_count": game_counts.get(_s, state.get(f"{_s}_game_count", 0))
               for _s in REGISTRY},
            # Own-tile display picks — falls back to morning state for unanalyzed sports.
            # Always capped at MAX_SINGLE_BETS so stale oversized states are corrected.
            **{f"{_s}_display": (fresh_own_displays.get(_s, state.get(f"{_s}_display", [])))[:MAX_SINGLE_BETS]
               for _s in REGISTRY if not REGISTRY[_s].caps.in_main_display_pool},
            # Morning backups — intentionally NOT updated; stay locked to first-run values.
            **{f"morning_{_s}_display": state.get(f"morning_{_s}_display")
               for _s in REGISTRY
               if not REGISTRY[_s].caps.in_main_display_pool
               and not REGISTRY[_s].caps.uses_pending_file},
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
    # Hydrate all own-tile sports in one pass, then refresh named aliases.
    fresh_own_displays = {
        _s: [_hydrate_bet(d) for d in _disp]
        for _s, _disp in fresh_own_displays.items()
    }
    fresh_ipl_display  = fresh_own_displays.get("ipl",  fresh_ipl_display)
    fresh_wnba_display = fresh_own_displays.get("wnba", fresh_wnba_display)
    fresh_mls_display  = fresh_own_displays.get("mls",  fresh_mls_display)

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
        fresh_odds=True,   # full run always fetches live odds
    )

    template_dir = Path(__file__).parent / "report" / "templates"
    env          = Environment(loader=FileSystemLoader(str(template_dir)))

    # Classic report
    template     = env.get_template("report.html")
    html         = template.render(report=report_data)

    out_dir  = Path(REPORT_DIR)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / REPORT_FILE
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {out_path}")

    # SPA report
    try:
        spa_template = env.get_template("report_spa.html")
        spa_html     = spa_template.render(report=report_data)
        spa_path     = out_dir / "index_spa.html"
        spa_path.write_text(spa_html, encoding="utf-8")
        logger.info(f"SPA report written to {spa_path}")
    except Exception as e:
        logger.warning(f"SPA render skipped: {e}")

    _write_sw(out_dir, today)
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
