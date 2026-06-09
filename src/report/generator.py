"""Assembles pre-serialised pick dicts into the report context for Jinja templating."""
import json
import logging
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from src.config import DAILY_BUDGET, MAX_SINGLE_BETS

logger = logging.getLogger(__name__)


def _load_calibration_panel() -> Dict:
    """
    Load the persisted calibration state and shape it for the template.

    Returns a dict with:
        has_data:        bool — True if state file exists and has any markets
        computed_at:     ISO timestamp
        markets:         list of per-(sport, market_type) entries with display-friendly fields
        phase_counts:    {"0": N, "A": N, "B": N, "C": N}
        thresholds:      phase sample-size thresholds (for "next milestone" labels)
    """
    out = {
        "has_data":     False,
        "computed_at":  None,
        "markets":      [],
        "phase_counts": {"0": 0, "A": 0, "B": 0, "C": 0},
        "thresholds":   {},
    }
    path = Path("state/calibration_state.json")
    if not path.exists():
        return out
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        logger.debug(f"Calibration panel load failed (non-fatal): {e}")
        return out

    out["computed_at"] = data.get("computed_at")
    out["thresholds"]  = data.get("phase_thresholds", {})

    markets = data.get("markets", []) or []
    phase_a_min = out["thresholds"].get("phase_a_min", 100)
    phase_b_min = out["thresholds"].get("phase_b_min", 400)

    for m in markets:
        phase     = m.get("phase", "0")
        n_settled = int(m.get("n_settled", 0))
        n_total   = int(m.get("n_total", n_settled))
        single_ratio = float(m.get("single_ratio", 1.0))
        # "Applied" = what edges are multiplied by right now. In Phase 0 this is
        # the target shrunk by sample-size credibility; in Phase A+ it equals the
        # target. Fall back to recomputing it if an older state file lacks the
        # field, so the panel works before the next calibration run rewrites state.
        if "applied_ratio" in m:
            applied_ratio = float(m["applied_ratio"])
        elif phase == "0":
            cred = min(1.0, n_settled / phase_a_min) if phase_a_min else 0.0
            applied_ratio = 1.0 + (single_ratio - 1.0) * cred
        else:
            applied_ratio = single_ratio

        # Status label and progress toward next phase
        if phase == "0":
            next_milestone = f"{max(0, phase_a_min - n_settled)} settled picks to Phase A"
            status_label   = "Building data"
            status_class   = "status-building"
        elif phase == "A":
            next_milestone = f"{max(0, phase_b_min - n_settled)} settled picks to Phase B (buckets)"
            status_label   = "Phase A — single ratio active"
            status_class   = "status-active"
        elif phase == "B":
            next_milestone = "Phase B — per-bucket calibration active"
            status_label   = "Phase B — bucket calibration"
            status_class   = "status-active-b"
        elif phase == "C":
            next_milestone = "Phase C — isotonic curve active"
            status_label   = "Phase C — isotonic"
            status_class   = "status-active-c"
        else:
            next_milestone = ""
            status_label   = phase
            status_class   = ""

        # Target = the full measured correction (realized ÷ predicted).
        # Applied = what's multiplying edges right now (ramps toward target as
        # data accumulates). They're equal in Phase A+; in Phase 0 applied is a
        # credibility-weighted fraction of target.
        ratio_pct_drift   = (single_ratio - 1.0) * 100
        applied_pct_drift = (applied_ratio - 1.0) * 100
        # Only show a numeric target once there's enough data to be meaningful.
        if n_settled > 0:
            ratio_display   = f"{single_ratio:.3f} ({ratio_pct_drift:+.1f}%)"
            applied_display = f"{applied_ratio:.3f} ({applied_pct_drift:+.1f}%)"
        else:
            ratio_display   = "—"
            applied_display = "1.000 (+0.0%)"

        out["phase_counts"][phase] = out["phase_counts"].get(phase, 0) + 1
        out["markets"].append({
            "sport":          m.get("sport", ""),
            "market_type":    m.get("market_type", ""),
            "phase":          phase,
            "n_total":        n_total,
            "n_settled":      n_settled,
            "single_ratio":   single_ratio,
            "ratio_display":  ratio_display,
            "ratio_drift_pct": ratio_pct_drift,
            "applied_ratio":   applied_ratio,
            "applied_display": applied_display,
            "applied_drift_pct": applied_pct_drift,
            "predicted_rate_pct": round(float(m.get("single_predicted_rate", 0)) * 100, 1) if n_settled > 0 else None,
            "realized_rate_pct":  round(float(m.get("single_realized_rate",  0)) * 100, 1) if n_settled > 0 else None,
            "status_label":   status_label,
            "status_class":   status_class,
            "next_milestone": next_milestone,
            "buckets":        m.get("buckets", []),
        })

    # Sort: active phases first (more interesting), then by sport / market
    phase_order = {"C": 0, "B": 1, "A": 2, "0": 3}
    out["markets"].sort(key=lambda m: (
        phase_order.get(m["phase"], 9),
        m["sport"],
        m["market_type"],
    ))

    out["has_data"] = bool(out["markets"])
    return out


def _now_pacific_str() -> str:
    now_utc = datetime.now(timezone.utc)
    offset = -7 if 3 <= now_utc.month <= 10 else -8
    label = "PDT" if offset == -7 else "PST"
    pacific = now_utc + timedelta(hours=offset)
    return pacific.strftime(f"%B %d, %Y at %I:%M %p {label}")


def build_report(
    run_date: date,
    singles: List[Dict],          # budget-qualifying picks (edge >= MIN_EDGE), for allocation table
    parlays: List[Dict],
    props: List[Dict],
    nba_game_count: int,
    mlb_game_count: int,
    errors: List[str],
    change_warnings: Optional[List[Dict]] = None,
    odds_api_credits: Optional[Dict] = None,
    nfl_game_count: int = 0,
    nhl_game_count: int = 0,
    ipl_game_count: int = 0,
    ipl_games_analyzed: int = 0,                    # raw Odds API game count (before pending override)
    ipl_display: Optional[List[Dict]] = None,       # all positive-EV IPL picks (watchlist only)
    singles_display: Optional[List[Dict]] = None,   # all positive-EV picks for league section display
    props_display: Optional[List[Dict]] = None,     # all positive-EV props for league section display
    wnba_game_count: int = 0,
    wnba_display: Optional[List[Dict]] = None,
    mls_display: Optional[List[Dict]] = None,
    mls_game_count: int = 0,
    wc_display: Optional[List[Dict]] = None,
    wc_game_count: int = 0,
    fresh_odds: bool = False,   # True only when a full odds-fetch run generated this report
) -> Dict:
    change_warnings = change_warnings or []

    # Display pool: use singles_display (all positive-EV picks) if provided and non-empty,
    # otherwise fall back to singles (budget-qualifying only).
    # NOTE: an empty list means the new code hasn't run yet (old state) or no games today;
    # in both cases falling back to singles is the right thing to do.
    _display = singles_display if singles_display else singles

    # NHL graduated to a budget sport (June 2026) — it now competes for the
    # top-5 budget slots alongside NBA/MLB/NFL (handled below via the shared
    # _top_singles_for_sport + _tag_alloc path). No separate watchlist routing.

    # IPL is watchlist-only — never enters budget allocation or parlays.
    def _wl_sort_key(r):
        return (0 if r.get("confidence") == "HIGH" else 1,
                -r.get("edge", 0),
                -r.get("model_prob_pct", 0))

    ipl_watchlist = sorted(
        ipl_display or [],
        key=_wl_sort_key,
    )[:MAX_SINGLE_BETS]

    # WNBA is watchlist-only — never enters budget allocation or parlays.
    wnba_watchlist = sorted(
        [s for s in (wnba_display or []) if s.get("sport") == "WNBA"],
        key=_wl_sort_key,
    )[:MAX_SINGLE_BETS]

    # MLS is watchlist-only — never enters budget allocation or parlays.
    mls_watchlist = sorted(
        [s for s in (mls_display or []) if s.get("sport") == "MLS"],
        key=_wl_sort_key,
    )[:MAX_SINGLE_BETS]

    # World Cup is watchlist-only — never enters budget allocation or parlays.
    wc_watchlist = sorted(
        [s for s in (wc_display or []) if s.get("sport") == "WC"],
        key=_wl_sort_key,
    )[:MAX_SINGLE_BETS]

    # Decorate settled NHL / WNBA / MLS watchlist cards with status='settled'
    # and result='WON'/'LOST' so the HTML template can stamp data-in-history="1"
    # on them.  Without this flag the JS updateWatchlistTiles() would count them
    # again on top of the data-hist-won/lost baseline that the morning run already
    # wrote (double-count).  IPL is handled separately via watchlist_pending.json.
    try:
        from src.data.outcome_checker import _load_watchlist_history
        from datetime import datetime as _dt
        _wl_hist = _load_watchlist_history()
        # Build lookup: (sport, pick, game, date) → result
        _wl_settled: Dict = {}
        for _r in _wl_hist:
            if _r.get("result") in ("WON", "LOST"):
                _wl_settled[(_r["sport"], _r["pick"], _r["game"], _r["date"])] = _r["result"]

        def _card_date(card: Dict) -> str:
            ct = card.get("commence_time", "")
            if ct:
                try:
                    return _dt.fromisoformat(ct.replace("Z", "+00:00")).date().isoformat()
                except Exception:
                    pass
            return ""

        def _mark_settled(cards: List[Dict], sport: str) -> List[Dict]:
            out = []
            for c in cards:
                key = (sport, c.get("pick", ""), c.get("game", ""), _card_date(c))
                result = _wl_settled.get(key)
                if result:
                    c = {**c, "status": "settled", "result": result}
                out.append(c)
            return out

        wnba_watchlist = _mark_settled(wnba_watchlist, "WNBA")
        mls_watchlist  = _mark_settled(mls_watchlist,  "MLS")
        wc_watchlist   = _mark_settled(wc_watchlist,   "WC")
        ipl_watchlist  = _mark_settled(ipl_watchlist,  "IPL")
    except Exception as _e:
        logger.debug(f"Watchlist settled decoration skipped: {_e}")

    # Top picks: HIGH confidence first, then by edge within each tier.
    # Deduplicate by (sport, home_team, away_team, bet_type) — a line move creates
    # a new pick label but it's the same bet; keep the first (highest-edge) occurrence.
    _seen_game_bets: set = set()
    _deduped_pool = []
    for s in sorted(
        singles,
        key=lambda r: (0 if r["confidence"] == "HIGH" else 1, -r["edge"],
                       -(r.get("model_prob_raw") or r.get("model_prob_pct", 50) / 100)),
    ):
        _key = (s.get("sport", ""), s.get("home_team", ""), s.get("away_team", ""), s.get("bet_type", ""))
        if _key not in _seen_game_bets:
            _seen_game_bets.add(_key)
            _deduped_pool.append(s)
    all_singles = _deduped_pool[:MAX_SINGLE_BETS]
    nba_singles = [s for s in all_singles if s["sport"] == "NBA"]
    mlb_singles = [s for s in all_singles if s["sport"] == "MLB"]
    nfl_singles = [s for s in all_singles if s["sport"] == "NFL"]
    nhl_singles = [s for s in all_singles if s["sport"] == "NHL"]

    # ── Per-league top-5 singles for display in league sections ─────────────
    def _top_singles_for_sport(sport: str) -> List[Dict]:
        """Sort by edge, deduplicate by game+bet_type, cap at MAX_SINGLE_BETS.
        Locked budget picks (game has started) are always included first so they
        remain visible even after the Odds API stops returning their game."""
        seen: set = set()
        out = []
        # 1. Locked budget picks go first — they must always show in their section.
        locked_budget = sorted(
            [s for s in singles if s.get("sport") == sport and s.get("locked")],
            key=lambda r: (-r.get("effective_edge", r["edge"]),
                           -(r.get("model_prob_raw") or r.get("model_prob_pct", 50) / 100)),
        )
        for s in locked_budget:
            k = (s.get("home_team", ""), s.get("away_team", ""), s.get("bet_type", ""))
            if k not in seen:
                seen.add(k)
                out.append(s)
        # 2. Fill remaining slots with display pool (sorted by confidence + effective edge).
        # Uses effective_edge (calibration-adjusted) for consistency with slot selection —
        # a MEDIUM MLB ML at 15% raw edge is only ~13% effective (MLB ratio ≈ 0.87).
        for s in sorted(
            [s for s in _display if s.get("sport") == sport],
            key=lambda r: (0 if r["confidence"] == "HIGH" else 1,
                           -r.get("effective_edge", r["edge"]),
                           -(r.get("model_prob_raw") or r.get("model_prob_pct", 50) / 100)),
        ):
            if len(out) >= MAX_SINGLE_BETS:
                break
            k = (s.get("home_team", ""), s.get("away_team", ""), s.get("bet_type", ""))
            if k not in seen:
                seen.add(k)
                out.append(s)
        # Final display order: confidence-first, then effective edge. Step 1 pins
        # locked (game-started) picks first ONLY to guarantee they keep a slot
        # (the odds API drops started games); without this re-sort a locked MEDIUM
        # would jump ahead of a HIGH-confidence pick once games begin — which is
        # exactly what made the evening MLB tab look mis-sorted vs the morning one.
        out.sort(key=lambda r: (0 if r.get("confidence") == "HIGH" else 1,
                                -r.get("effective_edge", r["edge"]),
                                -(r.get("model_prob_raw") or r.get("model_prob_pct", 50) / 100)))
        return out

    nba_top_singles_raw = _top_singles_for_sport("NBA")
    mlb_top_singles_raw = _top_singles_for_sport("MLB")
    nfl_top_singles_raw = _top_singles_for_sport("NFL")
    nhl_top_singles_raw = _top_singles_for_sport("NHL")   # NHL graduated to budget (Jun 2026)

    # Tag which singles are in the budget allocation pool (rank 1-5).
    # Also override cost/profit/contracts with the allocation pool's locked values so
    # the bet card data-attributes always match what the allocation table displays.
    # Without this, a subsequent run that refreshes odds can silently change the card
    # data-cost/data-profit while the allocation table retains morning-run pricing,
    # causing the live "Today's Profit" JS calculation to diverge from what the user sees.
    _alloc_rank_map = {(r["pick"], r["game"]): i for i, r in enumerate(all_singles, 1)}
    _alloc_data_map = {(r["pick"], r["game"]): r for r in all_singles}

    def _tag_alloc(lst):
        out = []
        for rec in lst:
            key = (rec["pick"], rec["game"])
            rank = _alloc_rank_map.get(key)
            if rank is not None:
                # Snap cost/profit/contract fields to the locked allocation values
                alloc = _alloc_data_map[key]
                rec = {
                    **rec,
                    "alloc_rank":           rank,
                    "total_cost":           alloc["total_cost"],
                    "profit_if_win":        alloc["profit_if_win"],
                    "loss_if_lose":         alloc.get("loss_if_lose", alloc["total_cost"]),
                    "num_contracts":        alloc["num_contracts"],
                    "contract_price_cents": alloc.get("contract_price_cents",
                                                      rec.get("contract_price_cents", 0)),
                    "edge_pct":             alloc["edge_pct"],
                    "model_prob_pct":       alloc.get("model_prob_pct", rec.get("model_prob_pct", 50)),
                    "market_prob_pct":      alloc.get("market_prob_pct", rec.get("market_prob_pct", 50)),
                }
            else:
                rec = {**rec, "alloc_rank": None}
            out.append(rec)
        return out

    nba_top_singles = _tag_alloc(nba_top_singles_raw)
    mlb_top_singles = _tag_alloc(mlb_top_singles_raw)
    nfl_top_singles = _tag_alloc(nfl_top_singles_raw)
    nhl_top_singles = _tag_alloc(nhl_top_singles_raw)

    # ── Per-league top-6 props for display ───────────────────────────────────
    # Locked props (from the curated `props` list) are always shown first so
    # that settled/in-progress picks never disappear from the display after a
    # subsequent run adds higher-ranked fresh picks to props_display.
    from src.config import MAX_PROPS_PER_SPORT
    _props_pool = props_display if props_display else props
    _locked_prop_keys = {
        f"{p.get('player')}|{p.get('prop_type')}"
        for p in props
        if p.get("locked")
    }

    def _build_sport_props(sport: str) -> list:
        locked = [p for p in props if p.get("sport") == sport and p.get("locked")]
        remaining = MAX_PROPS_PER_SPORT - len(locked)
        if remaining <= 0:
            return locked[:MAX_PROPS_PER_SPORT]
        fresh = sorted(
            [p for p in _props_pool
             if p.get("sport") == sport
             and f"{p.get('player')}|{p.get('prop_type')}" not in _locked_prop_keys],
            key=lambda p: (0 if p.get("confidence") == "HIGH" else 1, -p.get("edge_pct", 0)),
        )[:remaining]
        return locked + fresh

    nba_props_display = _build_sport_props("NBA")
    mlb_props_display = _build_sport_props("MLB")

    total_allocated  = sum(r["total_cost"] for r in all_singles)
    parlay_allocated = sum(p["total_cost"] for p in parlays)
    grand_total      = total_allocated + parlay_allocated
    reserve          = max(0.0, DAILY_BUDGET - grand_total)

    # ── Allocation table rows ────────────────────────────────────────────────
    allocation_rows = []
    for i, rec in enumerate(all_singles, 1):
        allocation_rows.append({
            "rank":          i,
            "label":         f"{rec['pick']}" + (" (ML)" if rec['bet_type'] == "Moneyline" else ""),
            "game":          rec["game"],
            "game_time":     rec.get("game_time", ""),
            "sport":         rec["sport"],
            "home_team":     rec.get("home_team", ""),
            "away_team":     rec.get("away_team", ""),
            "edge_pct":      rec["edge_pct"],
            "contracts":     rec["num_contracts"],
            "cost":          rec["total_cost"],
            "profit_if_win": rec["profit_if_win"],
            "pct_of_budget": round(rec["total_cost"] / DAILY_BUDGET * 100, 1),
            "confidence":    rec.get("kelly_fraction", 0),
            "locked":        rec.get("locked", False),
            "commence_time": rec.get("commence_time", ""),
            # Needed by JS for win-prob / WON-LOST
            "market_prob_pct": rec.get("market_prob_pct", 50),
            "bet_type":      rec["bet_type"],
            "pick":          rec["pick"],
            "contract_price_cents": rec.get("contract_price_cents", 0),
            "loss_if_lose":         rec.get("loss_if_lose", rec.get("total_cost", 0)),
            "expected_value":       rec.get("expected_value", 0),
            "conf_label":           rec.get("confidence", "MEDIUM"),
        })

    for j, par in enumerate(parlays, 1):
        # Build a compact legs JSON for live win-prob in the allocation table
        legs_json = json.dumps([{
            "home_team":      l.get("home_team", ""),
            "away_team":      l.get("away_team", ""),
            "bet_type":       l.get("bet_type", ""),
            "pick":           l.get("pick", ""),
            "sport":          l.get("sport", ""),
            "model_prob_pct": l.get("model_prob_pct", 50),
            "commence_time":  l.get("commence_time", ""),
        } for l in par.get("legs", [])])

        # Parlay is locked if any leg's game has started
        def _clean_parlay_label(raw: str) -> str:
            """Strip/replace bet_type suffixes baked into saved state labels."""
            import re
            raw = re.sub(r'\s*\(Moneyline\)', ' (ML)', raw)
            raw = re.sub(r'\s*\((Spread|Total)\)', '', raw)
            return raw.strip(" +").strip()

        allocation_rows.append({
            "rank":          f"P{j}",
            "label":         _clean_parlay_label(par["label"]),
            "game":          " / ".join(l["game"] for l in par.get("legs", [])),
            "game_time":     "",
            "sport":         "Parlay",
            "home_team":     "",
            "away_team":     "",
            "edge_pct":      par["edge_pct"],
            "contracts":     par["num_contracts"],
            "cost":          par["total_cost"],
            "profit_if_win": par["profit_if_win"],
            "pct_of_budget": round(par["total_cost"] / DAILY_BUDGET * 100, 1),
            "confidence":    0,
            "locked":        par.get("locked", False),
            "commence_time": min((l.get("commence_time", "") for l in par.get("legs", [])), default=""),
            "bet_type":      "Parlay",
            "pick":          par["label"],
            "legs_json":     legs_json,
            "contract_price_cents": par.get("contract_price_cents", 0),
            "loss_if_lose":         par.get("total_cost", 0),
            "expected_value":       par.get("expected_value", 0),
            "conf_label":           par.get("confidence", "MEDIUM"),
        })

    # ── Format change warnings for template ──────────────────────────────────
    formatted_warnings = []
    for w in change_warnings:
        wtype = w.get("type", "")
        if wtype == "single_replaced":
            formatted_warnings.append(
                f"⚡ Bet updated: '{w['removed_pick']}' ({w['removed_game']}, "
                f"+{w['removed_edge_pct']}% edge) → '{w['new_pick']}' "
                f"({w['new_game']}, +{w['new_edge_pct']}% edge). "
                f"Reason: {w['reason']}"
            )
        elif wtype == "edge_dropped":
            formatted_warnings.append(
                f"⚠ Edge drop: '{w['pick']}' ({w['game']}) — "
                f"{w['old_edge_pct']}% → {w['new_edge_pct']}%. {w['reason']}"
            )
        elif wtype == "parlay_replaced":
            formatted_warnings.append(
                f"⚡ Parlay updated: '{w['removed_label']}' (+{w['removed_edge_pct']}%) "
                f"→ '{w['new_label']}' (+{w['new_edge_pct']}%). Reason: {w['reason']}"
            )
        elif wtype == "prop_replaced":
            formatted_warnings.append(
                f"⚡ Prop updated: {w['removed']} → {w['new']}. Reason: {w['reason']}"
            )

    # ── Yesterday's recap (settled bets post-mortem) ─────────────────────────
    try:
        from src.report.recap import build_recap
        recap = build_recap(run_date - timedelta(days=1))
    except Exception:
        recap = {"has_recap": False}

    # ── Performance summary (from settled history) ───────────────────────────
    # Each block is independent so a chart-data failure never silences the
    # prop accuracy table (and vice-versa).
    try:
        from src.data.outcome_checker import load_performance_summary
        performance = load_performance_summary()
    except Exception:
        performance = {}

    try:
        from src.data.outcome_checker import load_watchlist_performance
        watchlist_performance = load_watchlist_performance()
    except Exception:
        watchlist_performance = {
            "NHL":  {"won": 0, "lost": 0, "total": 0, "win_rate_pct": None},
            "IPL":  {"won": 0, "lost": 0, "total": 0, "win_rate_pct": None},
            "WNBA": {"won": 0, "lost": 0, "total": 0, "win_rate_pct": None},
            "MLS":  {"won": 0, "lost": 0, "total": 0, "win_rate_pct": None},
            "WC":   {"won": 0, "lost": 0, "total": 0, "win_rate_pct": None},
        }

    try:
        from src.data.outcome_checker import build_chart_data
        chart_data = build_chart_data()
    except Exception:
        chart_data = {"has_data": False}

    try:
        from src.data.outcome_checker import load_prop_accuracy
        prop_accuracy = load_prop_accuracy()
    except Exception:
        prop_accuracy = {}

    try:
        from src.data.outcome_checker import build_prop_chart_data
        prop_chart_data = build_prop_chart_data()
    except Exception:
        prop_chart_data = {"has_data": False, "hist_records_json": "[]"}

    # Build a lookup so prop cards can show the most recent settled result
    # for the same player+prop_type: {(player, prop_type): record}
    # Normalize field names: raw records use hit/actual_stat/model_line;
    # the template expects result/actual/projected.
    prop_last_result: Dict = {}
    for rec in prop_accuracy.get("recent", []):
        key = (rec.get("player", ""), rec.get("prop_type", ""))
        prop_last_result[key] = {
            "result":    "HIT" if rec.get("hit") else "MISS",
            "actual":    rec.get("actual_stat", ""),
            "projected": rec.get("model_line", ""),
            "date":      rec.get("date", ""),
        }   # later records overwrite earlier → most recent wins

    # ── Prop accuracy split by sport ─────────────────────────────────────────
    # load_prop_accuracy() computes by_sport across ALL settled records.
    # We previously iterated only over "recent" (last 20), which caused the
    # by-league counts to be much lower than the true totals.
    prop_accuracy_by_sport: Dict = prop_accuracy.get("by_sport", {})

    return {
        "generated_at":       _now_pacific_str(),
        "run_date":           run_date.strftime("%A, %B %d, %Y"),
        "daily_budget":       DAILY_BUDGET,
        "nba_game_count":     nba_game_count,
        "mlb_game_count":     mlb_game_count,
        "nba_singles":        nba_singles,
        "mlb_singles":        mlb_singles,
        "nfl_singles":        nfl_singles,
        "nhl_singles":        nhl_singles,
        "all_singles":        all_singles,
        "parlays":            parlays,
        "props":              props,
        "allocation":         allocation_rows,
        "total_allocated":    round(grand_total, 2),
        "singles_allocated":  round(total_allocated, 2),
        "parlays_allocated":  round(parlay_allocated, 2),
        "reserve":            round(reserve, 2),
        "errors":             errors,
        "change_warnings":    formatted_warnings,
        "nfl_game_count":     nfl_game_count,
        "nhl_game_count":     nhl_game_count,
        "ipl_game_count":     ipl_game_count,
        "ipl_games_analyzed": max(ipl_games_analyzed, ipl_game_count),
        "has_nba":            nba_game_count > 0,
        "has_mlb":            mlb_game_count > 0,
        "has_nfl":            nfl_game_count > 0,
        "has_nhl":            nhl_game_count > 0,
        "ipl_watchlist":      ipl_watchlist,
        "has_ipl":            ipl_game_count > 0 or ipl_games_analyzed > 0,
        "wnba_watchlist":     wnba_watchlist,
        "wnba_game_count":    wnba_game_count,
        "has_wnba":           wnba_game_count > 0,
        "mls_watchlist":      mls_watchlist,
        "mls_game_count":     mls_game_count,
        "has_mls":            mls_game_count > 0,
        "wc_watchlist":       wc_watchlist,
        "wc_game_count":      wc_game_count,
        "has_wc":             wc_game_count > 0,
        "has_bets":           len(all_singles) > 0 or len(parlays) > 0,
        "performance":        performance,
        "has_performance":    bool(performance.get("total", 0)),
        "chart_data":         chart_data,
        "history_records":    performance.get("all_records", []),
        "run_date_iso":       run_date.isoformat(),
        "fresh_odds":         fresh_odds,
        "prop_accuracy":      prop_accuracy,
        "has_prop_accuracy":  bool(prop_accuracy.get("total", 0)),
        "prop_chart_data":    prop_chart_data,
        "prop_last_result":   prop_last_result,
        "recap":              recap,
        "odds_api_credits":   odds_api_credits or {},
        "nba_top_singles":         nba_top_singles,
        "mlb_top_singles":         mlb_top_singles,
        "nfl_top_singles":         nfl_top_singles,
        "nhl_top_singles":         nhl_top_singles,
        "nba_props":               nba_props_display,
        "mlb_props":               mlb_props_display,
        "has_nba_props":           len(nba_props_display) > 0,
        "has_mlb_props":           len(mlb_props_display) > 0,
        "has_nba_singles":         len(nba_top_singles) > 0,
        "has_mlb_singles":         len(mlb_top_singles) > 0,
        "has_nfl_singles":         len(nfl_top_singles) > 0,
        "has_nhl_singles":         len(nhl_top_singles) > 0,
        "prop_accuracy_by_sport":  prop_accuracy_by_sport,
        "watchlist_performance":   watchlist_performance,
        # Calibration panel — auto-promotes between phases as shadow log data
        # accumulates. During Phase 0 (cold start) it shows "Building data" for
        # every sport×market; once Phase A kicks in, ratios appear and live
        # picks start being calibration-adjusted at the slot-ranking chokepoint.
        "calibration":             _load_calibration_panel(),
        "cap_state":               _load_cap_panel(),
    }


def _load_cap_panel() -> Dict:
    """
    Load the cap auto-relaxation state for the report panel. Safe-default
    on any error so the panel never breaks the report.
    """
    try:
        from src.state.cap_state import load_panel_data
        return load_panel_data()
    except Exception as e:
        logger.debug(f"Cap panel load failed (non-fatal): {e}")
        return {"has_data": False, "caps": []}
