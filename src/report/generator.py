"""Assembles pre-serialised pick dicts into the report context for Jinja templating."""
import json
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Optional
from src.config import DAILY_BUDGET, MAX_SINGLE_BETS


def _now_pacific_str() -> str:
    now_utc = datetime.now(timezone.utc)
    offset = -7 if 3 <= now_utc.month <= 10 else -8
    label = "PDT" if offset == -7 else "PST"
    pacific = now_utc + timedelta(hours=offset)
    return pacific.strftime(f"%B %d, %Y at %I:%M %p {label}")


def build_report(
    run_date: date,
    singles: List[Dict],          # all sports, already serialised by state manager
    parlays: List[Dict],
    props: List[Dict],
    nba_game_count: int,
    mlb_game_count: int,
    errors: List[str],
    change_warnings: Optional[List[Dict]] = None,
    odds_api_credits: Optional[Dict] = None,
    nfl_game_count: int = 0,
    nhl_game_count: int = 0,
) -> Dict:
    change_warnings = change_warnings or []

    # NHL is monitoring-only — exclude from budget allocation pool entirely.
    # All other active sports (NBA, MLB, NFL) compete for the top-5 slots.
    nhl_watchlist = sorted(
        [s for s in singles if s.get("sport") == "NHL"],
        key=lambda r: (0 if r["confidence"] == "HIGH" else 1, -r["edge"]),
    )

    # Top picks: HIGH confidence first, then by edge within each tier.
    # Deduplicate by (sport, home_team, away_team, bet_type) — a line move creates
    # a new pick label but it's the same bet; keep the first (highest-edge) occurrence.
    _seen_game_bets: set = set()
    _deduped_pool = []
    for s in sorted(
        [s for s in singles if s.get("sport") != "NHL"],
        key=lambda r: (0 if r["confidence"] == "HIGH" else 1, -r["edge"]),
    ):
        _key = (s.get("sport", ""), s.get("home_team", ""), s.get("away_team", ""), s.get("bet_type", ""))
        if _key not in _seen_game_bets:
            _seen_game_bets.add(_key)
            _deduped_pool.append(s)
    all_singles = _deduped_pool[:MAX_SINGLE_BETS]
    nba_singles = [s for s in all_singles if s["sport"] == "NBA"]
    mlb_singles = [s for s in all_singles if s["sport"] == "MLB"]
    nfl_singles = [s for s in all_singles if s["sport"] == "NFL"]
    nhl_singles = [s for s in all_singles if s["sport"] == "NHL"]  # always empty (filtered above)

    # ── Per-league top-5 singles for display in league sections ─────────────
    def _top_singles_for_sport(sport: str) -> List[Dict]:
        """Sort by edge, deduplicate by game+bet_type, cap at MAX_SINGLE_BETS."""
        seen: set = set()
        out = []
        for s in sorted(
            [s for s in singles if s.get("sport") == sport],
            key=lambda r: (0 if r["confidence"] == "HIGH" else 1, -r["edge"]),
        ):
            k = (s.get("home_team", ""), s.get("away_team", ""), s.get("bet_type", ""))
            if k not in seen:
                seen.add(k)
                out.append(s)
            if len(out) == MAX_SINGLE_BETS:
                break
        return out

    nba_top_singles_raw = _top_singles_for_sport("NBA")
    mlb_top_singles_raw = _top_singles_for_sport("MLB")
    nfl_top_singles_raw = _top_singles_for_sport("NFL")

    # Tag which singles are in the budget allocation pool (rank 1-5)
    _alloc_rank_map = {(r["pick"], r["game"]): i for i, r in enumerate(all_singles, 1)}

    def _tag_alloc(lst):
        out = []
        for rec in lst:
            out.append({**rec, "alloc_rank": _alloc_rank_map.get((rec["pick"], rec["game"]))})
        return out

    nba_top_singles = _tag_alloc(nba_top_singles_raw)
    mlb_top_singles = _tag_alloc(mlb_top_singles_raw)
    nfl_top_singles = _tag_alloc(nfl_top_singles_raw)
    # nhl_watchlist: never in budget, no alloc_rank needed

    # ── Per-league top-6 props for display ───────────────────────────────────
    from src.config import MAX_PROPS_PER_SPORT
    nba_props_display = sorted(
        [p for p in props if p.get("sport") == "NBA"],
        key=lambda p: -p.get("edge_pct", 0),
    )[:MAX_PROPS_PER_SPORT]

    mlb_props_display = sorted(
        [p for p in props if p.get("sport") == "MLB"],
        key=lambda p: -p.get("edge_pct", 0),
    )[:MAX_PROPS_PER_SPORT]

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
            "commence_time": "",
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
    prop_last_result: Dict = {}
    for rec in prop_accuracy.get("recent", []):
        key = (rec.get("player", ""), rec.get("prop_type", ""))
        prop_last_result[key] = rec   # later records overwrite earlier → most recent wins

    # ── Prop accuracy split by sport ─────────────────────────────────────────
    _pa_by_sport: Dict = {}
    for _rec in prop_accuracy.get("recent", []):
        _sp = _rec.get("sport", "")
        if not _sp:
            continue
        if _sp not in _pa_by_sport:
            _pa_by_sport[_sp] = {"hits": 0, "misses": 0, "total": 0, "sum_err": 0.0}
        _pa_by_sport[_sp]["total"] += 1
        if _rec.get("hit"):
            _pa_by_sport[_sp]["hits"] += 1
        else:
            _pa_by_sport[_sp]["misses"] += 1
        _err = float(_rec.get("actual_stat", 0) or 0) - float(_rec.get("model_line", 0) or 0)
        _pa_by_sport[_sp]["sum_err"] += _err

    prop_accuracy_by_sport: Dict = {}
    for _sp, _d in _pa_by_sport.items():
        _t = _d["total"]
        prop_accuracy_by_sport[_sp] = {
            "total":    _t,
            "hits":     _d["hits"],
            "misses":   _d["misses"],
            "hit_rate": round(_d["hits"] / _t * 100, 1) if _t else None,
            "avg_err":  round(_d["sum_err"] / _t, 2) if _t else None,
        }

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
        "nhl_watchlist":      nhl_watchlist,
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
        "has_nba":            nba_game_count > 0,
        "has_mlb":            mlb_game_count > 0,
        "has_nfl":            nfl_game_count > 0,
        "has_nhl":            nhl_game_count > 0,
        "has_bets":           len(all_singles) > 0 or len(parlays) > 0,
        "performance":        performance,
        "has_performance":    bool(performance.get("total", 0)),
        "chart_data":         chart_data,
        "history_records":    performance.get("all_records", []),
        "run_date_iso":       run_date.isoformat(),
        "prop_accuracy":      prop_accuracy,
        "has_prop_accuracy":  bool(prop_accuracy.get("total", 0)),
        "prop_chart_data":    prop_chart_data,
        "prop_last_result":   prop_last_result,
        "recap":              recap,
        "odds_api_credits":   odds_api_credits or {},
        "nba_top_singles":         nba_top_singles,
        "mlb_top_singles":         mlb_top_singles,
        "nfl_top_singles":         nfl_top_singles,
        "nba_props":               nba_props_display,
        "mlb_props":               mlb_props_display,
        "has_nba_singles":         len(nba_top_singles) > 0,
        "has_mlb_singles":         len(mlb_top_singles) > 0,
        "has_nfl_singles":         len(nfl_top_singles) > 0,
        "prop_accuracy_by_sport":  prop_accuracy_by_sport,
    }
