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
) -> Dict:
    change_warnings = change_warnings or []

    # Top picks: HIGH confidence first, then by edge within each tier
    all_singles = sorted(
        singles,
        key=lambda r: (0 if r["confidence"] == "HIGH" else 1, -r["edge"]),
    )[:MAX_SINGLE_BETS]
    nba_singles = [s for s in all_singles if s["sport"] == "NBA"]
    mlb_singles = [s for s in all_singles if s["sport"] == "MLB"]

    total_allocated  = sum(r["total_cost"] for r in all_singles)
    parlay_allocated = sum(p["total_cost"] for p in parlays)
    grand_total      = total_allocated + parlay_allocated
    reserve          = max(0.0, DAILY_BUDGET - grand_total)

    # ── Allocation table rows ────────────────────────────────────────────────
    allocation_rows = []
    for i, rec in enumerate(all_singles, 1):
        allocation_rows.append({
            "rank":          i,
            "label":         f"{rec['pick']} ({rec['bet_type']})",
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
        })

    for j, par in enumerate(parlays, 1):
        # Build a compact legs JSON for live win-prob in the allocation table
        legs_json = json.dumps([{
            "home_team":     l.get("home_team", ""),
            "away_team":     l.get("away_team", ""),
            "bet_type":      l.get("bet_type", ""),
            "pick":          l.get("pick", ""),
            "sport":         l.get("sport", ""),
            "model_prob_pct": l.get("model_prob_pct", 50),
        } for l in par.get("legs", [])])

        # Parlay is locked if any leg's game has started
        allocation_rows.append({
            "rank":          f"P{j}",
            "label":         par["label"],
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
    try:
        from src.data.outcome_checker import load_performance_summary, build_chart_data, load_prop_accuracy
        performance   = load_performance_summary()
        chart_data    = build_chart_data()
        prop_accuracy = load_prop_accuracy()
    except Exception:
        performance   = {}
        chart_data    = {"has_data": False}
        prop_accuracy = {}

    # Build a lookup so prop cards can show the most recent settled result
    # for the same player+prop_type: {(player, prop_type): record}
    prop_last_result: Dict = {}
    for rec in prop_accuracy.get("recent", []):
        key = (rec.get("player", ""), rec.get("prop_type", ""))
        prop_last_result[key] = rec   # later records overwrite earlier → most recent wins

    return {
        "generated_at":       _now_pacific_str(),
        "run_date":           run_date.strftime("%A, %B %d, %Y"),
        "daily_budget":       DAILY_BUDGET,
        "nba_game_count":     nba_game_count,
        "mlb_game_count":     mlb_game_count,
        "nba_singles":        nba_singles,
        "mlb_singles":        mlb_singles,
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
        "has_nba":            nba_game_count > 0,
        "has_mlb":            mlb_game_count > 0,
        "has_bets":           len(all_singles) > 0 or len(parlays) > 0,
        "performance":        performance,
        "has_performance":    bool(performance.get("total", 0)),
        "chart_data":         chart_data,
        "history_records":    performance.get("all_records", []),
        "run_date_iso":       run_date.isoformat(),
        "prop_accuracy":      prop_accuracy,
        "has_prop_accuracy":  bool(prop_accuracy.get("total", 0)),
        "prop_last_result":   prop_last_result,
        "recap":              recap,
        "odds_api_credits":   odds_api_credits or {},
    }
