"""Assembles all analysis results into a structured report dict for templating."""
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List
from src.config import DAILY_BUDGET, MAX_SINGLE_BETS
from src.models.edge_finder import BetRecommendation
from src.models.parlay_builder import ParlayRecommendation
from src.models.props_analyzer import PropPick


def _now_pacific_str() -> str:
    now_utc = datetime.now(timezone.utc)
    offset = -7 if 3 <= now_utc.month <= 10 else -8
    label = "PDT" if offset == -7 else "PST"
    pacific = now_utc + timedelta(hours=offset)
    return pacific.strftime(f"%B %d, %Y at %I:%M %p {label}")


def build_report(
    run_date: date,
    nba_singles: List[BetRecommendation],
    mlb_singles: List[BetRecommendation],
    parlays: List[ParlayRecommendation],
    props: List[PropPick],
    nba_game_count: int,
    mlb_game_count: int,
    errors: List[str],
) -> Dict:
    all_singles = sorted(
        nba_singles + mlb_singles,
        key=lambda r: r.edge,
        reverse=True,
    )[:MAX_SINGLE_BETS]

    total_allocated = sum(r.sizing.total_cost for r in all_singles)
    parlay_allocated = sum(p.sizing.total_cost for p in parlays)
    grand_total = total_allocated + parlay_allocated
    reserve = max(0.0, DAILY_BUDGET - grand_total)

    allocation_rows = []
    for i, rec in enumerate(all_singles, 1):
        allocation_rows.append({
            "rank": i,
            "label": f"{rec.pick} ({rec.bet_type})",
            "game": rec.game,
            "game_time": rec.game_time,
            "sport": rec.sport,
            "contracts": rec.sizing.num_contracts,
            "cost": rec.sizing.total_cost,
            "profit_if_win": rec.sizing.profit_if_win,
            "pct_of_budget": round(rec.sizing.total_cost / DAILY_BUDGET * 100, 1),
            "confidence": rec.sizing.kelly_fraction,
        })

    for j, par in enumerate(parlays, 1):
        allocation_rows.append({
            "rank": f"P{j}",
            "label": par.label,
            "game": " / ".join(par.game_labels),
            "game_time": "",
            "sport": "Parlay",
            "contracts": par.sizing.num_contracts,
            "cost": par.sizing.total_cost,
            "profit_if_win": par.sizing.profit_if_win,
            "pct_of_budget": round(par.sizing.total_cost / DAILY_BUDGET * 100, 1),
            "confidence": par.sizing.kelly_fraction,
        })

    return {
        "generated_at": _now_pacific_str(),
        "run_date": run_date.strftime("%A, %B %d, %Y"),
        "daily_budget": DAILY_BUDGET,
        "nba_game_count": nba_game_count,
        "mlb_game_count": mlb_game_count,
        "nba_singles": [_rec_to_dict(r) for r in nba_singles if r in all_singles],
        "mlb_singles": [_rec_to_dict(r) for r in mlb_singles if r in all_singles],
        "all_singles": [_rec_to_dict(r) for r in all_singles],
        "parlays": [_parlay_to_dict(p) for p in parlays],
        "props": [_prop_to_dict(p) for p in props],
        "allocation": allocation_rows,
        "total_allocated": round(grand_total, 2),
        "singles_allocated": round(total_allocated, 2),
        "parlays_allocated": round(parlay_allocated, 2),
        "reserve": round(reserve, 2),
        "errors": errors,
        "has_nba": nba_game_count > 0,
        "has_mlb": mlb_game_count > 0,
        "has_bets": len(all_singles) > 0 or len(parlays) > 0,
    }


def _rec_to_dict(r: BetRecommendation) -> Dict:
    return {
        "sport": r.sport,
        "game": r.game,
        "game_time": r.game_time,
        "bet_type": r.bet_type,
        "pick": r.pick,
        "market_prob_pct": round(r.market_prob * 100, 1),
        "model_prob_pct": round(r.model_prob * 100, 1),
        "edge_pct": round(r.edge * 100, 1),
        "contract_price": round(r.contract_price, 4),
        "contract_price_cents": round(r.contract_price * 100, 1),
        "num_contracts": r.sizing.num_contracts,
        "total_cost": r.sizing.total_cost,
        "profit_if_win": r.sizing.profit_if_win,
        "loss_if_lose": r.sizing.loss_if_lose,
        "expected_value": r.sizing.expected_value,
        "confidence": r.confidence,
        "signals": r.signals,
        "research": r.research,
    }


def _parlay_to_dict(p: ParlayRecommendation) -> Dict:
    return {
        "label": p.label,
        "legs": [
            {
                "game": l.game,
                "pick": l.pick,
                "bet_type": l.bet_type,
                "model_prob_pct": round(l.model_prob * 100, 1),
                "edge_pct": round(l.edge * 100, 1),
            }
            for l in p.legs
        ],
        "combined_prob_pct": round(p.combined_prob * 100, 1),
        "contract_price_cents": round(p.contract_price * 100, 1),
        "edge_pct": round(p.edge * 100, 1),
        "num_contracts": p.sizing.num_contracts,
        "total_cost": p.sizing.total_cost,
        "profit_if_win": p.sizing.profit_if_win,
        "expected_value": p.expected_value,
        "confidence": p.confidence,
    }


def _prop_to_dict(p: PropPick) -> Dict:
    return {
        "sport": p.sport,
        "player": p.player,
        "team": p.team,
        "opponent": p.opponent,
        "prop_type": p.prop_type,
        "model_line": p.model_line,
        "confidence": p.confidence,
        "note": p.note,
        "signals": p.signals,
    }
