"""
Daily picks state manager.

State file: state/picks_YYYY-MM-DD.json  (committed to repo by GitHub Actions)

First run of the day  → saves all picks as the locked baseline.
Subsequent runs       →
  • Games already started  → bets are frozen (shown as locked, cannot be replaced).
  • Pre-game bets          → replaced only when a new bet not in the morning picks
                             has a higher edge than an existing pre-game pick.
  • Warnings generated for every substitution with reason.
"""
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import MAX_SINGLE_BETS

logger = logging.getLogger(__name__)

STATE_DIR = Path("state")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _state_path(run_date: date) -> Path:
    return STATE_DIR / f"picks_{run_date.isoformat()}.json"


def load_state(run_date: date) -> Optional[Dict]:
    """Return today's state dict, or None if this is the first run."""
    path = _state_path(run_date)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"State loaded from {path} (first run at {data.get('first_run_at', '?')})")
        return data
    except Exception as e:
        logger.error(f"Failed to load state {path}: {e}")
        return None


def save_state(run_date: date, state: Dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    path = _state_path(run_date)
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved → {path}")
    except Exception as e:
        logger.error(f"Failed to save state {path}: {e}")


# ---------------------------------------------------------------------------
# Serialisers  (BetRecommendation / ParlayRecommendation / PropPick → dict)
# All output dicts are template-ready AND state-storable.
# ---------------------------------------------------------------------------

def bet_to_dict(rec) -> Dict:
    """Serialize a BetRecommendation to a flat template-ready dict."""
    return {
        # Template display fields
        "sport": rec.sport,
        "game": rec.game,
        "game_time": rec.game_time,
        "bet_type": rec.bet_type,
        "pick": rec.pick,
        "market_prob_pct": round(rec.market_prob * 100, 1),
        "model_prob_pct": round(rec.model_prob * 100, 1),
        "edge_pct": round(rec.edge * 100, 1),
        "contract_price": round(rec.contract_price, 4),
        "contract_price_cents": round(rec.contract_price * 100, 1),
        "num_contracts": rec.sizing.num_contracts,
        "total_cost": rec.sizing.total_cost,
        "profit_if_win": rec.sizing.profit_if_win,
        "loss_if_lose": rec.sizing.loss_if_lose,
        "expected_value": rec.sizing.expected_value,
        "confidence": rec.confidence,
        "signals": rec.signals,
        "research": rec.research,
        # State management fields
        "home_team": rec.home_team,
        "away_team": rec.away_team,
        "commence_time": getattr(rec, "commence_time", ""),
        "edge": rec.edge,        # raw float for sorting / comparison
        "locked": False,
    }


def parlay_to_dict(par) -> Dict:
    return {
        "label": par.label,
        "legs": [
            {
                "game": l.game,
                "pick": l.pick,
                "bet_type": l.bet_type,
                "model_prob_pct": round(l.model_prob * 100, 1),
                "edge_pct": round(l.edge * 100, 1),
                "home_team": l.home_team,
                "away_team": l.away_team,
                "commence_time": getattr(l, "commence_time", ""),
            }
            for l in par.legs
        ],
        "combined_prob_pct": round(par.combined_prob * 100, 1),
        "contract_price_cents": round(par.contract_price * 100, 1),
        "edge_pct": round(par.edge * 100, 1),
        "num_contracts": par.sizing.num_contracts,
        "total_cost": par.sizing.total_cost,
        "profit_if_win": par.sizing.profit_if_win,
        "expected_value": par.expected_value,
        "confidence": par.confidence,
        # State management
        "edge": par.edge,
        "locked": False,
    }


def prop_to_dict(prop) -> Dict:
    return {
        "sport": prop.sport,
        "player": prop.player,
        "team": prop.team,
        "opponent": prop.opponent,
        "prop_type": prop.prop_type,
        "model_line": prop.model_line,
        "confidence": prop.confidence,
        "note": prop.note,
        "signals": prop.signals,
        "research": prop.research,
        # State management
        "commence_time": getattr(prop, "commence_time", ""),
        "locked": False,
    }


# ---------------------------------------------------------------------------
# Game status helpers
# ---------------------------------------------------------------------------

def _game_started(commence_time: str) -> bool:
    """True if the game's scheduled start time is now in the past."""
    if not commence_time:
        return False
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= dt
    except Exception:
        return False


def _update_lock_flags(picks: List[Dict], use_any_leg: bool = False) -> None:
    """Mutates picks in-place: set locked=True for any started game."""
    for p in picks:
        if use_any_leg:
            # Parlay: lock if ANY leg has started
            started = any(_game_started(l.get("commence_time", "")) for l in p.get("legs", []))
        else:
            started = _game_started(p.get("commence_time", ""))
        if started:
            p["locked"] = True


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _single_key(d: Dict) -> str:
    return f"{d['pick']}|{d['game']}"


def _parlay_label(d) -> str:
    """Works on both dict and ParlayRecommendation."""
    return d["label"] if isinstance(d, dict) else d.label


def _prop_key(d) -> str:
    player = d["player"] if isinstance(d, dict) else d.player
    prop_type = d["prop_type"] if isinstance(d, dict) else d.prop_type
    return f"{player}|{prop_type}"


_CONF_RANK = {"HIGH": 2, "MEDIUM": 1}


def merge_picks(
    state: Dict,
    new_singles: List[Dict],
    new_parlays: List[Dict],
    new_props: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Merge the locked morning state with fresh analysis results.

    Returns (final_singles, final_parlays, final_props, warnings).

    Rules:
    ▸ Game started     → bet is fully locked, shown as-is, never replaced.
    ▸ Pre-game bet     → replaced only if a *new* bet (not in morning picks)
                         has a higher edge (+0.5 % threshold to avoid noise).
    ▸ Warnings list    → one entry per substitution with reason.
    """
    warnings: List[Dict] = []

    # ── Singles ──────────────────────────────────────────────────────────────
    locked_singles: List[Dict] = state.get("singles", [])
    _update_lock_flags(locked_singles)

    started  = [p for p in locked_singles if p.get("locked")]
    pregame  = [p for p in locked_singles if not p.get("locked")]

    locked_keys = {_single_key(p) for p in locked_singles}
    new_edge_map = {_single_key(d): d["edge"] for d in new_singles}

    # New bets not already in the locked morning set
    truly_new = [d for d in new_singles if _single_key(d) not in locked_keys]
    truly_new.sort(key=lambda d: d["edge"], reverse=True)

    final_singles: List[Dict] = list(started)
    used_new: set = set()

    # Walk pre-game picks weakest-first so weakest gets replaced first
    for pick in sorted(pregame, key=lambda p: p["edge"]):
        current_edge = new_edge_map.get(_single_key(pick))  # edge in today's market
        edge_dropped  = (
            current_edge is not None
            and (pick["edge"] - current_edge) >= 0.02   # 2 pp drop
        )

        # Best available new bet that beats this pick
        best_new = next(
            (d for d in truly_new
             if id(d) not in used_new and d["edge"] > pick["edge"] + 0.005),
            None,
        )

        if best_new:
            reason_parts = []
            if edge_dropped:
                reason_parts.append(
                    f"edge on this bet dropped from {pick['edge']*100:.1f}% "
                    f"to {current_edge*100:.1f}%"
                )
            reason_parts.append(
                f"new bet '{best_new['pick']}' ({best_new['game']}) "
                f"has {best_new['edge']*100:.1f}% edge"
            )
            warnings.append({
                "type": "single_replaced",
                "removed_pick": pick["pick"],
                "removed_game": pick["game"],
                "removed_edge_pct": round(pick["edge"] * 100, 1),
                "new_pick": best_new["pick"],
                "new_game": best_new["game"],
                "new_edge_pct": round(best_new["edge"] * 100, 1),
                "reason": "; ".join(reason_parts),
            })
            used_new.add(id(best_new))
            final_singles.append(best_new)
        else:
            # Keep pick; update edge if market moved (display only)
            if current_edge is not None:
                pick = {**pick, "edge": current_edge, "edge_pct": round(current_edge * 100, 1)}
                if edge_dropped:
                    warnings.append({
                        "type": "edge_dropped",
                        "pick": pick["pick"],
                        "game": pick["game"],
                        "old_edge_pct": round(state_edge := next(
                            (s["edge"] for s in locked_singles if _single_key(s) == _single_key(pick)),
                            pick["edge"],
                        ) * 100, 1),
                        "new_edge_pct": round(current_edge * 100, 1),
                        "reason": "Market moved against this bet — still no better alternative found",
                    })
            final_singles.append(pick)

    # ── Parlays ───────────────────────────────────────────────────────────────
    locked_parlays: List[Dict] = state.get("parlays", [])
    _update_lock_flags(locked_parlays, use_any_leg=True)

    started_p = [p for p in locked_parlays if p.get("locked")]
    pregame_p = [p for p in locked_parlays if not p.get("locked")]
    locked_par_labels = {p["label"] for p in locked_parlays}

    truly_new_par = [d for d in new_parlays if _parlay_label(d) not in locked_par_labels]
    truly_new_par.sort(key=lambda d: d["edge"] if isinstance(d, dict) else d.edge, reverse=True)

    final_parlays: List[Dict] = list(started_p)
    used_new_par: set = set()

    for par in sorted(pregame_p, key=lambda p: p["edge"]):
        best_new = next(
            (d for d in truly_new_par
             if id(d) not in used_new_par
             and (d["edge"] if isinstance(d, dict) else d.edge) > par["edge"] + 0.005),
            None,
        )
        if best_new:
            new_d = best_new if isinstance(best_new, dict) else parlay_to_dict(best_new)
            warnings.append({
                "type": "parlay_replaced",
                "removed_label": par["label"],
                "removed_edge_pct": round(par["edge"] * 100, 1),
                "new_label": new_d["label"],
                "new_edge_pct": round(new_d["edge"] * 100, 1),
                "reason": f"new parlay has {new_d['edge']*100:.1f}% edge vs {par['edge']*100:.1f}%",
            })
            used_new_par.add(id(best_new))
            final_parlays.append(new_d)
        else:
            final_parlays.append(par)

    # ── Props ─────────────────────────────────────────────────────────────────
    locked_props: List[Dict] = state.get("props", [])
    _update_lock_flags(locked_props)

    started_pr = [p for p in locked_props if p.get("locked")]
    pregame_pr = [p for p in locked_props if not p.get("locked")]
    locked_prop_keys = {_prop_key(p) for p in locked_props}

    truly_new_props = [d for d in new_props if _prop_key(d) not in locked_prop_keys]
    truly_new_props.sort(
        key=lambda d: _CONF_RANK.get(d["confidence"] if isinstance(d, dict) else d.confidence, 0),
        reverse=True,
    )

    final_props: List[Dict] = list(started_pr)
    used_new_props: set = set()

    for prop in pregame_pr:
        prop_conf = _CONF_RANK.get(prop.get("confidence", "MEDIUM"), 1)
        best_new = next(
            (d for d in truly_new_props
             if id(d) not in used_new_props
             and _CONF_RANK.get(d["confidence"] if isinstance(d, dict) else d.confidence, 0) > prop_conf),
            None,
        )
        if best_new:
            new_d = best_new if isinstance(best_new, dict) else prop_to_dict(best_new)
            warnings.append({
                "type": "prop_replaced",
                "removed": f"{prop['player']} — {prop['prop_type']}",
                "new": f"{new_d['player']} — {new_d['prop_type']}",
                "reason": "Higher-confidence prop identified",
            })
            used_new_props.add(id(best_new))
            final_props.append(new_d)
        else:
            final_props.append(prop)

    if warnings:
        logger.info(f"State merge: {len(warnings)} substitution(s) / warning(s)")

    return final_singles, final_parlays, final_props, warnings
