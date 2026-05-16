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
from collections import Counter
from src.report.card_context import build_card_context, build_prop_context
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import MAX_SINGLE_BETS, MAX_PARLAYS, MAX_PROPS_PER_SPORT, LINE_MOVE_THRESHOLD

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
        "signals": rec.signals,    # kept raw for state management / merge logic
        "research": rec.research,  # kept raw for state management / merge logic
        # Display context — narrative + merged deduplicated stat list.
        # Computed here so confidence (set upstream) is never affected.
        **dict(zip(
            ("narrative", "context"),
            build_card_context(
                rec.sport, rec.pick, rec.bet_type,
                rec.signals, rec.research,
                rec.model_prob, rec.market_prob, rec.edge,
            ),
        )),
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
                "sport": l.sport,
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
        "model_margin": getattr(prop, "model_margin", 0.0),
        "confidence": prop.confidence,
        "note": prop.note,
        "signals": prop.signals,
        "research": prop.research,
        **dict(zip(
            ("narrative", "context"),
            build_prop_context(
                prop.sport, prop.prop_type, prop.player, prop.team, prop.opponent,
                prop.signals, prop.research,
                prop.model_line,
                getattr(prop, "market_line", prop.model_line),
                getattr(prop, "edge", 0.0),
            ),
        )),
        # Odds API market fields
        "market_line":  getattr(prop, "market_line", prop.model_line),
        "market_prob":  getattr(prop, "market_prob", 0.0),
        "model_prob":   getattr(prop, "model_prob", 0.0),
        "edge":         getattr(prop, "edge", 0.0),
        "edge_pct":     round(getattr(prop, "edge", 0.0) * 100, 1),
        "odds_display": getattr(prop, "odds_display", ""),
        "book":         getattr(prop, "book", ""),
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


def _game_team_key(d: Dict) -> Optional[tuple]:
    """Returns (game, team) for ML/Spread bets, None for Totals.
    Used to detect line moves: Spurs -11.5 and Spurs -10.5 share the same key."""
    bt = d.get("bet_type", "")
    if bt not in ("Moneyline", "Spread"):
        return None
    home = d.get("home_team", "")
    away = d.get("away_team", "")
    pick = d.get("pick", "")
    team = next((t for t in [home, away] if pick.startswith(t)), pick.split()[0] if pick else pick)
    return (d.get("game", ""), team)


def _signal_refresh_key(d: Dict) -> str:
    """Looser key for signal/display refresh.
    - Totals:  ignores exact line number (Over 214 vs Over 214.5)
    - Spreads: ignores exact line number (Spurs -11.5 vs Spurs -10.5) — uses team
    - ML:      exact pick match"""
    game = d.get("game", "")
    bt   = d.get("bet_type", "")
    pick = d.get("pick", "")
    if bt == "Total":
        direction = "Over" if pick.startswith("Over") else "Under"
        return f"{game}|{bt}|{direction}"
    if bt == "Spread":
        home = d.get("home_team", "")
        away = d.get("away_team", "")
        team = next((t for t in [home, away] if pick.startswith(t)), pick.split()[0] if pick else pick)
        return f"{game}|{bt}|{team}"
    return f"{game}|{bt}|{pick}"


def _parlay_label(d) -> str:
    """Works on both dict and ParlayRecommendation."""
    return d["label"] if isinstance(d, dict) else d.label


def _prop_key(d) -> str:
    player = d["player"] if isinstance(d, dict) else d.player
    prop_type = d["prop_type"] if isinstance(d, dict) else d.prop_type
    return f"{player}|{prop_type}"


def _dedup_cached_singles(singles: List[Dict]) -> List[Dict]:
    """
    Remove duplicates where the same team appears in both ML and Spread slots
    for the same game.  Priority (highest wins):
      1. Locked over pre-game (never evict a bet the game has already started on)
      2. Moneyline over Spread (purer directional bet; avoids correlated slots)
      3. Higher edge
    Totals are exempt — they have no single team to key on.
    """
    _BET_RANK = {"Moneyline": 0, "Total": 1, "Spread": 2}

    ordered = sorted(
        singles,
        key=lambda s: (
            0 if s.get("locked") else 1,          # locked first
            _BET_RANK.get(s.get("bet_type", ""), 9),  # ML before Spread
            -s.get("edge", 0),                    # higher edge first
        ),
    )
    seen: set = set()
    result: List[Dict] = []
    for s in ordered:
        bt = s.get("bet_type", "")
        if bt in ("Moneyline", "Spread"):
            home = s.get("home_team", "")
            away = s.get("away_team", "")
            pick = s.get("pick", "")
            team = next((t for t in [home, away] if pick.startswith(t)), pick)
            key = (s.get("game", ""), team)
            if key in seen:
                continue
            seen.add(key)
        elif bt == "Total":
            # Dedup same-game same-direction Totals (e.g. Under 214.5 and Under 216.5
            # for the same game are the same underlying bet — keep the higher-edge one,
            # which is already first due to the sort above).
            pick = s.get("pick", "")
            direction = "over" if pick.lower().startswith("over") else "under"
            key = (s.get("game", ""), "total_" + direction)
            if key in seen:
                continue
            seen.add(key)
        result.append(s)
    # Restore display order: locked first, then by edge descending
    result.sort(key=lambda s: (0 if s.get("locked") else 1, -s.get("edge", 0)))
    return result


def _cached_parlay_valid(d: Dict) -> bool:
    """
    Validate a cached parlay dict against current Robinhood rules.
    Drops it unconditionally if it violates the rules so invalid
    cached parlays never survive a fresh run.

    Rules:
      ML + Spread  → invalid everywhere
      Cross-game   → only ML + ML is valid
      Same-game    → anything except ML + Spread (already caught above)
    """
    legs = d.get("legs", [])
    if len(legs) < 2:
        return False
    types = {leg.get("bet_type", "") for leg in legs}
    if types == {"Moneyline", "Spread"}:
        return False
    games = {leg.get("game", "") for leg in legs}
    if len(games) > 1:  # cross-game
        return all(leg.get("bet_type") == "Moneyline" for leg in legs)
    return True


_CONF_RANK = {"HIGH": 2, "MEDIUM": 1}


def merge_picks(
    state: Dict,
    new_singles: List[Dict],
    new_parlays: List[Dict],
    new_props: List[Dict],
    all_fresh_singles: Optional[List[Dict]] = None,
    allow_replace: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Merge the locked morning state with fresh analysis results.

    Returns (final_singles, final_parlays, final_props, warnings).

    Rules:
    ▸ Game started      → bet is fully locked, shown as-is, never replaced.
    ▸ allow_replace=False (default)
                        → signals, odds, and model probs are refreshed in-place;
                          no pick is ever swapped out.  Use this for visual/code
                          deploys where you don't want the slate to change.
    ▸ allow_replace=True
                        → unlocked picks are replaced if a better option (higher
                          edge, +0.5 % threshold) is available.  Triggered via
                          the "Re-evaluate picks" workflow checkbox.
    ▸ Warnings list     → one entry per substitution with reason.
    """
    warnings: List[Dict] = []

    # Sports that are watchlist-only and must never appear in the budget pool or parlays.
    # Picks saved from a buggy run where these entered the budget pool are silently dropped here.
    _WATCHLIST_SPORTS = {"NHL", "IPL", "WNBA", "MLS"}

    # ── Singles ──────────────────────────────────────────────────────────────
    locked_singles: List[Dict] = _dedup_cached_singles([
        s for s in state.get("singles", [])
        if s.get("sport", "") not in _WATCHLIST_SPORTS
    ])
    _update_lock_flags(locked_singles)

    started  = [p for p in locked_singles if p.get("locked")]
    pregame  = [p for p in locked_singles if not p.get("locked")]

    locked_keys  = {_single_key(p) for p in locked_singles}
    new_edge_map = {_single_key(d): d["edge"] for d in new_singles}
    # Signal refresh map: uses ALL fresh bets (uncapped) so bets that dropped out
    # of the top-5 since morning still get updated signals/research/probs.
    # Looser key ignores exact line value for Totals and Spreads.
    _refresh_pool = all_fresh_singles if all_fresh_singles is not None else new_singles
    new_pick_map  = {_signal_refresh_key(d): d for d in _refresh_pool}

    # Line-move map: same game+team but different line (e.g. Spurs -11.5 → -10.5).
    # These are NOT "truly new" — they update an existing pre-game pick in place.
    pregame_gt_keys = {_game_team_key(p) for p in pregame} - {None}
    line_move_map: Dict[tuple, Dict] = {}
    for d in new_singles:
        if _single_key(d) in locked_keys:
            continue
        gt = _game_team_key(d)
        if gt in pregame_gt_keys:
            # Keep the highest-edge version for this game+team
            if gt not in line_move_map or d["edge"] > line_move_map[gt]["edge"]:
                line_move_map[gt] = d

    # Truly new: different game+team from any pre-game pick (not a line move)
    truly_new = [
        d for d in new_singles
        if _single_key(d) not in locked_keys
        and _game_team_key(d) not in pregame_gt_keys
    ]
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

        # ── Line-move check ──────────────────────────────────────────────────
        # If the same game+team now has a different line, treat it as an
        # in-place update (not a replacement) so we never show two spread bets
        # for the same team in the same game (e.g. Spurs -11.5 AND Spurs -10.5).
        gt = _game_team_key(pick)
        line_moved_fresh = line_move_map.get(gt) if gt else None
        if line_moved_fresh and _single_key(line_moved_fresh) != _single_key(pick):
            # Line moved — update pick in place to reflect the current line
            old_line = pick["pick"]
            morning_mkt = pick.get("market_prob", pick.get("contract_price", 0))
            current_mkt = line_moved_fresh.get("market_prob", morning_mkt)
            line_mv     = current_mkt - morning_mkt
            fresh_signals = list(line_moved_fresh["signals"])
            if abs(line_mv) >= LINE_MOVE_THRESHOLD:
                tag = (f"📈 Line moved toward pick (+{line_mv*100:.1f}% since morning)"
                       if line_mv > 0 else
                       f"📉 Line faded ({line_mv*100:.1f}% since morning)")
                fresh_signals.insert(0, tag)
            pick = {
                **pick,
                "pick":            line_moved_fresh["pick"],   # updated line e.g. -10.5
                "signals":         fresh_signals,
                "research":        line_moved_fresh["research"],
                "model_prob_pct":  line_moved_fresh["model_prob_pct"],
                "market_prob_pct": line_moved_fresh["market_prob_pct"],
                "edge":            line_moved_fresh["edge"],
                "edge_pct":        line_moved_fresh["edge_pct"],
            }
            logger.info(f"Line move: '{old_line}' → '{pick['pick']}' ({pick['game']})")
            final_singles.append(pick)
            continue

        # ── Normal replacement check ─────────────────────────────────────────
        # Best available new bet (different game+team) that beats this pick.
        # Only considered when allow_replace=True (explicit re-evaluate run).
        best_new = next(
            (d for d in truly_new
             if id(d) not in used_new and d["edge"] > pick["edge"] + 0.005),
            None,
        ) if allow_replace else None

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
            # Keep pick; refresh display fields from fresh analysis (signals, research,
            # model prob) so any model improvements show up without changing the core bet.
            fresh = new_pick_map.get(_signal_refresh_key(pick))
            if fresh:
                # Detect line movement: compare morning market_prob to current
                morning_mkt  = pick.get("market_prob", pick.get("contract_price", 0))
                current_mkt  = fresh.get("market_prob", morning_mkt)
                line_move    = current_mkt - morning_mkt  # positive = line moved toward our pick
                fresh_signals = list(fresh["signals"])
                if abs(line_move) >= LINE_MOVE_THRESHOLD:
                    if line_move > 0:
                        fresh_signals.insert(0,
                            f"📈 Sharp money: line moved +{line_move*100:.1f}% toward this pick since morning")
                    else:
                        fresh_signals.insert(0,
                            f"📉 Line faded: market moved {line_move*100:.1f}% against this pick since morning")
                pick = {
                    **pick,
                    "signals":         fresh_signals,
                    "research":        fresh["research"],
                    # Regenerate display context from fresh research so that any
                    # model/code improvements (e.g. new W-L fields) show up on
                    # subsequent runs without requiring a state reset.
                    "narrative":       fresh.get("narrative", pick.get("narrative", "")),
                    "context":         fresh.get("context",   pick.get("context",   [])),
                    "model_prob_pct":  fresh["model_prob_pct"],
                    "market_prob_pct": fresh["market_prob_pct"],
                    "edge":            fresh["edge"],
                    "edge_pct":        fresh["edge_pct"],
                }
            elif current_edge is not None:
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

    # Fill slots freed by dedup (pregame may be empty after removing cached
    # duplicates, leaving fewer than MAX_SINGLE_BETS in final_singles).
    for d in truly_new:
        if len(final_singles) >= MAX_SINGLE_BETS:
            break
        if id(d) not in used_new:
            final_singles.append(d)
            used_new.add(id(d))

    # ── Parlays ───────────────────────────────────────────────────────────────
    locked_parlays: List[Dict] = [
        p for p in state.get("parlays", [])
        if _cached_parlay_valid(p)
        and not any(leg.get("sport", "") in _WATCHLIST_SPORTS for leg in p.get("legs", []))
    ]
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
        ) if allow_replace else None
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

    # Fill any remaining slots (e.g. all cached parlays were invalid and dropped)
    for d in truly_new_par:
        if len(final_parlays) >= MAX_PARLAYS:
            break
        if id(d) not in used_new_par:
            new_d = d if isinstance(d, dict) else parlay_to_dict(d)
            final_parlays.append(new_d)
            used_new_par.add(id(d))

    # ── Props ─────────────────────────────────────────────────────────────────
    locked_props: List[Dict] = state.get("props", [])
    _update_lock_flags(locked_props)

    # Enforce per-sport cap on load — fixes state files that accumulated extras
    # from a previous bug.  Priority: locked > confidence > edge.
    _sport_tally = Counter(p.get("sport", "NBA") for p in locked_props)
    if any(v > MAX_PROPS_PER_SPORT for v in _sport_tally.values()):
        locked_props.sort(key=lambda p: (
            0 if p.get("locked") else 1,
            -_CONF_RANK.get(p.get("confidence", "MEDIUM"), 1),
            -p.get("edge", 0.0),
        ))
        _seen_sports: Counter = Counter()
        _trimmed: List[Dict] = []
        for p in locked_props:
            _s = p.get("sport", "NBA")
            if _seen_sports[_s] < MAX_PROPS_PER_SPORT:
                _trimmed.append(p)
                _seen_sports[_s] += 1
        locked_props = _trimmed
        logger.info(f"State: trimmed props to {MAX_PROPS_PER_SPORT}/sport (was over cap)")

    started_pr = [p for p in locked_props if p.get("locked")]
    pregame_pr = [p for p in locked_props if not p.get("locked")]
    locked_prop_keys = {_prop_key(p) for p in locked_props}

    # Build a fast lookup for signal/odds refresh (same-player same-prop_type)
    new_prop_map = {_prop_key(d): d for d in new_props}

    truly_new_props = [d for d in new_props if _prop_key(d) not in locked_prop_keys]
    truly_new_props.sort(
        key=lambda d: (
            -_CONF_RANK.get(d["confidence"] if isinstance(d, dict) else d.confidence, 0),
            -(d["edge"] if isinstance(d, dict) else getattr(d, "edge", 0.0)),
        ),
    )

    final_props: List[Dict] = list(started_pr)
    used_new_props: set = set()

    for prop in pregame_pr:
        prop_conf = _CONF_RANK.get(prop.get("confidence", "MEDIUM"), 1)
        prop_edge = prop.get("edge", 0.0)

        # Replacement criteria (only when allow_replace=True):
        #   1. Confidence upgrade (MEDIUM → HIGH), OR
        #   2. Same confidence but meaningfully better edge (≥ 0.5 pp, same as singles)
        best_new = next(
            (d for d in truly_new_props
             if id(d) not in used_new_props
             and (
                 _CONF_RANK.get(d["confidence"] if isinstance(d, dict) else d.confidence, 0) > prop_conf
                 or (d["edge"] if isinstance(d, dict) else getattr(d, "edge", 0.0)) > prop_edge + 0.005
             )),
            None,
        ) if allow_replace else None

        if best_new:
            new_d = best_new if isinstance(best_new, dict) else prop_to_dict(best_new)
            new_conf = new_d["confidence"]
            new_edge = new_d.get("edge", 0.0)
            reason = (
                "Confidence upgrade to HIGH"
                if _CONF_RANK.get(new_conf, 0) > prop_conf
                else f"Better edge: {new_edge*100:.1f}% vs {prop_edge*100:.1f}%"
            )
            warnings.append({
                "type": "prop_replaced",
                "removed": f"{prop['player']} — {prop['prop_type']}",
                "new": f"{new_d['player']} — {new_d['prop_type']}",
                "reason": reason,
            })
            used_new_props.add(id(best_new))
            final_props.append(new_d)
        else:
            # Keep pick; refresh signals/odds/model numbers in-place (same as singles)
            fresh = new_prop_map.get(_prop_key(prop))
            if fresh:
                fresh_d = fresh if isinstance(fresh, dict) else prop_to_dict(fresh)
                prop = {
                    **prop,
                    "signals":      fresh_d.get("signals",   prop.get("signals",   [])),
                    "research":     fresh_d.get("research",  prop.get("research",  [])),
                    # Regenerate display context alongside research (same pattern as singles)
                    "narrative":    fresh_d.get("narrative", prop.get("narrative", "")),
                    "context":      fresh_d.get("context",   prop.get("context",   [])),
                    "model_line":   fresh_d.get("model_line",   prop.get("model_line")),
                    "model_prob":   fresh_d.get("model_prob",   prop.get("model_prob",   0.0)),
                    "market_prob":  fresh_d.get("market_prob",  prop.get("market_prob",  0.0)),
                    "market_line":  fresh_d.get("market_line",  prop.get("market_line")),
                    "edge":         fresh_d.get("edge",         prop.get("edge",         0.0)),
                    "edge_pct":     fresh_d.get("edge_pct",     prop.get("edge_pct",     0.0)),
                    "odds_display": fresh_d.get("odds_display", prop.get("odds_display", "")),
                    "confidence":   fresh_d.get("confidence",   prop.get("confidence")),
                }
            final_props.append(prop)

    # Fill remaining prop slots — only up to MAX_PROPS_PER_SPORT per sport.
    # This is critical when the morning baseline had 0 props (first run with
    # new code), but must NOT add extras when slots are already full.
    sport_counts = Counter(
        p.get("sport", "NBA") for p in final_props
    )
    for d in truly_new_props:
        if id(d) not in used_new_props:
            sport = d.get("sport", "NBA") if isinstance(d, dict) else getattr(d, "sport", "NBA")
            if sport_counts[sport] >= MAX_PROPS_PER_SPORT:
                continue
            new_d = d if isinstance(d, dict) else prop_to_dict(d)
            final_props.append(new_d)
            sport_counts[sport] += 1
            used_new_props.add(id(d))

    if warnings:
        logger.info(f"State merge: {len(warnings)} substitution(s) / warning(s)")

    return final_singles, final_parlays, final_props, warnings
