"""
Decision log — the full candidate + feature archive for exhaustive CLV analysis.

WHY THIS EXISTS (separate from the shadow log):
  • shadow_log  → every pick the model PRODUCED (positive-edge). Feeds calibration
                  and caps, so it must stay clean — only real picks.
  • decision_log → EVERY candidate the model EVALUATED: both sides of every market
                  on every analyzed game, including sub-threshold and the rejected
                  side, PLUS the structured model inputs (features) behind each
                  probability. Pure analysis archive — NEVER feeds the display or
                  the calibration/cap engines.

This is the "log everything so you can improve later" layer. With it we can:
  • measure CLV/outcome for picks the model REJECTED (not just the ones it made)
  • re-evaluate any tweak against the full candidate universe + real closing lines
  • segment CLV by any model input (xFIP gap, rest, park, Elo, injury, …)

Design mirrors shadow_log: month-sharded, idempotent, game-locked, exception-safe.

Stable key:  (date | sport | game | market_type | side)
  "2026-06-13|MLB|SF @ ARI|Moneyline|Arizona Diamondbacks"
  "2026-06-13|MLB|SF @ ARI|Total|over"

Each row is self-contained (carries the game-level `features` dict) so downstream
analysis is a trivial flat scan — no joins required.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DECISION_LOG_DIR = Path("state/decision_log")
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Path / shard management  (mirrors shadow_log)
# ---------------------------------------------------------------------------

def _shard_path(run_date: date) -> Path:
    return DECISION_LOG_DIR / f"{run_date.year:04d}-{run_date.month:02d}.json"


def _load_shard(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "month": path.stem, "entries": {}}
    try:
        with open(path) as f:
            data = json.load(f)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("entries", {})
        return data
    except Exception as e:
        logger.warning(f"Decision log shard unreadable ({path}): {e} — starting fresh")
        return {"schema_version": SCHEMA_VERSION, "month": path.stem, "entries": {}}


def _save_shard_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_key(run_date: date, sport: str, game: str, market_type: str, side: str) -> str:
    return f"{run_date.isoformat()}|{sport}|{game}|{market_type}|{side}"


def _game_started(commence_time: str) -> bool:
    if not commence_time:
        return False
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= dt
    except Exception:
        return False


def _f(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SOCCER = {"MLS", "WC", "LIGAMX"}


def _grade_candidate(entry: Dict[str, Any], home_score: float, away_score: float) -> Optional[str]:
    """
    Grade one candidate from the final score. Returns 'win'/'loss'/'push'/None.

    Uses the structured market_type/side/line — for ML/Spread the `side` is the
    stored home/away team name, so no fuzzy matching is needed. Soccer moneylines
    are 3-way (a draw LOSES a team bet, it is not a push).
    """
    mt = entry.get("market_type", "")
    side = str(entry.get("side", "")).strip()
    line = _f(entry.get("line"))
    home = (entry.get("home_team") or "").strip()
    away = (entry.get("away_team") or "").strip()
    soccer = (entry.get("sport") or "").upper() in _SOCCER

    if mt == "Draw":
        return "win" if home_score == away_score else "loss"

    if mt in ("Moneyline", "Spread"):
        if side == home:
            my, opp = home_score, away_score
        elif side == away:
            my, opp = away_score, home_score
        else:
            # Fallback: case-insensitive containment either direction
            sl = side.lower()
            if home.lower() in sl or sl in home.lower():
                my, opp = home_score, away_score
            elif away.lower() in sl or sl in away.lower():
                my, opp = away_score, home_score
            else:
                return None
        if mt == "Moneyline":
            if my > opp:
                return "win"
            if my < opp:
                return "loss"
            return "loss" if soccer else "push"   # soccer draw loses a team ML
        # Spread (handicap): margin + line
        if line is None:
            return None
        m = (my - opp) + line
        if m > 0:
            return "win"
        if m < 0:
            return "loss"
        return "push"

    if mt == "Total":
        if line is None:
            return None
        total = home_score + away_score
        over = "over" in side.lower()
        if total == line:
            return "push"
        hit_over = total > line
        return "win" if (hit_over == over) else "loss"

    return None


def settle_decision_from_scores(today: date) -> int:
    """
    Grade decision-log candidate OUTCOMES from ESPN final scores — for BOTH sides
    of every market, including the picks the model rejected. Completes the loop:
    rejected-pick CLV (captured elsewhere) now joins rejected-pick win/loss.

    Isolated and additive: only reads scores + writes `outcome`/`settled_at` to
    the decision log. Never touches shadow/history settlement. Idempotent (already
    -graded rows skipped), never settles today's games, fully exception-safe.

    Score-based sports only: NBA, MLB (main paths) + NHL, WNBA, MLS, WC (watchlist
    paths; soccer uses the 90-minute score). IPL is skipped (no reliable historical
    score source — mirrors the shadow log's caution). Returns rows newly graded.
    """
    MAIN = {"NBA", "MLB"}
    WATCH = {"NHL", "WNBA", "MLS", "WC", "LIGAMX"}
    SUPPORTED = MAIN | WATCH
    try:
        from src.data.outcome_checker import (
            _fetch_espn_final_scores, _fetch_watchlist_final_scores, _find_game_score,
        )
        from collections import defaultdict as _dd
        if not DECISION_LOG_DIR.exists():
            return 0
        today_str = today.isoformat()
        loaded: Dict[Path, Dict[str, Any]] = {}
        needs: Dict[tuple, list] = _dd(list)
        for shard_path in sorted(DECISION_LOG_DIR.glob("*.json")):
            shard = _load_shard(shard_path)
            loaded[shard_path] = shard
            for key, entry in shard.get("entries", {}).items():
                if entry.get("outcome") is not None:
                    continue
                dstr = entry.get("date", "")
                if not dstr or dstr >= today_str:
                    continue
                if not (entry.get("game_locked") or _game_started(entry.get("commence_time", ""))):
                    continue
                sport = (entry.get("sport") or "").upper()
                if sport not in SUPPORTED:
                    continue
                needs[(dstr, sport)].append((shard_path, entry))
        if not needs:
            return 0

        score_cache: Dict[tuple, Dict] = {}
        for (dstr, sport) in needs:
            try:
                gd = date.fromisoformat(dstr)
            except Exception:
                continue
            score_cache[(dstr, sport)] = (
                _fetch_espn_final_scores(sport, gd) if sport in MAIN
                else _fetch_watchlist_final_scores(sport, gd)
            )

        modified: set = set()
        graded = 0
        now_iso = datetime.now(timezone.utc).isoformat()
        for (dstr, sport), items in needs.items():
            scores = score_cache.get((dstr, sport)) or {}
            if not scores:
                continue
            for shard_path, entry in items:
                parts = entry.get("game", "").split(" @ ")
                away_abbr = parts[0].strip() if len(parts) == 2 else ""
                home_abbr = parts[1].strip() if len(parts) == 2 else entry.get("game", "").strip()
                sd = _find_game_score(scores, home_abbr, away_abbr)
                if not sd:
                    continue
                res = _grade_candidate(entry, sd["home_score"], sd["away_score"])
                if res is None:
                    continue
                entry["outcome"] = res
                entry["settled_at"] = now_iso
                modified.add(shard_path)
                graded += 1

        for shard_path in modified:
            _save_shard_atomic(shard_path, loaded[shard_path])
        if graded:
            logger.info(f"Decision-log settlement: {graded} candidate(s) graded from final scores")
        return graded
    except Exception as e:
        logger.warning(f"Decision-log settlement failed (non-fatal): {e}")
        return 0


def record_candidates(
    run_date: date,
    sport: str,
    game: str,
    commence_time: str,
    home_team: str,
    away_team: str,
    candidates: List[Dict[str, Any]],
    features: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Record every evaluated candidate (both sides of every market) for one game.

    `candidates` is a list of dicts, each:
        {
          "market_type": "Moneyline" | "Spread" | "Total" | "Draw",
          "side":        team name | "over" | "under" | "draw",
          "model_prob":  float,            # model P(this side wins)
          "market_prob": float,            # opening market P (no-vig consensus)
          "edge":        float,            # model_prob - market_prob
          "made":        bool,             # did it clear MIN_EDGE (would be bet)
          "line":        float | None,     # the spread/total line, if any
        }

    `features` is the structured model-input snapshot for the game (shared by all
    candidates of that game). Stored on every row so each row is self-contained.

    Idempotent (stable key per side). Game-locked: once commence_time has passed,
    existing rows are frozen (only CLV/outcome get stamped later, elsewhere).
    Exception-safe: returns count written, 0 on any failure — never raises.
    """
    try:
        path = _shard_path(run_date)
        shard = _load_shard(path)
        entries: Dict[str, Any] = shard.setdefault("entries", {})
        now_iso = datetime.now(timezone.utc).isoformat()
        locked = _game_started(commence_time)
        written = 0

        for c in candidates:
            try:
                mtype = c.get("market_type", "")
                side = str(c.get("side", "")).strip()
                if not (mtype and side):
                    continue
                key = _stable_key(run_date, sport, game, mtype, side)

                if key in entries and entries[key].get("game_locked"):
                    continue  # frozen — don't overwrite post-commence snapshot

                model_p = _f(c.get("model_prob"))
                model_p_raw = _f(c.get("model_prob_raw"))
                market_p = _f(c.get("market_prob"))
                edge = _f(c.get("edge"))
                if edge is None and model_p is not None and market_p is not None:
                    edge = model_p - market_p
                raw_edge = _f(c.get("raw_edge"))
                if raw_edge is None and model_p_raw is not None and market_p is not None:
                    raw_edge = model_p_raw - market_p

                existing = entries.get(key, {})
                entries[key] = {
                    "date": run_date.isoformat(),
                    "sport": sport,
                    "game": game,
                    "home_team": home_team,
                    "away_team": away_team,
                    "market_type": mtype,
                    "side": side,
                    "line": _f(c.get("line")),
                    "commence_time": commence_time,
                    "model_prob": model_p,            # post-credibility-cap (acted on)
                    "model_prob_raw": model_p_raw,    # pre-cap (made picks only; None if rejected)
                    "market_prob_at_first_pick": existing.get("market_prob_at_first_pick", market_p)
                        if existing else market_p,
                    "market_prob_at_last_update": market_p,
                    "edge": edge,                     # post-cap edge
                    "raw_edge": raw_edge,             # pre-cap edge (made picks only)
                    "made": bool(c.get("made", False)),
                    "final_confidence_label": c.get("confidence") or existing.get("final_confidence_label"),
                    "features": features or existing.get("features") or {},
                    # CLV / outcome stamped later by the snapshot + settlement passes
                    "market_prob_at_close": existing.get("market_prob_at_close"),
                    "clv": existing.get("clv"),
                    "outcome": existing.get("outcome"),
                    "first_seen_at": existing.get("first_seen_at", now_iso),
                    "last_updated_at": now_iso,
                    "game_locked": locked,
                    "schema_version": SCHEMA_VERSION,
                }
                written += 1
            except Exception as e:
                logger.debug(f"Decision log: skipped candidate {c!r}: {e}")
                continue

        if written:
            _save_shard_atomic(path, shard)
        return written
    except Exception as e:
        logger.warning(f"Decision log record_candidates failed (non-fatal): {e}")
        return 0
