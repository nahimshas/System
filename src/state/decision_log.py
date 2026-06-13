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
                market_p = _f(c.get("market_prob"))
                edge = _f(c.get("edge"))
                if edge is None and model_p is not None and market_p is not None:
                    edge = model_p - market_p

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
                    "model_prob": model_p,
                    "market_prob_at_first_pick": existing.get("market_prob_at_first_pick", market_p)
                        if existing else market_p,
                    "market_prob_at_last_update": market_p,
                    "edge": edge,
                    "made": bool(c.get("made", False)),
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
