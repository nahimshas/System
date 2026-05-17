"""
Shadow log — calibration data foundation.

Records every pick the model produces (not just the top-5 displayed) so the
calibration engine can later learn per-sport realised hit rates and adjust
edges accordingly. The shadow log is independent of state files:

  • state/picks_YYYY-MM-DD.json  — display state (gets reset, re-run, edited)
  • state/shadow_log/YYYY-MM.json — append/update-only historical record

Design principles:
  • Idempotent — re-runs UPDATE the same entry, never duplicate
  • Game-locked — entries frozen once commence_time has passed
  • Exception-safe — failures are logged but never propagate
  • Separation — shadow log NEVER feeds the display layer
  • Schema-versioned — new optional fields can be added without migration

Stable key:  (date | sport | game | market_type | pick_side)

Example keys:
  "2026-05-16|MLB|ARI @ COL|RunLine|Rockies"
  "2026-05-16|NBA|LAL @ DEN|Total|over"

The same pick from morning + afternoon re-runs maps to the same row.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

SHADOW_LOG_DIR = Path("state/shadow_log")
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Path / shard management
# ---------------------------------------------------------------------------

def _shard_path(run_date: date) -> Path:
    """Return the path to the month-shard for this date."""
    return SHADOW_LOG_DIR / f"{run_date.year:04d}-{run_date.month:02d}.json"


def _load_shard(path: Path) -> Dict[str, Any]:
    """Load a month-shard. Returns a fresh empty structure if missing/corrupted."""
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "month": path.stem,
            "entries": {},
        }
    try:
        with open(path) as f:
            data = json.load(f)
        # Defensive: ensure required keys exist (future schema migrations land here)
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("entries", {})
        return data
    except Exception as e:
        # Don't lose data — back up the corrupted file and start fresh
        try:
            backup = path.with_suffix(path.suffix + f".corrupted-{int(datetime.now().timestamp())}")
            path.rename(backup)
            logger.error(f"Shadow log {path} corrupted; backed up to {backup}, starting fresh: {e}")
        except Exception:
            logger.error(f"Shadow log {path} corrupted and could not be backed up: {e}")
        return {
            "schema_version": SCHEMA_VERSION,
            "month": path.stem,
            "entries": {},
        }


def _save_shard_atomic(path: Path, data: Dict[str, Any]) -> None:
    """Atomic write — never leaves the file half-written if interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to temp file in same directory, then rename (atomic on POSIX)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
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
# Keys & helpers
# ---------------------------------------------------------------------------

def _safe_optional_float(v: Any) -> Optional[float]:
    """Coerce to float, or return None if value is None / not numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _market_type(rec: Any) -> str:
    """Map BetRecommendation.bet_type to canonical market_type label."""
    bt = getattr(rec, "bet_type", "") or ""
    # Already canonical for ML/Spread/Total; soccer uses "Draw"
    if bt in ("Moneyline", "Spread", "Total", "Draw"):
        return bt
    return bt or "Unknown"


def _pick_side(rec: Any) -> str:
    """
    Extract a stable pick-side label from the pick string.

    For ML/Spread: the team name (just the bare team token, no spread number)
    For Total:     "over" or "under"
    For Draw:      "draw"
    Other:         the full pick text (best-effort)
    """
    pick = (getattr(rec, "pick", "") or "").strip()
    bt   = getattr(rec, "bet_type", "") or ""

    if bt == "Total":
        low = pick.lower()
        if "over" in low:
            return "over"
        if "under" in low:
            return "under"
        return pick

    if bt == "Draw":
        return "draw"

    if bt in ("Moneyline", "Spread"):
        home = (getattr(rec, "home_team", "") or "").strip()
        away = (getattr(rec, "away_team", "") or "").strip()
        if home and pick.startswith(home):
            return home
        if away and pick.startswith(away):
            return away
        # Fallback: strip trailing spread number (e.g. "Rockies +1.5" → "Rockies")
        toks = pick.split()
        if len(toks) >= 2 and (toks[-1].startswith("+") or toks[-1].startswith("-")):
            return " ".join(toks[:-1])
        return pick

    return pick


def _stable_key(
    run_date: date, sport: str, game: str, market_type: str, pick_side: str
) -> str:
    """Deterministic key for a unique (date, sport, game, market, pick-side) tuple."""
    return f"{run_date.isoformat()}|{sport}|{game}|{market_type}|{pick_side}"


def _game_started(commence_time: str) -> bool:
    """True if commence_time has already passed (UTC)."""
    if not commence_time:
        return False
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) >= dt
    except Exception:
        return False


def _minutes_before_game(commence_time: str) -> Optional[int]:
    """Minutes between now and commence_time. Negative = game already started."""
    if not commence_time:
        return None
    try:
        dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        delta = dt - datetime.now(timezone.utc)
        return int(delta.total_seconds() / 60)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Entry builder
# ---------------------------------------------------------------------------

def _build_entry(
    rec: Any, run_date: date, *, displayed_in_top: bool, now_iso: str
) -> Dict[str, Any]:
    """Build a fresh shadow log entry from a BetRecommendation."""
    sport       = getattr(rec, "sport", "")
    game        = getattr(rec, "game", "")
    market_type = _market_type(rec)
    pick_side   = _pick_side(rec)
    commence    = getattr(rec, "commence_time", "") or ""

    return {
        # Identity
        "date":          run_date.isoformat(),
        "sport":         sport,
        "game":          game,
        "market_type":   market_type,
        "pick_side":     pick_side,
        "pick":          getattr(rec, "pick", ""),
        "bet_type":      getattr(rec, "bet_type", ""),
        "commence_time": commence,

        # Model output — raw value and cap trigger flags populated by sport
        # analyzers when they apply caps. Defaults preserve backward compat.
        "model_prob":             float(getattr(rec, "model_prob", 0.0)),
        "model_prob_raw":         _safe_optional_float(getattr(rec, "model_prob_raw", None)),
        "credibility_cap_fired":  bool(getattr(rec, "credibility_cap_fired", False)),
        "injury_cap_fired":       bool(getattr(rec, "injury_cap_fired",      False)),
        "hardcap_fired":          bool(getattr(rec, "hardcap_fired",         False)),

        # Market data
        "market_prob_at_first_pick":     float(getattr(rec, "market_prob", 0.0)),
        "market_prob_at_last_update":    float(getattr(rec, "market_prob", 0.0)),
        "last_update_minutes_before_game": _minutes_before_game(commence),
        "market_prob_at_close":          None,  # populated only if closing-line capture added

        # Edges
        "raw_edge":       None,                                # populated with analyzer integration
        "displayed_edge": float(getattr(rec, "edge", 0.0)),
        "effective_edge": float(getattr(rec, "edge", 0.0)),    # = displayed during Phase 0

        # Confidence components
        "signal_count":          len(getattr(rec, "signals", []) or []),
        "stats_available":       bool(getattr(rec, "stats_available", True)),
        "final_confidence_label": getattr(rec, "confidence", ""),

        # Routing / lifecycle
        "displayed_in_top": bool(displayed_in_top),
        "first_seen_at":   now_iso,
        "last_updated_at": now_iso,
        "game_locked":     False,

        # Settlement (populated by settler later)
        "outcome":    None,
        "settled_at": None,

        # Schema
        "schema_version": SCHEMA_VERSION,
    }


def _update_entry(existing: Dict[str, Any], rec: Any, now_iso: str) -> Dict[str, Any]:
    """Update an existing entry with fresh values from a re-run."""
    commence = existing.get("commence_time") or getattr(rec, "commence_time", "") or ""

    # Game-time lock: do not modify entries after game start
    if _game_started(commence):
        if not existing.get("game_locked"):
            existing["game_locked"] = True
        return existing

    # Refresh mutable fields (model belief, market, edges, signals, cap flags)
    existing["model_prob"]                      = float(getattr(rec, "model_prob", 0.0))
    existing["market_prob_at_last_update"]      = float(getattr(rec, "market_prob", 0.0))
    existing["last_update_minutes_before_game"] = _minutes_before_game(commence)
    existing["displayed_edge"]                  = float(getattr(rec, "edge", 0.0))
    existing["effective_edge"]                  = float(getattr(rec, "edge", 0.0))
    existing["signal_count"]                    = len(getattr(rec, "signals", []) or [])
    existing["final_confidence_label"]          = getattr(rec, "confidence", "")
    existing["last_updated_at"]                 = now_iso
    # Refresh calibration metadata if the analyzer populated it; preserve
    # whatever was already stored otherwise (handles old shadow log entries).
    _raw_new = _safe_optional_float(getattr(rec, "model_prob_raw", None))
    if _raw_new is not None:
        existing["model_prob_raw"]        = _raw_new
        existing["credibility_cap_fired"] = bool(getattr(rec, "credibility_cap_fired", False))
        existing["injury_cap_fired"]      = bool(getattr(rec, "injury_cap_fired",      False))
        existing["hardcap_fired"]         = bool(getattr(rec, "hardcap_fired",         False))
        existing["stats_available"]       = bool(getattr(rec, "stats_available",       True))
    return existing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_picks(
    picks: Iterable[Any],
    run_date: date,
    *,
    displayed_top_keys: Optional[Set[str]] = None,
) -> int:
    """
    Append/update shadow log entries for every pick in `picks`.

    Idempotent: same (date, sport, game, market, pick-side) tuple maps to the
    same row across re-runs. Game-time lock prevents post-game modification.

    Args:
        picks: iterable of BetRecommendation (or compatible duck-typed) objects.
        run_date: today's date (controls which month-shard to write).
        displayed_top_keys: optional set of stable keys that ended up in the
            top-5 display slot — used to populate `displayed_in_top` flag.

    Returns:
        Number of entries written or updated. 0 on any failure.
    """
    try:
        path  = _shard_path(run_date)
        shard = _load_shard(path)
        entries: Dict[str, Any] = shard.setdefault("entries", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        displayed_top_keys = displayed_top_keys or set()
        written = 0

        for rec in picks:
            try:
                sport       = getattr(rec, "sport", "")
                game        = getattr(rec, "game", "")
                market_type = _market_type(rec)
                pick_side   = _pick_side(rec)
                if not (sport and game and market_type and pick_side):
                    continue   # malformed — skip silently

                key = _stable_key(run_date, sport, game, market_type, pick_side)
                in_top = key in displayed_top_keys

                if key in entries:
                    _update_entry(entries[key], rec, now_iso)
                    # `displayed_in_top` can be promoted from False → True (a re-run
                    # may surface a pick that wasn't in the top-5 earlier). Never demote.
                    if in_top:
                        entries[key]["displayed_in_top"] = True
                else:
                    entries[key] = _build_entry(
                        rec, run_date, displayed_in_top=in_top, now_iso=now_iso
                    )
                written += 1
            except Exception as e:
                logger.warning(f"Shadow log: skipped malformed pick ({rec!r}): {e}")
                continue

        if written:
            _save_shard_atomic(path, shard)
            logger.info(
                f"Shadow log: {written} entries recorded → {path} "
                f"(total in shard: {len(entries)})"
            )
        return written

    except Exception as e:
        # Never propagate — shadow logging is non-critical
        logger.error(f"Shadow log write failed (non-fatal): {e}")
        return 0


def compute_top_keys(top_picks: Iterable[Any], run_date: date) -> Set[str]:
    """Helper: compute the set of stable keys for picks that made the top-5 slots."""
    keys: Set[str] = set()
    for rec in top_picks:
        try:
            keys.add(_stable_key(
                run_date,
                getattr(rec, "sport", ""),
                getattr(rec, "game", ""),
                _market_type(rec),
                _pick_side(rec),
            ))
        except Exception:
            continue
    return keys


# ---------------------------------------------------------------------------
# Settlement integration
# ---------------------------------------------------------------------------

# History file paths (matches the existing settlement infrastructure)
_HISTORY_PATHS = [
    Path("state/history.json"),            # main singles + parlays (NBA/MLB)
    Path("state/watchlist_history.json"),  # NHL/IPL/WNBA/MLS
]


def _normalize_outcome(result: str) -> Optional[str]:
    """Map history.json result labels to canonical shadow log outcomes."""
    r = (result or "").upper()
    if r == "WON":
        return "win"
    if r == "LOST":
        return "loss"
    if r == "PUSH":
        return "push"
    if r in ("CANCEL", "CANCELED", "CANCELLED", "POSTPONED"):
        return "cancel"
    return None


class _HistoryRec:
    """Duck-typed wrapper so history records can use the same key extractors."""
    __slots__ = ("bet_type", "pick", "home_team", "away_team")

    def __init__(self, rec: Dict[str, Any]):
        self.bet_type  = rec.get("bet_type", "") or ""
        self.pick      = rec.get("pick", "") or ""
        self.home_team = rec.get("home_team", "") or ""
        self.away_team = rec.get("away_team", "") or ""


def _key_for_history_rec(rec: Dict[str, Any]) -> Optional[str]:
    """Compute the shadow log stable key for a history record. None if malformed."""
    date_str = rec.get("date", "")
    sport    = rec.get("sport", "")
    game     = rec.get("game", "")
    if not (date_str and sport and game):
        return None
    try:
        d = date.fromisoformat(date_str)
    except Exception:
        return None
    wrapper = _HistoryRec(rec)
    return _stable_key(d, sport, game, _market_type(wrapper), _pick_side(wrapper))


def settle_from_history() -> int:
    """
    Read settled outcomes from state/history.json and state/watchlist_history.json
    and propagate them to matching shadow log entries.

    Run AFTER the existing settlers (check_and_settle, check_and_settle_watchlist,
    settle_watchlist_pending) have completed for the day.

    Idempotent: shadow log entries whose `outcome` is already set are skipped.
    Records that don't match any shadow log entry are silently ignored (the
    shadow log may not have been writing yet when that pick was made).

    Returns the number of shadow log entries newly settled by this call.
    Failures are logged but never raised.
    """
    try:
        # Step 1: gather (key, outcome, date) triples from all history files
        updates: Dict[str, Dict[str, str]] = {}  # month-shard → {key: outcome}
        records_seen = 0

        for path in _HISTORY_PATHS:
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    hist = json.load(f)
            except Exception as e:
                logger.warning(f"Shadow settler: failed to read {path}: {e}")
                continue
            if not isinstance(hist, list):
                continue

            for rec in hist:
                if not isinstance(rec, dict):
                    continue
                # Skip parlays — they don't map 1:1 to shadow log entries
                if rec.get("type") == "parlay":
                    continue
                records_seen += 1
                outcome = _normalize_outcome(rec.get("result", ""))
                if outcome is None:
                    continue
                key = _key_for_history_rec(rec)
                if not key:
                    continue
                # Extract month from key for shard routing
                month = key.split("|", 1)[0][:7]  # "2026-05-16|..." → "2026-05"
                updates.setdefault(month, {})[key] = outcome

        if not updates:
            return 0

        # Step 2: apply updates per shard
        now_iso = datetime.now(timezone.utc).isoformat()
        total_settled = 0

        for month, key_outcomes in updates.items():
            path = SHADOW_LOG_DIR / f"{month}.json"
            if not path.exists():
                continue
            shard = _load_shard(path)
            entries: Dict[str, Any] = shard.get("entries", {})
            changed = 0
            for key, outcome in key_outcomes.items():
                entry = entries.get(key)
                if not entry:
                    continue
                if entry.get("outcome") is not None:
                    continue   # already settled — idempotent
                entry["outcome"]    = outcome
                entry["settled_at"] = now_iso
                changed += 1
            if changed:
                _save_shard_atomic(path, shard)
                total_settled += changed

        if total_settled:
            logger.info(
                f"Shadow log settlement: {total_settled} entries newly settled "
                f"(scanned {records_seen} history records)"
            )
        return total_settled

    except Exception as e:
        logger.error(f"Shadow log settlement failed (non-fatal): {e}")
        return 0
