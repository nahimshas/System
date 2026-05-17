#!/usr/bin/env python3
"""
One-time backfill: import settled historical picks into the shadow log.

Reads state/history.json and state/watchlist_history.json and creates
synthetic shadow log entries for every settled record so the calibration
engine has Phase A data immediately instead of waiting weeks for new
picks to accumulate.

Limitations of backfilled entries:
  - model_prob_raw is null      — never captured pre-deployment
  - credibility_cap_fired is False — cap didn't exist back then
  - hardcap_fired is False
  - injury_cap_fired is False
  - signal_count is 0           — not stored in history files

These missing fields only affect cap counterfactual analysis (Steps 6/7),
which has to accumulate fresh post-deployment data anyway. The Phase A and
Phase B calibration math uses only (model_prob, outcome), both fully
available in the historical records.

Idempotent: existing shadow log entries (keyed by date|sport|game|market|side)
are never overwritten by this script.

Run once after deploying the calibration system:
  python3 backfill_shadow_log.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

# Allow `from src...` imports when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.state.shadow_log import (
    SHADOW_LOG_DIR,
    SCHEMA_VERSION,
    _stable_key,
    _market_type,
    _pick_side,
    _load_shard,
    _save_shard_atomic,
)


HISTORY_FILES = [
    Path("state/history.json"),
    Path("state/watchlist_history.json"),
]


class _HistoryRec:
    """Duck-typed wrapper so history records can use the shadow log key extractors."""

    __slots__ = ("bet_type", "pick", "home_team", "away_team")

    def __init__(self, rec):
        self.bet_type  = rec.get("bet_type", "")  or ""
        self.pick      = rec.get("pick", "")      or ""
        self.home_team = rec.get("home_team", "") or ""
        self.away_team = rec.get("away_team", "") or ""


def _outcome_normalize(result: str):
    """Map history result label → canonical shadow log outcome."""
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


def _build_backfill_entry(rec: dict) -> dict:
    """Build a synthetic shadow log entry from a settled history record."""
    settled_at_iso = f"{rec.get('date', '')}T23:59:59Z"   # approximation
    market_prob    = float(rec.get("market_prob_pct", 0)) / 100.0
    model_prob     = float(rec.get("model_prob_pct",  0)) / 100.0
    edge           = float(rec.get("edge_pct",        0)) / 100.0
    wrapper        = _HistoryRec(rec)

    return {
        # Identity
        "date":          rec.get("date", ""),
        "sport":         rec.get("sport", ""),
        "game":          rec.get("game", ""),
        "market_type":   _market_type(wrapper),
        "pick_side":     _pick_side(wrapper),
        "pick":          rec.get("pick", ""),
        "bet_type":      rec.get("bet_type", ""),
        "commence_time": "",   # game already finished — not needed

        # Model output (raw + cap flags unavailable for historical picks)
        "model_prob":             model_prob,
        "model_prob_raw":         None,
        "credibility_cap_fired":  False,
        "injury_cap_fired":       False,
        "hardcap_fired":          False,

        # Market
        "market_prob_at_first_pick":     market_prob,
        "market_prob_at_last_update":    market_prob,
        "last_update_minutes_before_game": None,
        "market_prob_at_close":          None,

        # Edges
        "raw_edge":       None,
        "displayed_edge": edge,
        "effective_edge": edge,

        # Confidence components (signal_count not stored historically)
        "signal_count":           0,
        "stats_available":        True,
        "final_confidence_label": rec.get("confidence", ""),

        # Routing / lifecycle
        "displayed_in_top": True,   # historical records were all displayed picks
        "first_seen_at":    settled_at_iso,
        "last_updated_at":  settled_at_iso,
        "game_locked":      True,

        # Settlement
        "outcome":    _outcome_normalize(rec.get("result", "")),
        "settled_at": settled_at_iso,

        # Schema + provenance
        "schema_version": SCHEMA_VERSION,
        "backfilled":     True,
    }


def backfill():
    SHADOW_LOG_DIR.mkdir(parents=True, exist_ok=True)

    files_read           = 0
    records_seen         = 0
    backfilled           = 0
    skipped_existing     = 0
    skipped_no_outcome   = 0
    skipped_parlay       = 0
    skipped_malformed    = 0
    updates_by_month: dict = {}

    print("Reading history files...\n")

    for path in HISTORY_FILES:
        if not path.exists():
            print(f"  {path} not found, skipping")
            continue
        try:
            with open(path) as f:
                records = json.load(f)
        except Exception as e:
            print(f"  Failed to read {path}: {e}")
            continue
        files_read += 1
        if not isinstance(records, list):
            print(f"  {path}: not a list, skipping")
            continue
        print(f"  {path}: {len(records)} records")

        for rec in records:
            if not isinstance(rec, dict):
                continue
            records_seen += 1

            if rec.get("type") == "parlay":
                skipped_parlay += 1
                continue

            outcome = _outcome_normalize(rec.get("result", ""))
            if outcome is None:
                skipped_no_outcome += 1
                continue

            date_str = rec.get("date", "")
            sport    = rec.get("sport", "")
            game     = rec.get("game", "")
            if not (date_str and sport and game):
                skipped_malformed += 1
                continue

            try:
                d = date.fromisoformat(date_str)
            except Exception:
                skipped_malformed += 1
                continue

            wrapper = _HistoryRec(rec)
            market_type = _market_type(wrapper)
            pick_side   = _pick_side(wrapper)
            if not (market_type and pick_side):
                skipped_malformed += 1
                continue

            key   = _stable_key(d, sport, game, market_type, pick_side)
            month = f"{d.year:04d}-{d.month:02d}"
            updates_by_month.setdefault(month, {})[key] = _build_backfill_entry(rec)

    print(f"\nWriting shadow log shards...")

    for month, new_entries in sorted(updates_by_month.items()):
        path  = SHADOW_LOG_DIR / f"{month}.json"
        shard = _load_shard(path)
        existing = shard.setdefault("entries", {})

        added = 0
        for key, entry in new_entries.items():
            if key in existing:
                skipped_existing += 1
                continue
            existing[key] = entry
            added += 1

        if added:
            _save_shard_atomic(path, shard)
            print(f"  {path}: +{added} new entries (total in shard: {len(existing)})")
            backfilled += added

    # Per-sport breakdown of what we actually wrote
    sport_counts: dict = {}
    for month_entries in updates_by_month.values():
        for entry in month_entries.values():
            sport_counts[entry["sport"]] = sport_counts.get(entry["sport"], 0) + 1

    print(f"\n{'='*50}")
    print(f"Backfill complete")
    print(f"{'='*50}")
    print(f"  Files read:           {files_read}")
    print(f"  Records seen:         {records_seen}")
    print(f"  Backfilled:           {backfilled}")
    print(f"  Skipped (existing):   {skipped_existing}")
    print(f"  Skipped (no outcome): {skipped_no_outcome}")
    print(f"  Skipped (parlay):     {skipped_parlay}")
    print(f"  Skipped (malformed):  {skipped_malformed}")

    if sport_counts:
        print(f"\nPer-sport entries (what feeds calibration):")
        for sport in sorted(sport_counts.keys()):
            n = sport_counts[sport]
            # Phase threshold cues
            if   n >= 400: phase_note = "→ Phase B eligible"
            elif n >= 100: phase_note = "→ Phase A eligible"
            else:          phase_note = f"→ {100 - n} more for Phase A"
            print(f"  {sport:5s}: {n:4d}  {phase_note}")

    print(f"\nNext step: commit + push + run a workflow.")
    print(f"The calibration panel will immediately show Phase A ratios for sports with 100+ picks.")


if __name__ == "__main__":
    backfill()
