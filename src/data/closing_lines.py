"""
Closing-line capture (CLV) — fetches the market's final pre-game probabilities
from The Odds API historical archive and stamps them onto shadow log entries.

Why the historical endpoint:
  • The standard /odds endpoint drops games the moment they start; closing
    lines must be witnessed live. The historical archive preserves snapshots
    at 5-minute intervals, so the closing line for ANY past game can be
    fetched after the fact — minutes, days, or weeks later.
  • This makes capture self-healing: every run scans the shadow log for
    entries missing `market_prob_at_close` and backfills them, so a failed
    or skipped night costs nothing — the data is picked up on the next pass.

Cost model (Odds API): a historical odds request costs 10 × markets × regions.
One request returns ALL games for that sport at that timestamp, so cost scales
with (sport × start-time wave), not with pick count. Requests are grouped per
wave and only the market types actually needed by that wave's entries are
requested. Every public entry point takes a `max_credits` budget and stops
cleanly when it is exhausted.

CLV convention:
  clv = market_prob_at_close − market_prob_at_first_pick
  Positive → the market moved toward our pick after we took it (we beat the
  close). Negative → the market moved against us (the close beat us).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    NBA_SPORT, MLB_SPORT, NFL_SPORT, NHL_SPORT,
    IPL_SPORT, WNBA_SPORT, MLS_SPORT,
)
from src.data.odds_client import _get, american_to_prob, remove_vig
from src.state.shadow_log import (
    SHADOW_LOG_DIR, _load_shard, _save_shard_atomic,
)

logger = logging.getLogger(__name__)

# Sport label (as stored in shadow log entries) → Odds API sport key
_SPORT_KEYS: Dict[str, str] = {
    "NBA":  NBA_SPORT,
    "MLB":  MLB_SPORT,
    "NFL":  NFL_SPORT,
    "NHL":  NHL_SPORT,
    "IPL":  IPL_SPORT,
    "WNBA": WNBA_SPORT,
    "MLS":  MLS_SPORT,
}
try:  # WC config only exists June 2026+; degrade gracefully if removed
    from src.config import WC_SPORT
    _SPORT_KEYS["WC"] = WC_SPORT
except ImportError:
    pass

# Sports whose h2h market is 3-way (Home/Draw/Away)
_THREE_WAY_SPORTS = {"MLS", "WC"}

# market_type (shadow log) → Odds API market key
_MARKET_KEYS = {
    "Moneyline": "h2h",
    "Spread":    "spreads",
    "Total":     "totals",
    "Draw":      "h2h",
}

# Historical odds request cost: 10 credits per market per region (1 region: us)
_HIST_COST_PER_MARKET = 10
# Historical events request cost (used for commence_time repair)
_HIST_EVENTS_COST = 1
# Give up on an entry after this many failed match attempts (stops a
# permanently unmatchable entry from burning credits every night).
_MAX_FETCH_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Pick parsing helpers
# ---------------------------------------------------------------------------

_POINT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*$")


def _parse_pick_point(pick: str) -> Optional[float]:
    """Extract the trailing line point from a pick string.

    "Rockies +1.5" → 1.5 · "Over 8.5" → 8.5 · "Lakers -6.5" → -6.5
    Returns None when the pick has no numeric suffix (e.g. plain moneyline).
    """
    m = _POINT_RE.search((pick or "").strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _team_match(query: str, candidate: str) -> bool:
    """Fuzzy team-name match: substring either way, or ≥1 shared long token."""
    q, c = _norm(query), _norm(candidate)
    if not q or not c:
        return False
    if q in c or c in q:
        return True
    q_toks = {t for t in q.split() if len(t) >= 4}
    c_toks = {t for t in c.split() if len(t) >= 4}
    return bool(q_toks & c_toks)


def _split_game(game: str) -> Tuple[str, str]:
    """'AWAY @ HOME' → (away, home). Best-effort for non-standard strings."""
    parts = (game or "").split(" @ ")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", (game or "").strip()


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Historical API fetch
# ---------------------------------------------------------------------------

def _fetch_historical_snapshot(
    sport_key: str, snapshot_iso: str, market_keys: List[str]
) -> Optional[List[Dict]]:
    """One historical odds snapshot for a sport at (or just before) a timestamp.

    The API returns the latest archived snapshot at or before `date`, so
    requesting the game's commence_time yields the closing snapshot.
    """
    data = _get(
        f"/historical/sports/{sport_key}/odds",
        {
            "regions": "us",
            "markets": ",".join(sorted(set(market_keys))),
            "oddsFormat": "american",
            "date": snapshot_iso,
        },
    )
    if not data or not isinstance(data, dict):
        return None
    return data.get("data") or []


def _find_event(events: List[Dict], home: str, away: str,
                commence_time: str) -> Optional[Dict]:
    """Match a shadow entry's game to an event in the snapshot."""
    want_dt = _parse_iso(commence_time)
    for ev in events:
        eh, ea = ev.get("home_team", ""), ev.get("away_team", "")
        if not ((_team_match(home, eh) and _team_match(away, ea))
                or (_team_match(home, ea) and _team_match(away, eh))):
            continue
        ev_dt = _parse_iso(ev.get("commence_time", ""))
        if want_dt and ev_dt and abs((ev_dt - want_dt).total_seconds()) > 6 * 3600:
            continue
        return ev
    return None


# ---------------------------------------------------------------------------
# Closing probability per entry — consensus no-vig across all books,
# mirroring the morning pipeline's _consensus_probs* logic so CLV compares
# like with like.
# ---------------------------------------------------------------------------

def _closing_ml_prob(event: Dict, pick_side: str, sport: str) -> Optional[float]:
    """No-vig consensus closing probability for a moneyline (or Draw) pick."""
    three_way = sport in _THREE_WAY_SPORTS
    probs: List[float] = []
    for book in event.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "h2h":
                continue
            outcomes = mkt.get("outcomes", [])
            if three_way:
                if len(outcomes) < 3:
                    break
                raw = {o["name"]: american_to_prob(o["price"]) for o in outcomes}
                total = sum(raw.values())
                if total <= 0:
                    break
                for name, p in raw.items():
                    if _norm(name) == "draw" and _norm(pick_side) == "draw":
                        probs.append(p / total)
                    elif _norm(pick_side) != "draw" and _team_match(pick_side, name):
                        probs.append(p / total)
            else:
                if len(outcomes) < 2:
                    break
                raw = {o["name"]: american_to_prob(o["price"]) for o in outcomes}
                names = list(raw.keys())
                p0, p1 = remove_vig(raw[names[0]], raw[names[1]])
                novig = {names[0]: p0, names[1]: p1}
                for name, p in novig.items():
                    if _team_match(pick_side, name):
                        probs.append(p)
            break
    if not probs:
        return None
    return sum(probs) / len(probs)


def _closing_spread_prob(event: Dict, pick_side: str,
                         pick_point: Optional[float]) -> Optional[Tuple[float, Optional[float], bool]]:
    """No-vig consensus closing cover probability for a spread pick.

    Prefers books still offering the picked team at the SAME point as the
    morning pick; if none, falls back to all books with the same sign
    (mirrors _consensus_probs_for_spread) and flags the point drift.

    Returns (prob, close_point, point_differs) or None.
    """
    exact: List[float] = []
    same_sign: List[Tuple[float, float]] = []  # (prob, point)
    for book in event.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "spreads":
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) < 2:
                break
            picked = next((o for o in outcomes if _team_match(pick_side, o["name"])), None)
            other  = next((o for o in outcomes if o is not picked), None)
            if picked is None or other is None:
                break
            p_raw = american_to_prob(picked["price"])
            o_raw = american_to_prob(other["price"])
            p_nv, _ = remove_vig(p_raw, o_raw)
            book_point = picked.get("point")
            if pick_point is not None and book_point is not None:
                if abs(float(book_point) - pick_point) < 0.01:
                    exact.append(p_nv)
                elif (float(book_point) > 0) == (pick_point > 0):
                    same_sign.append((p_nv, float(book_point)))
            else:
                same_sign.append((p_nv, float(book_point) if book_point is not None else 0.0))
            break
    if exact:
        return sum(exact) / len(exact), pick_point, False
    if same_sign:
        probs = [p for p, _ in same_sign]
        points = [pt for _, pt in same_sign]
        return sum(probs) / len(probs), median(points), True
    return None


def _closing_total_prob(event: Dict, pick_side: str,
                        pick_point: Optional[float]) -> Optional[Tuple[float, Optional[float], bool]]:
    """No-vig consensus closing probability for a total (Over/Under) pick.

    Same exact-point-first / any-point-fallback approach as spreads. The
    display layer shifts integer lines by 0.5 ("Over 10" → "Over 9.5"), so
    exact-point matching uses a ±0.5 tolerance.

    Returns (prob, close_point, point_differs) or None.
    """
    side = "over" if "over" in _norm(pick_side) else "under"
    exact: List[float] = []
    fallback: List[Tuple[float, float]] = []
    for book in event.get("bookmakers", []):
        for mkt in book.get("markets", []):
            if mkt.get("key") != "totals":
                continue
            outcomes = mkt.get("outcomes", [])
            if len(outcomes) < 2:
                break
            o_side  = next((o for o in outcomes if _norm(o["name"]) == side), None)
            o_other = next((o for o in outcomes if o is not o_side), None)
            if o_side is None or o_other is None:
                break
            p_raw = american_to_prob(o_side["price"])
            q_raw = american_to_prob(o_other["price"])
            p_nv, _ = remove_vig(p_raw, q_raw)
            book_point = o_side.get("point")
            if pick_point is not None and book_point is not None \
                    and abs(float(book_point) - pick_point) <= 0.51:
                exact.append(p_nv)
            else:
                fallback.append((p_nv, float(book_point) if book_point is not None else 0.0))
            break
    if exact:
        return sum(exact) / len(exact), pick_point, False
    if fallback:
        probs = [p for p, _ in fallback]
        points = [pt for _, pt in fallback]
        return sum(probs) / len(probs), median(points), True
    return None


def _closing_prob_for_entry(event: Dict, entry: Dict) -> Optional[Dict[str, Any]]:
    """Compute the closing probability for one shadow log entry.

    Returns {"prob": float, "close_point": float|None, "point_differs": bool}
    or None when the pick can't be matched in the snapshot.
    """
    market_type = entry.get("market_type", "")
    pick_side   = entry.get("pick_side", "")
    pick_point  = _parse_pick_point(entry.get("pick", ""))

    if market_type in ("Moneyline", "Draw"):
        prob = _closing_ml_prob(event, pick_side, (entry.get("sport") or "").upper())
        if prob is None:
            return None
        return {"prob": prob, "close_point": None, "point_differs": False}

    if market_type == "Spread":
        res = _closing_spread_prob(event, pick_side, pick_point)
        if res is None:
            return None
        prob, close_point, differs = res
        return {"prob": prob, "close_point": close_point, "point_differs": differs}

    if market_type == "Total":
        res = _closing_total_prob(event, pick_side, pick_point)
        if res is None:
            return None
        prob, close_point, differs = res
        return {"prob": prob, "close_point": close_point, "point_differs": differs}

    return None


# ---------------------------------------------------------------------------
# Shadow log scan + stamp (the self-healing core)
# ---------------------------------------------------------------------------

def _wave_key(sport: str, commence_time: str) -> Tuple[str, str]:
    """Group entries into (sport, 5-minute wave) so one snapshot serves all."""
    dt = _parse_iso(commence_time)
    if dt is None:
        return sport, commence_time
    dt = dt - timedelta(minutes=dt.minute % 5, seconds=dt.second,
                        microseconds=dt.microsecond)
    return sport, dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def update_shadow_log_clv(
    *,
    max_credits: int = 600,
    lookback_days: int = 7,
    since: Optional[str] = None,
) -> Dict[str, int]:
    """Scan the shadow log for entries missing `market_prob_at_close` whose
    game has started, fetch their closing snapshots from the historical
    archive, and stamp `market_prob_at_close` + `clv` onto each entry.

    Self-healing and idempotent: already-stamped entries are skipped, so a
    night where this never ran is automatically repaired on the next pass.
    `since` (YYYY-MM-DD) overrides lookback_days for deep backfills.

    Never raises — CLV capture must not block anything. Returns a summary:
    {"stamped": n, "credits_spent": n, "waves": n, "unmatched": n}.
    """
    summary = {"stamped": 0, "credits_spent": 0, "waves": 0, "unmatched": 0}
    try:
        if not SHADOW_LOG_DIR.exists():
            return summary

        now = datetime.now(timezone.utc)
        if since:
            floor = since
        else:
            floor = (now - timedelta(days=lookback_days)).date().isoformat()

        # ── Collect candidate entries grouped by (sport, wave) ───────────────
        loaded_shards: Dict[Any, Dict] = {}
        waves: Dict[Tuple[str, str], List[Tuple[Any, Dict]]] = defaultdict(list)

        for shard_path in sorted(SHADOW_LOG_DIR.glob("*.json")):
            shard = _load_shard(shard_path)
            loaded_shards[shard_path] = shard
            for entry in shard.get("entries", {}).values():
                if entry.get("market_prob_at_close") is not None:
                    continue
                if entry.get("clv_fetch_attempts", 0) >= _MAX_FETCH_ATTEMPTS:
                    continue
                if (entry.get("date") or "") < floor:
                    continue
                sport = (entry.get("sport") or "").upper()
                if sport not in _SPORT_KEYS:
                    continue
                ct = entry.get("commence_time", "")
                ct_dt = _parse_iso(ct)
                if ct_dt is None or ct_dt > now:
                    continue  # no usable start time, or game not started yet
                if entry.get("market_type") not in _MARKET_KEYS:
                    continue
                waves[_wave_key(sport, ct)].append((shard_path, entry))

        if not waves:
            return summary

        # Most recent waves first — freshest data wins when budget runs out
        ordered = sorted(waves.items(), key=lambda kv: kv[0][1], reverse=True)
        logger.info(
            f"CLV capture: {sum(len(v) for v in waves.values())} entries missing "
            f"closing line across {len(waves)} wave(s) (floor {floor})"
        )

        modified: set = set()
        for (sport, wave_iso), items in ordered:
            market_keys = sorted({
                _MARKET_KEYS[e.get("market_type")] for _, e in items
            })
            cost = _HIST_COST_PER_MARKET * len(market_keys)
            if summary["credits_spent"] + cost > max_credits:
                logger.info(
                    f"CLV capture: credit budget reached "
                    f"({summary['credits_spent']}/{max_credits}) — stopping; "
                    f"remaining waves self-heal on the next run"
                )
                break

            events = _fetch_historical_snapshot(_SPORT_KEYS[sport], wave_iso, market_keys)
            summary["credits_spent"] += cost
            summary["waves"] += 1
            if events is None:
                continue  # API error — attempts NOT incremented (retry next run)

            now_iso = datetime.now(timezone.utc).isoformat()
            for shard_path, entry in items:
                away, home = _split_game(entry.get("game", ""))
                event = _find_event(events, home, away, entry.get("commence_time", ""))
                result = _closing_prob_for_entry(event, entry) if event else None
                if result is None:
                    entry["clv_fetch_attempts"] = entry.get("clv_fetch_attempts", 0) + 1
                    summary["unmatched"] += 1
                    modified.add(shard_path)
                    continue
                open_prob = entry.get("market_prob_at_first_pick")
                entry["market_prob_at_close"] = round(result["prob"], 4)
                entry["clv"] = (
                    round(result["prob"] - float(open_prob), 4)
                    if open_prob is not None else None
                )
                entry["close_point"]         = result["close_point"]
                entry["close_point_differs"] = bool(result["point_differs"])
                entry["clv_captured_at"]     = now_iso
                summary["stamped"] += 1
                modified.add(shard_path)

        for shard_path in modified:
            _save_shard_atomic(shard_path, loaded_shards[shard_path])

        logger.info(
            f"CLV capture: {summary['stamped']} entries stamped, "
            f"{summary['unmatched']} unmatched, ~{summary['credits_spent']} credits "
            f"across {summary['waves']} snapshot(s)"
        )
        return summary

    except Exception as e:
        logger.error(f"CLV capture failed (non-fatal): {e}")
        return summary


# ---------------------------------------------------------------------------
# Commence-time repair (for old backfilled entries that lack start times)
# ---------------------------------------------------------------------------

def repair_missing_commence_times(*, max_credits: int = 50,
                                  since: str = "2026-04-01") -> int:
    """Fill missing `commence_time` on shadow log entries via the historical
    events endpoint (1 credit per (sport, date) request). Needed only for
    entries imported by the original backfill — new entries always carry it.

    Returns the number of entries repaired. Never raises.
    """
    repaired = 0
    spent = 0
    try:
        if not SHADOW_LOG_DIR.exists():
            return 0

        loaded_shards: Dict[Any, Dict] = {}
        groups: Dict[Tuple[str, str], List[Tuple[Any, Dict]]] = defaultdict(list)

        for shard_path in sorted(SHADOW_LOG_DIR.glob("*.json")):
            shard = _load_shard(shard_path)
            loaded_shards[shard_path] = shard
            for entry in shard.get("entries", {}).values():
                if entry.get("commence_time"):
                    continue
                if entry.get("ct_fetch_attempts", 0) >= _MAX_FETCH_ATTEMPTS:
                    continue
                d = entry.get("date") or ""
                if d < since:
                    continue
                sport = (entry.get("sport") or "").upper()
                if sport not in _SPORT_KEYS:
                    continue
                groups[(sport, d)].append((shard_path, entry))

        if not groups:
            return 0

        modified: set = set()
        for (sport, d), items in sorted(groups.items(), reverse=True):
            if spent + _HIST_EVENTS_COST > max_credits:
                break
            # Snapshot at 18:00 UTC (morning Pacific) lists that day's upcoming
            # games with their start times — same vantage point as the model run.
            data = _get(
                f"/historical/sports/{_SPORT_KEYS[sport]}/events",
                {"date": f"{d}T18:00:00Z"},
            )
            spent += _HIST_EVENTS_COST
            events = (data or {}).get("data") or []
            for shard_path, entry in items:
                away, home = _split_game(entry.get("game", ""))
                ev = next(
                    (e for e in events
                     if (_team_match(home, e.get("home_team", ""))
                         and _team_match(away, e.get("away_team", "")))
                     or (_team_match(home, e.get("away_team", ""))
                         and _team_match(away, e.get("home_team", "")))),
                    None,
                )
                if ev and ev.get("commence_time"):
                    entry["commence_time"] = ev["commence_time"]
                    repaired += 1
                else:
                    entry["ct_fetch_attempts"] = entry.get("ct_fetch_attempts", 0) + 1
                modified.add(shard_path)

        for shard_path in modified:
            _save_shard_atomic(shard_path, loaded_shards[shard_path])

        if repaired:
            logger.info(f"CLV repair: {repaired} commence_time(s) filled (~{spent} credits)")
        return repaired

    except Exception as e:
        logger.error(f"Commence-time repair failed (non-fatal): {e}")
        return 0


# ---------------------------------------------------------------------------
# CLV lookup for the results snapshot (consumed by the nightly debrief)
# ---------------------------------------------------------------------------

def clv_lookup_for_date(run_date: date) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Build {(game, bet_type, pick): {clv, market_prob_at_close, ...}} for one
    run date, read straight from the shadow log. Used by results_snapshot to
    attach CLV to each resolved pick. Never raises — returns {} on failure.
    """
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    try:
        shard = _load_shard(SHADOW_LOG_DIR / f"{run_date.year:04d}-{run_date.month:02d}.json")
        date_str = run_date.isoformat()
        for entry in shard.get("entries", {}).values():
            if entry.get("date") != date_str:
                continue
            if entry.get("market_prob_at_close") is None:
                continue
            key = (entry.get("game", ""), entry.get("bet_type", ""), entry.get("pick", ""))
            out[key] = {
                "market_prob_at_open":  entry.get("market_prob_at_first_pick"),
                "market_prob_at_close": entry.get("market_prob_at_close"),
                "clv":                  entry.get("clv"),
                "close_point_differs":  entry.get("close_point_differs", False),
            }
    except Exception as e:
        logger.warning(f"CLV lookup failed (non-fatal): {e}")
    return out
