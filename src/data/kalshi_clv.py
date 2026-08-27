"""
CLV from Kalshi — the market we actually trade in.

WHY THIS EXISTS
  1. F5 has NO closing line at all today. The Odds API historical endpoint does
     not serve *_1st_5_innings at any price, so F5 CLV was switched off Aug 10.
     Kalshi has it, free, and retains settled markets with full candlestick
     history back past our first F5 pick — so it is BACKFILLABLE.
  2. Odds API historical CLV is ~54% of daily credit usage (~190 of ~353).
  3. Kalshi's close is the closing price of the book we buy in, not a
     sportsbook we never touch.

⚠️ APPLES-TO-APPLES IS THE WHOLE BALLGAME. The existing `clv` field is
sportsbook no-vig close minus sportsbook no-vig open. Kalshi CLV is Kalshi mid
at close minus Kalshi mid at pick. Mixing the two ends — a sportsbook open
against a Kalshi close — would fold the venue difference into the signal and
corrupt it. So Kalshi CLV is SELF-CONSISTENT (both ends from Kalshi) and is
stored in its OWN fields (kalshi_*). Nothing existing is overwritten, which
also means the calibration engine and the CLV governor keep reading exactly
what they read before until we deliberately switch them over.

Mid, not ask, is the fair-value benchmark: mid overround is ~0% while buying at
the ask costs ~1%. Execution cost is measured separately by the execution log.

Never raises. Kalshi data is free, so there is no credit budget here.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.data.kalshi import (SERIES, enabled, event_date, fetch_candles,
                             fetch_open_markets, resolve_pick, _get)

logger = logging.getLogger(__name__)

# sport -> {our market_type: kalshi series}. MLB is what we bet most; the rest
# are wired so watchlist sports get CLV too (coverage verified Aug 25 2026).
# ⚠️ A sport is only USABLE here if kalshi.TEAM_TO_KALSHI can map its team
# names to Kalshi's tokens. That map is MLB-only today, so MLB is the only
# sport actually resolved — the rest are listed because their series exist and
# were verified present, and they light up the moment a team map is added.
# Entries for unmapped sports are SKIPPED WITHOUT burning a retry attempt:
# a missing feature must not permanently mark rows unmatchable.
SPORT_SERIES: Dict[str, Dict[str, str]] = {
    "MLB": SERIES,
    "NBA":  {"Moneyline": "KXNBAGAME",  "Spread": "KXNBASPREAD",  "Total": "KXNBATOTAL"},
    "NFL":  {"Moneyline": "KXNFLGAME",  "Spread": "KXNFLSPREAD",  "Total": "KXNFLTOTAL"},
    "NHL":  {"Moneyline": "KXNHLGAME",  "Spread": "KXNHLSPREAD",  "Total": "KXNHLTOTAL"},
    "WNBA": {"Moneyline": "KXWNBAGAME", "Spread": "KXWNBASPREAD", "Total": "KXWNBATOTAL"},
    "MLS":  {"Moneyline": "KXMLSGAME",  "Spread": "KXMLSSPREAD",  "Total": "KXMLSTOTAL"},
    "LIGAMX": {"Moneyline": "KXLIGAMXGAME", "Spread": "KXLIGAMXSPREAD",
               "Total": "KXLIGAMXTOTAL"},
    # IPL deliberately absent: Kalshi lists no per-match IPL markets.
}

MAX_ATTEMPTS = 3
# Averaging window (seconds) at each CLV endpoint — see prices_at().
WINDOW_SECONDS = 900


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_markets(series: str, status: str, max_pages: int = 60) -> List[Dict]:
    """Markets for a series at a given status ('open' or 'settled').

    max_pages must be generous: TOTAL and SPREAD series list a LADDER of
    strikes (~8 markets per game, ~120/day), so 200-row pages run out after
    about three weeks of history. At 12 pages the backfill silently returned
    "0 markets" for older dates and marked 385 resolvable MLB rows unmatched.
    """
    out: List[Dict] = []
    cursor = None
    for _ in range(max_pages):
        params = {"series_ticker": series, "status": status, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        if not d:
            break
        out.extend(d.get("markets", []))
        cursor = d.get("cursor")
        if not cursor:
            break
    return out


def build_index(series_list: List[str], dates: set,
                max_pages: int = 60) -> Dict[str, List[Dict]]:
    """Open + settled markets for each series, filtered to the dates we need.

    Past games live under 'settled', current ones under 'open', and a backfill
    spans both — so we always union them.
    """
    idx: Dict[str, List[Dict]] = {}
    for s in series_list:
        markets = (fetch_markets(s, "open", max_pages) +
                   fetch_markets(s, "settled", max_pages))
        idx[s] = [m for m in markets if event_date(m) in dates]
        oldest = min((event_date(m) for m in idx[s] if event_date(m)), default=None)
        logger.info(f"Kalshi index {s}: {len(idx[s])} markets across {len(dates)} "
                    f"date(s), oldest {oldest}")
        if idx[s] and dates and oldest and oldest > min(dates):
            logger.warning(f"Kalshi index {s}: history starts {oldest} but "
                           f"oldest requested date is {min(dates)} — earlier "
                           f"entries cannot be backfilled from this series")
    return idx


def _mid_from_candle(candle: Dict, side: str) -> Optional[float]:
    """Mid price from a candle's closing bid/ask. Mirrors for the NO side."""
    yb = _f((candle.get("yes_bid") or {}).get("close_dollars"))
    ya = _f((candle.get("yes_ask") or {}).get("close_dollars"))
    if yb is None or ya is None or yb <= 0 or ya <= 0:
        # Fall back to traded price when the book is not quoted in that candle.
        p = _f((candle.get("price") or {}).get("close_dollars"))
        if p is None or p <= 0:
            return None
        yb = ya = p
    mid = (yb + ya) / 2.0
    if side == "no":
        mid = 1.0 - mid
    if not (0.0 < mid < 1.0):
        return None
    return round(mid, 4)


def prices_at(series: str, ticker: str, side: str,
              pick_dt: datetime, close_dt: datetime) -> Tuple[Optional[float], Optional[float]]:
    """(mid at pick time, mid at close). Both ends from Kalshi — never mix
    venues across the two ends of a CLV measurement."""
    start = int(pick_dt.timestamp()) - 3600
    end = int(close_dt.timestamp())
    if end <= start:
        return None, None
    # 1-MINUTE candles. Hourly is far too coarse: picks are locked ~1-8h before
    # first pitch, so an hourly series can hold a single candle and CLV comes
    # out an artificial 0.0000. (Kalshi supports 1, 60 and 1440 only — 5 is a
    # 400.) Falls back to hourly if the fine series is unavailable.
    candles = fetch_candles(series, ticker, start, end, period_interval=1)
    if not candles:
        candles = fetch_candles(series, ticker, start, end, period_interval=60)
    if not candles:
        return None, None

    pick_ts = int(pick_dt.timestamp())
    # AVERAGE A WINDOW OF MIDS, don't sample a single candle.
    #
    # Kalshi trades in 1-cent ticks, so a single mid can only ever land on a
    # half-cent and CLV comes out quantised to 0.5pp — LARGER than the ~0.4pp
    # effect we are trying to measure. (Measured Aug 27 2026: 2022 of 2022
    # kalshi_clv values were exact tick multiples, versus 78% continuous from
    # the sportsbook feed.) Averaging the mids over a short window around each
    # endpoint recovers sub-tick resolution and damps single-print noise, the
    # same way a closing line is normally taken over a window rather than an
    # instant.
    def _window_mean(centre_ts: int, before: int, after: int) -> Optional[float]:
        vals = []
        for c in candles:
            ts = c.get("end_period_ts")
            if ts is None or not (centre_ts - before <= ts <= centre_ts + after):
                continue
            m = _mid_from_candle(c, side)
            if m is not None:
                vals.append(m)
        return round(sum(vals) / len(vals), 5) if vals else None

    # At pick: look forward, the pick is taken at that moment.
    at_pick = _window_mean(pick_ts, 0, WINDOW_SECONDS)
    # At close: look back from first pitch — there is no "after".
    at_close = _window_mean(end, WINDOW_SECONDS, 0)

    # Fall back to single candles when a window is empty (thin markets).
    if at_pick is None:
        at_pick = next((m for c in candles
                        if (m := _mid_from_candle(c, side)) is not None
                        and (c.get("end_period_ts") or 0) >= pick_ts), None)
    if at_pick is None:
        at_pick = next((m for c in candles
                        if (m := _mid_from_candle(c, side)) is not None), None)
    if at_close is None:
        last = [m for c in candles
                if (c.get("end_period_ts") or 0) <= end
                and (m := _mid_from_candle(c, side)) is not None]
        at_close = last[-1] if last else None
    return at_pick, at_close


def _teams_mappable(game: str) -> bool:
    """Both teams must map to Kalshi tokens or the market can never resolve.

    Checked up front so unmapped sports are skipped cleanly rather than
    counted as failures — otherwise WNBA/MLS rows accumulate retry attempts
    against a feature that simply is not built yet, and would stay poisoned
    (attempts >= MAX_ATTEMPTS) even after a team map is added.
    """
    from src.data.kalshi import _team_token
    if " @ " not in (game or ""):
        return False
    away, home = game.split(" @ ", 1)
    return bool(_team_token(away.strip()) and _team_token(home.strip()))


def _entry_to_pick(e: Dict) -> Dict[str, Any]:
    """Shadow-log entry -> the shape resolve_pick() expects."""
    game = e.get("game", "") or ""
    away, home = ("", "")
    if " @ " in game:
        away, home = game.split(" @ ", 1)
    # `pick` always carries the line ("Over 6.5", "Chicago White Sox +1.5");
    # `pick_side` sometimes strips it to a bare side ("over", team name), which
    # leaves the resolver unable to pick a strike off Kalshi's ladder.
    return {"bet_type": e.get("market_type"), "pick": e.get("pick") or e.get("pick_side"),
            "home_team": home.strip(), "away_team": away.strip()}


def update_shadow_log_kalshi_clv(since: str = "0000-00-00",
                                 max_entries: int = 1500,
                                 sports: Optional[List[str]] = None,
                                 recompute: bool = False) -> Dict[str, int]:
    """Stamp kalshi_prob_at_pick / kalshi_prob_at_close / kalshi_clv.

    Additive only — existing clv / market_prob_at_close are never touched.
    Idempotent: entries that already carry kalshi_clv are skipped, so this is
    safe to run repeatedly and safe to interrupt.
    """
    summary = {"stamped": 0, "unmatched": 0, "no_candles": 0, "skipped": 0,
               "unmapped": 0}
    if not enabled():
        return summary
    try:
        from src.state.shadow_log import SHADOW_LOG_DIR, _load_shard, _save_shard_atomic

        now = datetime.now(timezone.utc)
        shards = {}
        todo: List[Tuple[Any, Dict]] = []
        for path in sorted(SHADOW_LOG_DIR.glob("*.json")):
            shard = _load_shard(path)
            shards[path] = shard
            for e in shard.get("entries", {}).values():
                if e.get("kalshi_clv") is not None and not recompute:
                    continue
                if (e.get("kalshi_clv_attempts") or 0) >= MAX_ATTEMPTS and not recompute:
                    continue
                if (e.get("date") or "") < since:
                    continue
                sport = (e.get("sport") or "").upper()
                if sports and sport not in sports:
                    continue
                mt = e.get("market_type")
                if sport not in SPORT_SERIES or mt not in SPORT_SERIES[sport]:
                    continue
                if not _teams_mappable(e.get("game", "")):
                    summary["unmapped"] += 1
                    continue
                ct = _parse_iso(e.get("commence_time", ""))
                if ct is None or ct > now:
                    summary["skipped"] += 1
                    continue
                todo.append((path, e))
        if not todo:
            return summary
        todo = todo[:max_entries]

        dates = {e.get("date") for _, e in todo if e.get("date")}
        series_needed = sorted({SPORT_SERIES[(e.get("sport") or "").upper()][e["market_type"]]
                                for _, e in todo})
        # ±0.5 F5 spreads resolve against the F5 winner book.
        if any(e["market_type"] == "F5 Spread" for _, e in todo):
            series_needed = sorted(set(series_needed) | {"KXMLBF5"})
        logger.info(f"Kalshi CLV: {len(todo)} entries, {len(dates)} date(s), "
                    f"series {series_needed}")
        idx = build_index(series_needed, dates)

        modified = set()
        for path, e in todo:
            pick = _entry_to_pick(e)
            r = resolve_pick(pick, idx, e.get("date", ""), require_quote=False)
            if not r:
                e["kalshi_clv_attempts"] = (e.get("kalshi_clv_attempts") or 0) + 1
                summary["unmatched"] += 1
                modified.add(path)
                continue
            pick_dt = _parse_iso(e.get("first_seen_at") or "") or (
                _parse_iso(e.get("commence_time", "")) - timedelta(hours=8))
            close_dt = _parse_iso(e.get("commence_time", ""))
            at_pick, at_close = prices_at(r["series"], r["ticker"], r["side"],
                                          pick_dt, close_dt)
            if at_pick is None or at_close is None:
                e["kalshi_clv_attempts"] = (e.get("kalshi_clv_attempts") or 0) + 1
                summary["no_candles"] += 1
                modified.add(path)
                continue
            e["kalshi_ticker"] = r["ticker"]
            e["kalshi_side"] = r["side"]
            e["kalshi_prob_at_pick"] = at_pick
            e["kalshi_prob_at_close"] = at_close
            e["kalshi_clv"] = round(at_close - at_pick, 4)
            e["kalshi_clv_captured_at"] = now.isoformat()
            summary["stamped"] += 1
            modified.add(path)

            # Checkpoint periodically. A full backfill is hundreds of throttled
            # requests and can run for many minutes; writing only at the end
            # meant an interrupt threw away every stamp. Writes are atomic and
            # the pass is idempotent, so a partial flush is always safe.
            done = summary["stamped"] + summary["unmatched"] + summary["no_candles"]
            if done % 50 == 0:
                for _p in list(modified):
                    _save_shard_atomic(_p, shards[_p])
                logger.info(f"Kalshi CLV: checkpoint at {done} processed "
                            f"({summary['stamped']} stamped)")

        for path in modified:
            _save_shard_atomic(path, shards[path])
        logger.info(f"Kalshi CLV: {summary}")
        return summary
    except Exception as ex:
        logger.error(f"Kalshi CLV update failed (non-fatal): {ex}")
        return summary

def _decision_entry_to_pick(e: Dict) -> Optional[Dict[str, Any]]:
    """Decision-log entry -> the shape resolve_pick() expects.

    The decision log stores `side` and `line` SEPARATELY, unlike the shadow log
    whose `pick` already carries the line. Reconstruct the pick text so the
    resolver can choose the right rung off Kalshi's strike ladder.
    """
    game = e.get("game", "") or ""
    if " @ " not in game:
        return None
    away, home = [x.strip() for x in game.split(" @ ", 1)]
    mt = e.get("market_type") or ""
    side = str(e.get("side") or "")
    line = e.get("line")

    if mt in ("Moneyline", "F5 Moneyline"):
        pick = side
    elif mt in ("Spread", "F5 Spread"):
        if line is None:
            return None
        pick = f"{side} {float(line):+.1f}"
    elif mt in ("Total", "F5 Total"):
        if line is None:
            return None
        pick = f"{'Over' if side.lower().startswith('o') else 'Under'} {float(line)}"
    else:
        return None          # F5 Tie has no reliable draw price — skip
    return {"bet_type": mt, "pick": pick, "home_team": home, "away_team": away}


def update_decision_log_kalshi_clv(since: str = "0000-00-00",
                                   max_entries: int = 2000,
                                   recompute: bool = False) -> Dict[str, int]:
    """Stamp kalshi_clv onto DECISION-LOG candidates — including rejected sides.

    This is the half that makes the archive worth keeping: CLV on picks the
    model REJECTED, which is how we test whether our selection adds anything.
    It used to ride along with the Odds API pass; that feed is off, so without
    this the candidate archive silently stops accruing CLV (183 of 719 rows in
    the week the switch landed).

    Additive and idempotent, exactly like the shadow-log pass.
    """
    summary = {"stamped": 0, "unmatched": 0, "no_candles": 0, "skipped": 0,
               "unmapped": 0}
    if not enabled():
        return summary
    try:
        from src.state import decision_log as _dl

        now = datetime.now(timezone.utc)
        shards, todo = {}, []
        for path in sorted(_dl.DECISION_LOG_DIR.glob("*.json")):
            shard = _dl._load_shard(path)
            shards[path] = shard
            for e in shard.get("entries", {}).values():
                if e.get("kalshi_clv") is not None and not recompute:
                    continue
                if (e.get("kalshi_clv_attempts") or 0) >= MAX_ATTEMPTS and not recompute:
                    continue
                if (e.get("date") or "") < since:
                    continue
                sport = (e.get("sport") or "").upper()
                mt = e.get("market_type")
                if sport not in SPORT_SERIES or mt not in SPORT_SERIES[sport]:
                    continue
                if not _teams_mappable(e.get("game", "")):
                    summary["unmapped"] += 1
                    continue
                ct = _parse_iso(e.get("commence_time", ""))
                if ct is None or ct > now:
                    summary["skipped"] += 1
                    continue
                todo.append((path, e))
        if not todo:
            return summary
        todo = todo[:max_entries]

        dates = {e.get("date") for _, e in todo if e.get("date")}
        series = sorted({SPORT_SERIES[(e.get("sport") or "").upper()][e["market_type"]]
                         for _, e in todo})
        if any(e["market_type"] == "F5 Spread" for _, e in todo):
            series = sorted(set(series) | {"KXMLBF5"})
        logger.info(f"Kalshi decision CLV: {len(todo)} candidates, {len(dates)} date(s)")
        idx = build_index(series, dates)

        modified = set()
        for i, (path, e) in enumerate(todo, 1):
            pick = _decision_entry_to_pick(e)
            r = resolve_pick(pick, idx, e.get("date", ""), require_quote=False) if pick else None
            if not r:
                e["kalshi_clv_attempts"] = (e.get("kalshi_clv_attempts") or 0) + 1
                summary["unmatched"] += 1
                modified.add(path)
                continue
            pick_dt = _parse_iso(e.get("first_seen_at") or "") or (
                _parse_iso(e.get("commence_time", "")) - timedelta(hours=8))
            close_dt = _parse_iso(e.get("commence_time", ""))
            at_pick, at_close = prices_at(r["series"], r["ticker"], r["side"],
                                          pick_dt, close_dt)
            if at_pick is None or at_close is None:
                e["kalshi_clv_attempts"] = (e.get("kalshi_clv_attempts") or 0) + 1
                summary["no_candles"] += 1
                modified.add(path)
                continue
            e["kalshi_ticker"] = r["ticker"]
            e["kalshi_prob_at_pick"] = at_pick
            e["kalshi_prob_at_close"] = at_close
            e["kalshi_clv"] = round(at_close - at_pick, 5)
            e["kalshi_clv_captured_at"] = now.isoformat()
            summary["stamped"] += 1
            modified.add(path)
            if i % 50 == 0:
                for _p in list(modified):
                    _dl._save_shard_atomic(_p, shards[_p])

        for path in modified:
            _dl._save_shard_atomic(path, shards[path])
        logger.info(f"Kalshi decision CLV: {summary}")
        return summary
    except Exception as ex:
        logger.error(f"Kalshi decision CLV failed (non-fatal): {ex}")
        return summary
