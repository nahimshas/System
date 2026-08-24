"""
Post-game execution settlement.

For each logged pick, replay Kalshi's own candlesticks between pick time and
first pitch and ask two things:

  1. WOULD A RESTING BID HAVE FILLED?  If the market traded at or below our
     posted price, a limit order at that price would have been reached. This is
     a proxy, not a certainty (queue position matters), so it is an OPTIMISTIC
     upper bound on fill rate — read the results as "at best this often".

  2. WHERE DID IT CLOSE?  Closing price is the fair-value benchmark, exactly
     like the CLV we already track. Comparing clv_if_bid against clv_if_ask
     gives the saving; comparing fill-weighted CLV against the unconditional
     close exposes adverse selection — if we only fill when the price is about
     to fall, clv_if_bid on FILLED picks will be systematically worse than on
     all picks.

Never raises. Kalshi data is free, so there is no credit budget to respect.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.data.kalshi import fetch_candles, enabled
from src.state.execution_log import FEES, load_entries, make_key, update_entry

logger = logging.getLogger(__name__)


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


def settle_executions(since: str = "0000-00-00", max_rows: int = 400) -> Dict[str, int]:
    """Fill in close_price / bid_would_fill / clv_* for started games."""
    summary = {"settled": 0, "skipped": 0, "no_data": 0}
    if not enabled():
        return summary
    try:
        now = datetime.now(timezone.utc)
        pending = [e for e in load_entries(since)
                   if e.get("close_price") is None and e.get("kalshi_ticker")]
        for e in pending[:max_rows]:
            ct = _parse_iso(e.get("commence_time", ""))
            if ct is None or ct > now:
                summary["skipped"] += 1          # game hasn't started yet
                continue
            start = int(ct.timestamp()) - 12 * 3600
            end = int(ct.timestamp())
            candles = fetch_candles(e["kalshi_series"], e["kalshi_ticker"], start, end)
            if not candles:
                summary["no_data"] += 1
                continue

            side = e.get("kalshi_side", "yes")
            lows, closes = [], []
            for c in candles:
                pr = c.get("price") or {}
                lo, hi, cl = _f(pr.get("low_dollars")), _f(pr.get("high_dollars")), _f(pr.get("close_dollars"))
                # Buying NO means the mirror book: our "low" is 1 - their high.
                if side == "no":
                    lo, hi, cl = (1 - hi if hi is not None else None,
                                  1 - lo if lo is not None else None,
                                  1 - cl if cl is not None else None)
                if lo is not None:
                    lows.append(lo)
                if cl is not None:
                    closes.append(cl)
            if not closes:
                summary["no_data"] += 1
                continue

            close_price = closes[-1]
            bid = e.get("bid_at_pick")
            ask = e.get("ask_at_pick")
            would_fill = (min(lows) <= bid) if (lows and bid is not None) else None

            # CLV in the same convention as the rest of the system: how much
            # better than what we paid the market ended up. All-in cost includes
            # fees, because that is the money that actually leaves the account.
            clv_bid = round(close_price - (bid + FEES), 4) if bid is not None else None
            clv_ask = round(close_price - (ask + FEES), 4) if ask is not None else None

            key = make_key(e["date"], e.get("game", ""), e.get("bet_type", ""),
                           e.get("pick", ""))
            if update_entry(e["date"], key, {
                "close_price": round(close_price, 4),
                "bid_would_fill": would_fill,
                "clv_if_bid": clv_bid,
                "clv_if_ask": clv_ask,
                "settled_at": now.isoformat(),
            }):
                summary["settled"] += 1
        logger.info(f"execution settle: {summary}")
        return summary
    except Exception as e:
        logger.error(f"execution settle failed (non-fatal): {e}")
        return summary
