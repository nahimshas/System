"""
Execution log — what we WOULD have paid, and what a resting bid would have got.

Separate from the shadow log (calibration) and the decision log (model
analysis). This one is purely about PRICE: for every pick, the Kalshi book at
pick time, and afterwards whether a limit order at the bid would have filled
and where the market closed.

It exists to answer one question: our measured edge is ~0 and we pay ask +
$0.02 (~2pp over fair on a coin flip), so does buying at the bid instead turn
a losing card into a break-even one — or does adverse selection take the
saving back? Resting at the bid fills preferentially when the price is about to
move against you, so the quoted spread OVERSTATES the real saving. That gap is
what this measures.

Same safety patterns as the other logs: month-sharded, stable keys, idempotent,
atomic writes, never raises.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXECUTION_LOG_DIR = Path("state/execution_log")
SCHEMA_VERSION = 1
FEES = 0.02          # Robinhood commission ($0.01) + Kalshi exchange fee ($0.01)


def _shard_path(date_str: str) -> Path:
    return EXECUTION_LOG_DIR / f"{date_str[:7]}.json"


def _load_shard(path: Path) -> Dict:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"execution log unreadable ({path}): {e}")
    return {"schema_version": SCHEMA_VERSION, "month": path.stem, "entries": {}}


def _save_atomic(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def make_key(date_str: str, game: str, bet_type: str, pick: str) -> str:
    return f"{date_str}|{game}|{bet_type}|{pick}"


def record_snapshot(date_str: str, rows: List[Dict[str, Any]]) -> int:
    """Store the Kalshi book for each matched pick. Idempotent per key.

    Prices are captured ONCE per pick (first write wins for the at-pick book),
    mirroring market_prob_at_first_pick in the shadow log — a later re-run must
    not overwrite the price that was actually available when the card was set.
    """
    if not rows:
        return 0
    try:
        path = _shard_path(date_str)
        shard = _load_shard(path)
        entries = shard.setdefault("entries", {})
        n = 0
        for row in rows:
            p, k = row.get("pick") or {}, row.get("kalshi") or {}
            if not p or not k:
                continue
            key = make_key(date_str, p.get("game", ""), p.get("bet_type", ""),
                           p.get("pick", ""))
            e = entries.get(key)
            if e is None:
                e = {
                    "schema_version": SCHEMA_VERSION,
                    "date": date_str,
                    "game": p.get("game", ""),
                    "sport": p.get("sport", ""),
                    "bet_type": p.get("bet_type", ""),
                    "pick": p.get("pick", ""),
                    "commence_time": p.get("commence_time", ""),
                    "in_budget": bool(p.get("_in_budget")),
                    # our view
                    "our_market_prob": p.get("market_prob_pct"),
                    "our_model_prob": p.get("model_prob_pct"),
                    # the book we would actually have bought
                    "kalshi_ticker": k.get("ticker"),
                    "kalshi_series": k.get("series"),
                    "kalshi_side": k.get("side"),
                    "bid_at_pick": k.get("bid"),
                    "ask_at_pick": k.get("ask"),
                    "mid_at_pick": k.get("mid"),
                    "open_interest_at_pick": k.get("open_interest"),
                    # filled in after the game
                    "close_price": None,
                    "bid_would_fill": None,
                    "clv_if_bid": None,
                    "clv_if_ask": None,
                    "settled_at": None,
                }
                entries[key] = e
                n += 1
            else:
                # Refresh only the live-moving fields; never the at-pick book.
                e["open_interest_latest"] = k.get("open_interest")
        shard["entries"] = entries
        _save_atomic(path, shard)
        logger.info(f"execution log: {n} new row(s) for {date_str} "
                    f"({len(entries)} in shard)")
        return n
    except Exception as e:
        logger.error(f"execution log write failed (non-fatal): {e}")
        return 0


def load_entries(since: str = "0000-00-00") -> List[Dict]:
    out: List[Dict] = []
    try:
        if not EXECUTION_LOG_DIR.exists():
            return out
        for path in sorted(EXECUTION_LOG_DIR.glob("*.json")):
            for e in _load_shard(path).get("entries", {}).values():
                if e.get("date", "") >= since:
                    out.append(e)
    except Exception as e:
        logger.warning(f"execution log read failed: {e}")
    return out


def update_entry(date_str: str, key: str, fields: Dict[str, Any]) -> bool:
    try:
        path = _shard_path(date_str)
        shard = _load_shard(path)
        e = shard.get("entries", {}).get(key)
        if e is None:
            return False
        e.update(fields)
        _save_atomic(path, shard)
        return True
    except Exception as e:
        logger.warning(f"execution log update failed: {e}")
        return False
