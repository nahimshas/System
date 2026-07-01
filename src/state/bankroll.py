"""Persistent rolling bankroll — carries daily P&L forward into Kelly sizing."""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

BANKROLL_PATH = Path("state/bankroll.json")
_BANKROLL_FLOOR = 10.0   # never let the bankroll collapse to zero


def load_bankroll() -> float:
    """Return current bankroll, falling back to DAILY_BUDGET when no state exists."""
    try:
        if BANKROLL_PATH.exists():
            data = json.loads(BANKROLL_PATH.read_text())
            amount = float(data.get("bankroll", 0))
            if amount >= _BANKROLL_FLOOR:
                return amount
    except Exception as e:
        logger.warning(f"Bankroll load failed (using default): {e}")
    from src.config import DAILY_BUDGET
    return float(DAILY_BUDGET)


def save_bankroll(amount: float, note: str = "") -> None:
    """Persist bankroll to state/bankroll.json. Non-fatal on error."""
    try:
        BANKROLL_PATH.parent.mkdir(exist_ok=True)
        payload: dict = {
            "bankroll": round(amount, 2),
            "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if note:
            payload["note"] = note
        BANKROLL_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        logger.warning(f"Bankroll save failed (non-fatal): {e}")


def reconcile_bankroll_from_history() -> float:
    """
    Recompute bankroll from state/history.json (the full settled-picks ledger)
    and correct state/bankroll.json if it has drifted.

    bankroll.json is normally advanced incrementally, one night at a time, by
    the Results Snapshot workflow. That workflow only runs when the nightly
    debrief routine successfully triggers it — if that trigger fails (e.g. a
    proxy outage), bankroll.json silently stops advancing while history.json
    keeps accumulating settled results the next morning via the outcome
    checker (a separate, independent path). The two then disagree by exactly
    the missed night(s) of P&L. Recomputing from history.json — the complete
    ledger, always kept current regardless of the nightly workflow's health —
    is the authoritative self-heal, run once per morning before Kelly sizing.
    """
    from src.config import DAILY_BUDGET
    from src.data.outcome_checker import HISTORY_PATH
    try:
        if not HISTORY_PATH.exists():
            return load_bankroll()
        history = json.loads(HISTORY_PATH.read_text())
        total_pnl = sum(
            float(e.get("actual_pnl", 0) or 0)
            for e in history if e.get("result") in ("WON", "LOST", "PUSH")
        )
        correct = max(_BANKROLL_FLOOR, round(float(DAILY_BUDGET) + total_pnl, 2))
        current = load_bankroll()
        if abs(correct - current) >= 0.01:
            logger.info(f"Bankroll reconciled from history.json: ${current:.2f} -> ${correct:.2f}")
            save_bankroll(correct, note=f"Auto-reconciled from history.json all-time P&L ({total_pnl:+.2f})")
        return correct
    except Exception as e:
        logger.warning(f"Bankroll reconciliation failed (using cached value): {e}")
        return load_bankroll()
