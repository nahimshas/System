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
