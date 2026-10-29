"""Shadow-log rows produced by a model we have since proven was broken.

A pick logged under a broken input is still a true record of what we picked and
what happened, so it is never deleted — but it must not TEACH. Calibration
learns a probability→outcome mapping and the CLV governor gates markets on
average CLV; feeding either one probabilities that came from a bug makes it
learn the bug and then defend it.

⚠️ This is why a quarantine and not a delete. `state/shadow_log/` is the
append/update-only record the whole project is built on, the decision log is a
pure archive, and `watchlist_history.json` is what the user's tiles show. All
three keep every row. Only the two LEARNING consumers skip them, and both go
through `is_quarantined()` here so there is exactly one definition.

To add a window: append a `_Window`, say what broke and what fixed it, and
nothing else needs touching — `health_report` derives its summary from this
list, so a quarantine can never be silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

__all__ = ["QUARANTINE", "is_quarantined", "summary"]


class _Window(NamedTuple):
    sport: str
    markets: Optional[Tuple[str, ...]]   # None = every market for that sport
    date_from: str                       # inclusive, YYYY-MM-DD
    date_to: str                         # inclusive
    reason: str
    fixed_by: str


QUARANTINE: List[_Window] = [
    _Window(
        sport="MLS",
        # EVERY market, not just Total and Draw. The bug collapsed expected
        # goals, and every MLS probability is read off the same two Poisson
        # means — a moneyline from P(home > away) with both means near 0.65
        # instead of 1.45 is a different number even though the RANKING it
        # produced was fine. Calibration learns the number, not the ranking.
        markets=None,
        date_from="2026-05-17",   # first MLS row in the shadow log
        date_to="2026-10-28",     # last run on the broken code; the fix landed
                                  # after that morning's run and MLS had no
                                  # fixtures on the 29th, so there are no
                                  # good rows inside this window to protect.
        reason=(
            "Attacking xG read 0.0 for every team: the ASA field is "
            "`xgoals_for` and the code tried xGoals/xgoals/xgf, while "
            "`xgoals_against` happened to be in the defensive fallback list "
            "and worked. One-sided, so it hid — team RANKING stayed right off "
            "xGA and moneylines looked calibrated, but expected total goals "
            "collapsed to ~1.3 against a market implying ~2.9. Every total "
            "became Under at the capped +10% edge and draws were inflated "
            "(two teams projected to score nothing draw 0-0 often). Raw "
            "model-vs-market gap was 39.8pp, 3.9pp after the fix."
        ),
        fixed_by="81cc594f",
    ),
]


def is_quarantined(entry: Dict[str, Any]) -> bool:
    """True if this shadow-log entry must not feed calibration or the governor.

    Never raises: a malformed entry is simply not quarantined, so a bad row can
    only ever leave the previous behaviour in place, never hide a good market.
    """
    try:
        sport = (entry.get("sport") or "").upper()
        if not sport:
            return False
        date = entry.get("date") or ""
        market = entry.get("market_type") or ""
        for w in QUARANTINE:
            if sport != w.sport.upper():
                continue
            if w.markets is not None and market not in w.markets:
                continue
            if w.date_from <= date <= w.date_to:
                return True
    except Exception:
        return False
    return False


def summary(entries: Any) -> List[Dict[str, Any]]:
    """Per-window counts over `entries`, for the health report to print.

    Derived from QUARANTINE so a window added above shows up without a second
    edit — the hardcoded-sport-list failure mode, which here would mean a
    quarantine nobody is told about.
    """
    out: List[Dict[str, Any]] = []
    rows = list(entries or [])
    for w in QUARANTINE:
        n = settled = 0
        for e in rows:
            try:
                if (e.get("sport") or "").upper() != w.sport.upper():
                    continue
                if w.markets is not None and (e.get("market_type") or "") not in w.markets:
                    continue
                if not (w.date_from <= (e.get("date") or "") <= w.date_to):
                    continue
                n += 1
                if e.get("outcome") is not None:
                    settled += 1
            except Exception:
                continue
        out.append({
            "sport": w.sport,
            "markets": list(w.markets) if w.markets else "ALL",
            "window": f"{w.date_from}..{w.date_to}",
            "rows": n,
            "settled_excluded": settled,
            "fixed_by": w.fixed_by,
        })
    return out
