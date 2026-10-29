"""Does the model's OUTPUT look like a model, or like a broken input?

Every existing automated check in this project asks "is data flowing?" —
settlement rate, CLV coverage, Kalshi map freshness, log liveness. None of them
asks "is what the model just produced plausible?", which is the one question
that would have caught the three most expensive bugs of October 2026:

  • MLS attacking xG read 0.0 for five months. Tell: every MLS total was
    "Under", pinned at the credibility cap, and MLS emitted ZERO moneylines.
  • NBA spread signs inverted on every game where the away team was favoured.
    Tell: all 8 spread picks were "+X.5" underdogs near 50c while the same
    games' moneylines had those teams at 84.5%.
  • Props projecting off one-game averages. Tell: median edge +14.0% with
    48 of 54 projections above the market.

All three are the same shape — a market that has stopped disagreeing with the
book in a *varied* way and started disagreeing in ONE direction, at the cap.
A real edge is noisy and two-sided. A bug is confident and one-sided.

⚠️ DESIGN RULE: thresholds are DERIVED from each market's own trailing history
wherever possible, not tuned. A hand-set band would need re-tuning per sport and
per season, would rot exactly like the hardcoded sport lists, and would make the
check another thing to maintain. The few ABSOLUTE floors below are not tuning
parameters — they are levels at which the output cannot be a working model
regardless of sport.

⚠️ THIS DETECTS, IT DOES NOT GATE. Routing money is the CLV governor's job and
the user's decision. A noisy detector wired straight to the budget could kill a
healthy market silently, which is the failure mode this whole module exists to
prevent. Findings are returned, logged and reported — never acted on.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["evaluate", "findings_for_log"]

# --- Absolute floors. Not tuning: a model cannot be working at these levels. ---

# A market where we take the same side of every single bet is not expressing a
# view, it is expressing an input error. MLS totals ran 100% Under for months.
ONE_SIDED_MIN_N = 10
ONE_SIDED_PCT = 95.0

# The credibility cap exists to clip occasional overconfidence. If it fires on
# nearly everything it is not clipping noise, it is hiding a systematic gap —
# and the post-cap number looks tidy precisely because the cap flattened it.
CAP_PINNED_MIN_N = 10
CAP_PINNED_PCT = 85.0

# |model − market| before the cap. 25pp is far outside anything observed on a
# working market (MLB ~6pp, NBA/NHL/NFL 10-15pp) and inside what the MLS bug
# produced (39.8pp).
RAW_GAP_MIN_N = 10
RAW_GAP_PP = 25.0

# --- Derived drift. Compares a market against ITSELF, so there is no band. ---

RECENT_DAYS = 14
BASELINE_DAYS = 90
DRIFT_MIN_N = 15            # per side of the comparison
RAW_GAP_DRIFT_PP = 8.0      # recent median gap this far above its own baseline
CAP_DRIFT_PCT = 30.0        # cap-firing rate this many points above baseline


try:
    from src.state.quarantine import is_quarantined as _is_quarantined
except Exception:            # pragma: no cover
    def _is_quarantined(_e) -> bool:  # type: ignore[misc]
        return False


def _rows(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Usable rows, with KNOWN-BROKEN windows excluded.

    ⚠️ Quarantined rows must not be re-flagged. They are already recorded as
    broken, with a cause and a fix, and re-reporting them every day for the two
    weeks they stay inside the recent window would make this check noisy in
    exactly the way the project warns about: an alert that fires on a known
    condition is one the reader learns to skip, which is how the earlier silent
    failures survived. The quarantine IS the acknowledgement.
    """
    out = []
    for e in entries or []:
        try:
            if e.get("sport") and e.get("date") and not _is_quarantined(e):
                out.append(e)
        except Exception:
            continue
    return out


def _gap(e: Dict[str, Any]) -> Optional[float]:
    """|model_prob_raw − market_prob|, in percentage points, pre-cap.

    RAW on purpose. The post-cap number is what the cap decided to show us, so
    reading it is how a 47pp error becomes a tidy-looking +10% edge.
    """
    try:
        m, k = e.get("model_prob_raw"), e.get("market_prob_at_first_pick")
        if m is None or k is None:
            return None
        return abs(float(m) - float(k)) * 100.0
    except Exception:
        return None


def _median(vals: List[float]) -> Optional[float]:
    return statistics.median(vals) if vals else None


def _by_market(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    out: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for e in rows:
        out[((e.get("sport") or "").upper(),
             e.get("market_type") or "Unknown")].append(e)
    return out


def _absolute_checks(key, rows) -> List[Dict[str, Any]]:
    sport, market = key
    found: List[Dict[str, Any]] = []

    sides = [e.get("pick_side") for e in rows if e.get("pick_side")]
    if len(sides) >= ONE_SIDED_MIN_N:
        top, n = max(((s, sides.count(s)) for s in set(sides)), key=lambda kv: kv[1])
        pct = 100.0 * n / len(sides)
        if pct >= ONE_SIDED_PCT:
            found.append({
                "sport": sport, "market": market, "code": "one_sided",
                "severity": "high", "n": len(sides),
                "detail": f"{pct:.0f}% of picks are '{top}' ({n}/{len(sides)})",
                "why": "a market taking one side of everything is an input error, "
                       "not a view — this is the MLS xGF signature",
            })

    capped = [e for e in rows if e.get("credibility_cap_fired") is not None]
    if len(capped) >= CAP_PINNED_MIN_N:
        fired = sum(1 for e in capped if e.get("credibility_cap_fired"))
        pct = 100.0 * fired / len(capped)
        if pct >= CAP_PINNED_PCT:
            found.append({
                "sport": sport, "market": market, "code": "cap_pinned",
                "severity": "high", "n": len(capped),
                "detail": f"credibility cap fired on {pct:.0f}% of picks ({fired}/{len(capped)})",
                "why": "the cap is hiding a systematic gap rather than clipping "
                       "noise; read model_prob_raw, not the post-cap edge",
            })

    gaps = [g for g in (_gap(e) for e in rows) if g is not None]
    if len(gaps) >= RAW_GAP_MIN_N:
        med = statistics.median(gaps)
        if med >= RAW_GAP_PP:
            found.append({
                "sport": sport, "market": market, "code": "raw_gap_extreme",
                "severity": "high", "n": len(gaps),
                "detail": f"median raw |model-market| = {med:.1f}pp",
                "why": "far outside any working market observed here "
                       f"(>= {RAW_GAP_PP:.0f}pp); suspect a feed or field name",
            })
    return found


def _drift_checks(key, recent, baseline) -> List[Dict[str, Any]]:
    """Compare a market against its OWN past. No tuned band to rot."""
    sport, market = key
    found: List[Dict[str, Any]] = []

    r_gaps = [g for g in (_gap(e) for e in recent) if g is not None]
    b_gaps = [g for g in (_gap(e) for e in baseline) if g is not None]
    if len(r_gaps) >= DRIFT_MIN_N and len(b_gaps) >= DRIFT_MIN_N:
        r, b = statistics.median(r_gaps), statistics.median(b_gaps)
        if r - b >= RAW_GAP_DRIFT_PP:
            found.append({
                "sport": sport, "market": market, "code": "raw_gap_drift",
                "severity": "medium", "n": len(r_gaps),
                "detail": f"median raw gap {b:.1f}pp -> {r:.1f}pp (+{r - b:.1f}pp)",
                "why": "the model started disagreeing with the book much harder "
                       "than it used to; an input changed, not the sport",
            })

    def cap_pct(rs):
        c = [e for e in rs if e.get("credibility_cap_fired") is not None]
        if len(c) < DRIFT_MIN_N:
            return None
        return 100.0 * sum(1 for e in c if e.get("credibility_cap_fired")) / len(c)

    rc, bc = cap_pct(recent), cap_pct(baseline)
    if rc is not None and bc is not None and rc - bc >= CAP_DRIFT_PCT:
        found.append({
            "sport": sport, "market": market, "code": "cap_drift",
            "severity": "medium", "n": len(recent),
            "detail": f"cap firing {bc:.0f}% -> {rc:.0f}% (+{rc - bc:.0f}pp)",
            "why": "a jump in cap firing is the cheapest early warning that an "
                   "input broke",
        })
    return found


def _missing_market_checks(rows, recent_rows) -> List[Dict[str, Any]]:
    """A sport that used to price a market and has stopped.

    MLS emitted ZERO moneylines for months while still emitting draws and
    totals. Nothing noticed, because a market with no picks looks exactly like
    a market with no opportunities.
    """
    found: List[Dict[str, Any]] = []
    hist: Dict[str, set] = defaultdict(set)
    for e in rows:
        hist[(e.get("sport") or "").upper()].add(e.get("market_type") or "Unknown")
    now: Dict[str, set] = defaultdict(set)
    for e in recent_rows:
        now[(e.get("sport") or "").upper()].add(e.get("market_type") or "Unknown")
    for sport, markets in sorted(hist.items()):
        if sport not in now or not now[sport]:
            continue              # sport is simply quiet / out of season
        gone = sorted(markets - now[sport])
        for market in gone:
            found.append({
                "sport": sport, "market": market, "code": "market_went_quiet",
                "severity": "medium", "n": 0,
                "detail": f"no picks in {RECENT_DAYS}d, but other {sport} "
                          f"markets are active",
                "why": "a market that stops producing while its siblings keep "
                       "going is the MLS zero-moneylines signature",
            })
    return found


def evaluate(entries: Iterable[Dict[str, Any]],
             today: Optional[date] = None) -> Dict[str, Any]:
    """Return {"ok": bool, "findings": [...]} over shadow-log entries.

    Never raises. A detector that can break the morning run is worse than no
    detector, so every failure path returns ok=True with an error noted.
    """
    out: Dict[str, Any] = {"ok": True, "findings": [], "checked_markets": 0}
    try:
        today = today or date.today()
        rows = _rows(entries)
        if not rows:
            return out
        r_cut = (today - timedelta(days=RECENT_DAYS)).isoformat()
        b_cut = (today - timedelta(days=BASELINE_DAYS)).isoformat()
        recent = [e for e in rows if e["date"] >= r_cut]
        baseline = [e for e in rows if b_cut <= e["date"] < r_cut]

        findings: List[Dict[str, Any]] = []
        by_recent = _by_market(recent)
        by_base = _by_market(baseline)
        for key, rs in sorted(by_recent.items()):
            findings += _absolute_checks(key, rs)
            findings += _drift_checks(key, rs, by_base.get(key, []))
        findings += _missing_market_checks(rows, recent)

        order = {"high": 0, "medium": 1}
        findings.sort(key=lambda f: (order.get(f["severity"], 9), f["sport"], f["market"]))
        out["findings"] = findings
        out["checked_markets"] = len(by_recent)
        out["ok"] = not any(f["severity"] == "high" for f in findings)
    except Exception as e:                       # pragma: no cover
        out["error"] = str(e)
        logger.warning(f"Model sanity check failed (non-fatal): {e}")
    return out


def findings_for_log(result: Dict[str, Any]) -> List[str]:
    """One line per finding, for the daily run log."""
    lines = []
    for f in (result or {}).get("findings") or []:
        lines.append(f"MODEL SANITY [{f['severity']}] {f['sport']} {f['market']}: "
                     f"{f['detail']} — {f['why']}")
    return lines
