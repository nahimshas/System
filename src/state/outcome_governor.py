"""Budget gating on OUTCOMES, as a second arm beside the CLV governor.

⚠️ WHY THIS EXISTS: every automatic gate in this system reads CLV, and CLV
missed the worst sport we price. NFL carries POSITIVE closing-line value
(+0.14pp on budget picks, the only positive budget sport) while its picks won
43.4% against a market implying 50.7% — a 7.3pp deficit over 242 settled picks.
So NFL sat squarely in the CLV governor's keep zone while losing money. That is
the second time our fastest signal has failed to separate winners from losers,
the first being MLB's inverted HIGH-confidence label (Aug 4 2026).

CLV asks "did the market move our way after we bet?". This asks the blunter and
slower question: "did we actually beat the price we paid?". They disagree, and
when they do, the protective reading wins — so this arm can only ever turn a
gate ON, never off one the CLV governor set.

THE METRIC is actual win rate minus the mean market-implied probability of the
same picks. Not raw win rate: a 45% win rate is excellent on underdogs priced at
35% and terrible on favourites priced at 60%, so only the difference is
comparable across markets. Positive = we beat the prices we paid.

⚠️ OUTCOMES ARE FAR NOISIER THAN CLV, which is the whole reason CLV was adopted:
~50 graded picks give a CLV verdict that win/loss needs 500+ for. A gate on
outcomes at CLV's sample sizes would be gating on coin flips. So the thresholds
here are DERIVED from the standard error rather than hand-set per phase:

    se = sqrt(0.25 / n)          # p=0.5 maximises variance → conservative
    z  = deficit / se

and a market is gated only when BOTH hold:

    z <= Z_GATE          the deficit is unlikely to be chance
    deficit <= MIN_PP    the deficit is big enough to be worth acting on

Both conditions are load-bearing. z alone would gate a -0.3pp deficit at
n=50,000 — statistically real, economically meaningless. Magnitude alone would
gate a -6pp reading at n=20, which is one bad week. Requiring both is what keeps
this from firing on noise or on trivia.

⚠️ GATES BUDGET ROUTING ONLY. Display, watchlist tiles, the shadow log and the
decision log are untouched, so a gated market keeps producing and keeps being
measured — which is what lets it RECOVER automatically once it earns its way
back. A gate here is never permanent.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["compute_outcome_stats", "outcome_gate", "persist_state",
           "EXEMPTIONS", "active_exemptions"]

STATE_PATH = Path("state/outcome_state.json")

# Minimum graded picks before this arm says anything at all. At n=200 one
# standard error is 3.5pp, so a 2-sigma call needs a ~7pp deficit — about the
# size of the NFL problem. Below this, observe only.
MIN_N = 200

# Sigmas below zero required to gate.
Z_GATE = -2.0

# Minimum deficit in percentage points, independent of significance.
# ⚠️ Not zero on purpose. Market-implied probabilities carry a little residual
# overround even after de-vigging (Kalshi two-sided midpoints, Odds API no-vig
# consensus), which biases every deficit slightly negative. A floor at 0 would
# eventually gate every market in the system on that artefact alone.
MIN_DEFICIT_PP = -2.0

SPORT_KEY = "*"          # the all-markets row for a sport

_stats_cache: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None


# --------------------------------------------------------------------------- #
#  Exemptions — explicit, dated, self-expiring
# --------------------------------------------------------------------------- #
# ⚠️ An exemption records a decision the USER made with the evidence in front of
# them. It is not a way to silence an inconvenient gate, and it is not a default.
# Each one carries an expiry so it cannot outlive the reasoning behind it, and
# the health report prints every active one — an invisible exemption would be
# strictly worse than no governor, because the panel would claim protection the
# system is not providing.
EXEMPTIONS: List[Dict[str, str]] = [
    {
        "sport": "NFL",
        "market": SPORT_KEY,          # whole sport
        "until": "2026-12-07",
        "reason": (
            "User decided Oct 29 2026, shown the -7.3pp figure and the three "
            "options, to leave NFL on budget until nfl_budget_probation "
            "evaluates. The deficit is improving as the warm-start prior decays "
            "(Sep -11.6pp, Oct -3.7pp), which is why waiting is reasonable. "
            "⚠️ Without this exemption NFL would be gated TODAY (z=-2.27). "
            "Expires on the checkpoint date — do not extend it without a new "
            "decision from the user."
        ),
    },
]


def active_exemptions(today: Optional[date] = None) -> List[Dict[str, str]]:
    """Exemptions still in force. Expired ones simply stop applying."""
    today = today or date.today()
    out = []
    for e in EXEMPTIONS:
        try:
            if date.fromisoformat(e["until"]) >= today:
                out.append(e)
        except Exception:
            continue
    return out


def _is_exempt(sport: str, market: str, today: Optional[date] = None) -> Optional[Dict[str, str]]:
    for e in active_exemptions(today):
        if e["sport"].upper() != sport.upper():
            continue
        if e["market"] in (SPORT_KEY, market):
            return e
    return None


# --------------------------------------------------------------------------- #
#  Statistics
# --------------------------------------------------------------------------- #

def _usable(entry: Dict[str, Any]) -> bool:
    """A settled pick with a recorded price, excluding known-broken rows."""
    try:
        if entry.get("outcome") not in ("win", "loss"):
            return False          # push / cancel / pending carry no information
        if entry.get("market_prob_at_first_pick") is None:
            return False
        from src.state.quarantine import is_quarantined
        return not is_quarantined(entry)
    except Exception:
        return entry.get("outcome") in ("win", "loss")


def _summarise(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    try:
        wins = sum(1 for r in rows if r["outcome"] == "win")
        n = len(rows)
        actual = wins / n
        implied = statistics.fmean(float(r["market_prob_at_first_pick"]) for r in rows)
        deficit_pp = (actual - implied) * 100.0
        se_pp = (0.25 / n) ** 0.5 * 100.0
        return {
            "n": n,
            "wins": wins,
            "losses": n - wins,
            "actual_pct": round(actual * 100.0, 2),
            "implied_pct": round(implied * 100.0, 2),
            "deficit_pp": round(deficit_pp, 2),
            "se_pp": round(se_pp, 2),
            "z": round(deficit_pp / se_pp, 2) if se_pp else 0.0,
        }
    except Exception:
        return None


def _phase_and_gate(st: Dict[str, Any]) -> Tuple[int, bool]:
    """Phase 0 observe below MIN_N; above it, gate on z AND magnitude."""
    if st["n"] < MIN_N:
        return 0, False
    gated = st["z"] <= Z_GATE and st["deficit_pp"] <= MIN_DEFICIT_PP
    return 1, bool(gated)


def compute_outcome_stats(force: bool = False) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(SPORT, market_type|'*'): stats} from the shadow log. Cached per process.

    The sport-level '*' row exists because per-market outcome samples are small:
    NFL's three markets hold 91, 85 and 66 settled picks, all far below MIN_N,
    while the sport as a whole holds 242 and is clearly negative. Gating only
    per market would therefore have missed NFL entirely — the exact failure this
    arm was built for. A market is gated if its own row gates OR its sport's
    does; NFL's markets are -4.4pp, -6.0pp and -12.9pp, so the sport-level call
    is not hiding one bad market behind two good ones.

    Never raises — returns {} on failure, which fails open.
    """
    global _stats_cache
    if _stats_cache is not None and not force:
        return _stats_cache
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        from src.state.shadow_log import SHADOW_LOG_DIR, _load_shard
        by_market: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        by_sport: Dict[str, List[Dict[str, Any]]] = {}
        if SHADOW_LOG_DIR.exists():
            for shard_path in sorted(SHADOW_LOG_DIR.glob("*.json")):
                for entry in (_load_shard(shard_path).get("entries") or {}).values():
                    if not _usable(entry):
                        continue
                    sport = (entry.get("sport") or "").upper()
                    if not sport:
                        continue
                    market = entry.get("market_type") or "Unknown"
                    by_market.setdefault((sport, market), []).append(entry)
                    by_sport.setdefault(sport, []).append(entry)
        for key, rows in by_market.items():
            s = _summarise(rows)
            if s:
                stats[key] = s
        for sport, rows in by_sport.items():
            s = _summarise(rows)
            if s:
                stats[(sport, SPORT_KEY)] = s
        for key, st in stats.items():
            phase, gated = _phase_and_gate(st)
            st["phase"] = phase
            st["gated_on_own_record"] = gated
    except Exception as e:
        logger.warning(f"Outcome stats computation failed (non-fatal): {e}")
        stats = {}
    _stats_cache = stats
    return stats


# --------------------------------------------------------------------------- #
#  Gate
# --------------------------------------------------------------------------- #

def _market_type_for_rec(rec: Any) -> str:
    for attr in ("market_type", "bet_type"):
        v = getattr(rec, attr, None)
        if v:
            return str(v)
    return "Unknown"


def outcome_gate(rec: Any, today: Optional[date] = None) -> Tuple[bool, str]:
    """May this pick enter the BUDGET pool? Returns (allowed, reason).

    Default is always allow. Never raises — any failure allows the pick, so
    this arm can never empty the card.
    """
    try:
        sport = (getattr(rec, "sport", "") or "").upper()
        market = _market_type_for_rec(rec)
        if not sport:
            return True, ""
        stats = compute_outcome_stats()
        for key, scope in (((sport, market), market), ((sport, SPORT_KEY), "all markets")):
            st = stats.get(key)
            if not st or not st.get("gated_on_own_record"):
                continue
            ex = _is_exempt(sport, key[1], today)
            if ex:
                logger.info(
                    f"Outcome governor: {sport} {scope} WOULD be gated "
                    f"({st['deficit_pp']:+.1f}pp vs market over {st['n']} picks, "
                    f"z={st['z']}) — exempt until {ex['until']}")
                continue
            return False, (
                f"Outcome governor: {sport} {scope} won {st['actual_pct']}% "
                f"where the market implied {st['implied_pct']}% "
                f"({st['deficit_pp']:+.1f}pp, z={st['z']}) over {st['n']} picks")
        return True, ""
    except Exception as e:
        logger.warning(f"Outcome gate failed open (non-fatal): {e}")
        return True, ""


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #

def _save_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def persist_state() -> None:
    """Write state/outcome_state.json for the panels. Never raises."""
    try:
        from datetime import datetime, timezone
        stats = compute_outcome_stats()
        rows = []
        for (sport, market), st in sorted(stats.items()):
            ex = _is_exempt(sport, market)
            rows.append({
                "sport": sport,
                "market_type": market,
                **{k: st[k] for k in ("n", "wins", "losses", "actual_pct",
                                      "implied_pct", "deficit_pp", "se_pp", "z",
                                      "phase")},
                "gated_on_own_record": st["gated_on_own_record"],
                "gated": bool(st["gated_on_own_record"] and not ex),
                "exempt_until": ex["until"] if ex else None,
            })
        _save_atomic(STATE_PATH, {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "thresholds": {"min_n": MIN_N, "z_gate": Z_GATE,
                           "min_deficit_pp": MIN_DEFICIT_PP},
            "exemptions": active_exemptions(),
            "markets": rows,
        })
    except Exception as e:
        logger.warning(f"Outcome state persist failed (non-fatal): {e}")
