"""
Decision-log backtest engine — reruns the MLB model with arbitrary constants
against logged features and simulates the daily budget selection.

This is the canonical methodology used for the Jul 2026 optimization package
(see docs/DEVELOPMENT_PLAN.md "Jul 4 2026"): reconstruct each game's model
probability from the decision log's `features` dict, apply caps/gates, rank
candidates confidence-first (promoted pattern first, then edge), take the top
5 per day, and score flat-stake ROI at realistic friction (4.5% vig).

Usage:
    python -m tools.analysis.backtest                          # live constants, all data
    python -m tools.analysis.backtest --since 2026-07-05       # window
    python -m tools.analysis.backtest --set HA=0.0             # variant vs live
    python -m tools.analysis.backtest --set RLSIG=3.0 --since 2026-07-05
    python -m tools.analysis.backtest --pattern-only           # promoted picks + edges >= 0.08 only

Importable: `simulate(overrides, since, until, pattern_only)` returns a metrics
dict — used by tools/analysis/health_report.py for checkpoint evaluation.

Stdlib only (statistics.NormalDist, no scipy) so it runs anywhere.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

_ND = NormalDist()
_cdf = _ND.cdf
_ppf = _ND.inv_cdf

# Realistic round-trip friction, validated against actual recorded P&L
# (82 matched picks, Jul 3 2026: actual -15.9%/$ vs sim -15.1% at this vig).
VIG = 0.045

# LIVE constants — keep in sync with src/models/edge_finder.py + config.
# These are the Jul 4 2026 optimization package values.
LIVE = dict(
    PITCH=0.80,   # _PITCH_COEFF
    OFF=0.4,      # offense weight (_OFF_W)
    INJ=0.0,      # _INJ_RUNS_PER_PCT
    HA=0.015,     # MLB_HOME_ADVANTAGE
    STD=2.2,      # MLB_SPREAD_STD
    FORM=1.0,     # recent-form delta scale (1.0 = as logged)
    CRED=0.10,    # MLB credibility cap
    MINE=0.05,    # BUDGET_MIN_EDGE
    RLSIG=3.5,    # MLB_RUNLINE_SIGMA
    RDCAP=1.8,    # MLB_RUN_DIFF_CAP
    KPEN=True,    # K-matchup penalty on
    PROMO=True,   # dog-with-better-starter HIGH promotion
    PROMO_BYPASS=False,   # promoted picks ignore the MINE budget floor (candidate rule,
                          # not shipped — see checkpoints.json promo_bypass_floor)
)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_games(since: str = "0000-00-00", until: str = "9999-99-99") -> dict:
    """Load MLB decision-log rows grouped per (date, game) with features."""
    games = {}
    for path in sorted(glob.glob(os.path.join(_repo_root(), "state/decision_log/*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        entries = data.get("entries", data) if isinstance(data, dict) else data
        if isinstance(entries, dict):
            entries = list(entries.values())
        for e in entries:
            if e.get("sport") != "MLB":
                continue
            if not (since <= e.get("date", "") <= until):
                continue
            k = (e["date"], e["game"])
            g = games.setdefault(k, {"rows": []})
            g["rows"].append(e)
            if "feat" not in g and e.get("features"):
                g["feat"] = e["features"]
                g["home"] = e.get("home_team")
                g["away"] = e.get("away_team")
    return games


def _pnl(row) -> float:
    p = row["market_prob_at_first_pick"]
    return ((1.0 / p) * (1 - VIG) - 1.0) if row["outcome"] == "win" else -1.0


def _cap(model, market, drift):
    return min(market + drift, max(market - drift, model))


def _reconstruct_home_prob(feat: dict, P: dict) -> float:
    """Rebuild the model's pre-cap home win probability from logged features.

    Mirrors analyze_mlb_game's expected-runs formula. Validated to reproduce
    logged model_prob_raw with ~zero median error (Jul 3 2026)."""
    LAR = feat.get("league_avg_runs", 4.35)

    def side(prefix, opp, home_side):
        sp_ip = min(7.0, max(2.0, feat.get(f"{opp}_sp_ip") or 5.0))
        spc = P["PITCH"] * (sp_ip / 9.0)
        bpc = P["PITCH"] * (1 - sp_ip / 9.0)
        sp_sc = feat.get(f"{opp}_sp_score") or 0.0
        bp_sc = feat.get(f"{opp}_bullpen_score")
        if bp_sc is None:
            bp_era = feat.get(f"{opp}_bp_era")
            bp_sc = (4.20 - bp_era) / 1.50 if bp_era is not None else 0.0
        off = feat.get(f"{prefix}_offense_adj") or 0.0
        r = LAR - sp_sc * spc - bp_sc * bpc + off * P["OFF"]
        if home_side:
            r += P["HA"] * 5
        kpen = feat.get(f"{prefix}_k_penalty") or 0.0
        if P["KPEN"] and kpen > 0.01:
            r -= kpen
        fd = feat.get(f"{prefix}_form_delta")
        if fd is not None:
            r += fd * P["FORM"]
        ump = feat.get("ump_run_factor") or 1.0
        if abs(ump - 1.0) >= 0.02:
            r *= ump
        r += (feat.get("weather_run_adj") or 0.0) / 2
        lp = feat.get(f"{prefix}_load_pen") or 0.0
        if lp > 0:
            r *= (1 - lp)
        if feat.get("playoff"):
            r *= 0.92
        inj = feat.get(f"{prefix}_inj") or 0.0
        if inj > 0.01:
            r -= inj * P["INJ"] * 100
        return max(1.5, r)

    rd = side("home", "away", True) - side("away", "home", False)
    rdc = P["RDCAP"] * math.tanh(rd / P["RDCAP"])
    return min(0.85, max(0.15, _cdf(rdc / P["STD"])))


def simulate(overrides: dict = None, since: str = "0000-00-00",
             until: str = "9999-99-99", pattern_only: bool = False,
             games: dict = None) -> dict:
    """Run the budget simulation. Returns metrics dict or None if no picks.

    pattern_only: budget accepts ONLY promoted (dog+better-starter) picks and
    edges >= 0.08 — the "pattern-concentrated card" configuration. Card size
    is variable (thin days produce fewer/zero picks).
    """
    P = dict(LIVE)
    if overrides:
        P.update(overrides)
    if games is None:
        games = load_games(since, until)

    by_day = defaultdict(list)
    for (dte, _), gd in games.items():
        if not (since <= dte <= until):
            continue
        feat = gd.get("feat")
        if not feat or not feat.get("stats_available", True):
            continue
        home, away = gd["home"], gd["away"]
        php = _reconstruct_home_prob(feat, P)

        mh_row = ma_row = sh = sa = None
        for r in gd["rows"]:
            if not r.get("market_prob_at_first_pick"):
                continue
            if r["market_type"] == "Moneyline":
                if r["side"] == home:
                    mh_row = r
                elif r["side"] == away:
                    ma_row = r
            elif r["market_type"] == "Spread":
                if r["side"] == home:
                    sh = r
                elif r["side"] == away:
                    sa = r

        cands = []
        ml_home_prob = php
        if mh_row and ma_row:
            mh = mh_row["market_prob_at_first_pick"]
            ma = ma_row["market_prob_at_first_pick"]
            mp = _cap(php, mh, P["CRED"])
            if (feat.get("home_inj") or 0) >= 0.030:
                mp = min(mp, min(mh * 1.10, 0.85))
            ap = 1 - mp
            if (feat.get("away_inj") or 0) >= 0.030:
                ap2 = min(ap, min(ma * 1.10, 0.85))
                if ap2 < ap:
                    ap = ap2
                    mp = 1 - ap
            if mp - mh >= P["MINE"]:
                cands.append((mp - mh, mh_row, False))
            if ap - ma >= P["MINE"]:
                cands.append((ap - ma, ma_row, False))
            ml_home_prob = mp

        if sh and sa and sh.get("line") is not None:
            line = sh["line"]
            hs = feat.get("home_sp_score") or 0
            as_ = feat.get("away_sp_score") or 0
            margin = _ppf(min(0.999, max(0.001, ml_home_prob))) * P["STD"]
            phc = _cdf((margin + line) / P["RLSIG"])
            mhc = sh["market_prob_at_first_pick"]
            mac = sa["market_prob_at_first_pick"]
            he = _cap(phc, mhc, P["CRED"]) - mhc
            ae = _cap(1 - phc, mac, P["CRED"]) - mac
            h_promo = P["PROMO"] and line > 0 and (hs - as_) > 0.1
            a_promo = P["PROMO"] and line < 0 and (as_ - hs) > 0.1
            if he >= P["MINE"]:
                cands.append((he, sh, h_promo))
            if ae >= P["MINE"]:
                cands.append((ae, sa, a_promo))

        for edge, row, promo in cands:
            if row.get("outcome") in ("win", "loss"):
                if pattern_only and not (promo or edge >= 0.08):
                    continue
                by_day[dte].append((edge, row, promo))

    picks = []
    for dte, cs in by_day.items():
        # confidence-first: promoted first, then edge — mirrors _slot_sort_key
        cs.sort(key=lambda x: (0 if x[2] else 1, -x[0]))
        seen = set()
        day = []
        for edge, row, promo in cs:
            k = (row["game"], row["market_type"])
            if k in seen:
                continue
            seen.add(k)
            day.append((row, promo))
            if len(day) == 5:
                break
        picks.extend(day)

    if not picks:
        return None
    w = sum(1 for r, _ in picks if r["outcome"] == "win")
    n = len(picks)
    clvs = [r["clv"] for r, _ in picks if r.get("clv") is not None]
    return dict(
        n=n,
        days=len(by_day),
        wr=round(w / n * 100, 1),
        roi=round(sum(_pnl(r) for r, _ in picks) / n * 100, 2),
        units=round(sum(_pnl(r) for r, _ in picks), 2),
        avg_clv=round(sum(clvs) / len(clvs) * 100, 2) if clvs else None,
        clv_n=len(clvs),
        n_promoted=sum(1 for _, p in picks if p),
        promoted_wins=sum(1 for r, p in picks if p and r["outcome"] == "win"),
    )


def main():
    ap = argparse.ArgumentParser(description="MLB decision-log backtest")
    ap.add_argument("--since", default="0000-00-00")
    ap.add_argument("--until", default="9999-99-99")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a constant, e.g. --set HA=0.0 --set RLSIG=3.0")
    ap.add_argument("--pattern-only", action="store_true",
                    help="budget = promoted picks + edges >= 0.08 only, variable card size")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        k = k.upper()
        if k not in LIVE:
            raise SystemExit(f"unknown constant {k}; valid: {', '.join(LIVE)}")
        overrides[k] = (v.lower() in ("1", "true", "yes")) if isinstance(LIVE[k], bool) else float(v)

    games = load_games(args.since, args.until)
    base = simulate({}, args.since, args.until, games=games)
    print(f"LIVE constants        : {base}")
    if overrides or args.pattern_only:
        var = simulate(overrides, args.since, args.until,
                       pattern_only=args.pattern_only, games=games)
        label = "PATTERN-ONLY" if args.pattern_only else f"VARIANT {overrides}"
        print(f"{label:22}: {var}")
        if base and var:
            print(f"delta ROI: {var['roi'] - base['roi']:+.2f}pp")


if __name__ == "__main__":
    main()
