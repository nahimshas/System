"""
Weekly model health report — deterministic measurements for the health routine.

Checks, in order:
  1. Bankroll integrity — bankroll.json vs the history.json ledger (drift = bug).
  2. Budget performance — record / P&L, last 7 days and since the last package.
  3. Budget CLV — shadow-log displayed_in_top entries, avg CLV since package.
  4. Promoted pattern — dog_better_starter picks: record since it went live.
  5. Governor / calibration / cap state — phase changes and active gates.
  6. Checkpoints — evaluates every open entry in checkpoints.json whose
     evaluate_after date has arrived (backtest variants run via backtest.py).

Output: human-readable text (default) or --json for machines.
Read-only: this script NEVER modifies state. Stdlib only.

Usage:
    python -m tools.analysis.health_report
    python -m tools.analysis.health_report --json
"""
import argparse
import glob
import json
import os
from datetime import date, timedelta

from tools.analysis import backtest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PACKAGE_START = "2026-07-05"   # first morning run on Jul 4 package constants


def _load(path):
    with open(os.path.join(_ROOT, path)) as fh:
        return json.load(fh)


def _shadow_entries():
    rows = []
    for p in sorted(glob.glob(os.path.join(_ROOT, "state/shadow_log/*.json"))):
        data = json.load(open(p))
        e = data.get("entries", data) if isinstance(data, dict) else data
        if isinstance(e, dict):
            e = list(e.values())
        rows.extend(e)
    return rows


def check_bankroll():
    """Drift between bankroll.json and the history.json ledger."""
    out = {"name": "bankroll_integrity"}
    try:
        br = _load("state/bankroll.json")["bankroll"]
        hist = _load("state/history.json")
        pnl = sum(float(e.get("actual_pnl", 0) or 0)
                  for e in hist if e.get("result") in ("WON", "LOST", "PUSH"))
        correct = max(10.0, round(100.0 + pnl, 2))
        out.update(bankroll=br, ledger_value=correct,
                   drift=round(br - correct, 2),
                   ok=abs(br - correct) < 0.01)
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


def check_budget_perf(since=PACKAGE_START, last_days=7):
    """Budget record + P&L from history.json (settled singles + parlays)."""
    out = {"name": "budget_performance", "since": since}
    try:
        hist = _load("state/history.json")
        cutoff7 = (date.today() - timedelta(days=last_days)).isoformat()
        for label, lo in (("since_package", since), (f"last_{last_days}d", cutoff7)):
            seg = [e for e in hist
                   if e.get("date", "") >= lo and e.get("result") in ("WON", "LOST")]
            w = sum(1 for e in seg if e["result"] == "WON")
            pnl = round(sum(float(e.get("actual_pnl", 0) or 0) for e in seg), 2)
            out[label] = {"record": f"{w}-{len(seg) - w}", "pnl": pnl, "n": len(seg)}
        out["ok"] = True
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


def check_budget_clv(since=PACKAGE_START):
    """Avg CLV of budget-displayed picks since the package went live."""
    out = {"name": "budget_clv", "since": since}
    try:
        rows = [r for r in _shadow_entries()
                if r.get("date", "") >= since and r.get("displayed_in_top")
                and r.get("clv") is not None]
        out["n"] = len(rows)
        out["avg_clv_pct"] = round(sum(r["clv"] for r in rows) / len(rows) * 100, 2) if rows else None
        out["ok"] = True
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


def check_promoted_pattern(since=PACKAGE_START):
    """dog_better_starter picks (live flag) — settled record + CLV."""
    out = {"name": "promoted_pattern", "since": since}
    try:
        rows = [r for r in _shadow_entries()
                if r.get("date", "") >= since and r.get("dog_better_starter")]
        settled = [r for r in rows if r.get("outcome") in ("win", "loss")]
        w = sum(1 for r in settled if r["outcome"] == "win")
        clvs = [r["clv"] for r in rows if r.get("clv") is not None]
        out.update(
            n_flagged=len(rows), n_settled=len(settled),
            record=f"{w}-{len(settled) - w}",
            wr_pct=round(w / len(settled) * 100, 1) if settled else None,
            avg_clv_pct=round(sum(clvs) / len(clvs) * 100, 2) if clvs else None,
            ok=True,
        )
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


def check_governors():
    """Active CLV gates, calibration phases, current cap values."""
    out = {"name": "governors"}
    try:
        clv = _load("state/clv_state.json")
        gates = []
        for v in clv.values():
            if isinstance(v, list):
                gates = [f"{m['sport']} {m['market_type']} (avg {m['avg_clv']:+.4f}, n={m['n']})"
                         for m in v if m.get("gated")]
        cal = _load("state/calibration_state.json")
        phases = {f"{m['sport']} {m['market_type']}": m["phase"]
                  for m in cal.get("markets", []) if m.get("phase") != "0"}
        caps = _load("state/cap_state.json")["caps"]
        mlb_caps = {k: v["current_value"] for k, v in caps.items() if k.startswith("mlb")}
        out.update(clv_gates=gates, calibration_phases=phases, mlb_caps=mlb_caps, ok=True)
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


def evaluate_checkpoints(today=None):
    """Evaluate every open checkpoint whose evaluate_after date has arrived."""
    today = today or date.today().isoformat()
    cfg = _load("tools/analysis/checkpoints.json")
    results = []
    games_cache = backtest.load_games()
    for cp in cfg["checkpoints"]:
        res = {"id": cp["id"], "title": cp["title"], "status": cp.get("status", "open")}
        if cp.get("status") != "open":
            res["verdict"] = "resolved"
            results.append(res)
            continue
        if today < cp["evaluate_after"]:
            res["verdict"] = f"not_due (evaluate after {cp['evaluate_after']})"
            results.append(res)
            continue
        win = cp.get("data_window_start", "0000-00-00")
        rules = cp.get("rules", {})
        try:
            if cp["type"] == "backtest_variant":
                base = backtest.simulate({}, since=win, games=games_cache)
                var = backtest.simulate(cp["overrides"], since=win, games=games_cache)
                if not base or base["n"] < rules.get("min_picks", 40):
                    res["verdict"] = f"insufficient_data (n={base['n'] if base else 0})"
                else:
                    delta = var["roi"] - base["roi"]
                    res.update(live=base, variant=var, roi_delta_pp=round(delta, 2))
                    res["verdict"] = ("PASS — " + cp["on_pass"]
                                      if delta > rules.get("pass_if_roi_delta_gt", 0.0)
                                      else "FAIL — " + cp["on_fail"])
            elif cp["type"] == "backtest_pattern_only":
                m = backtest.simulate({}, since=win, pattern_only=True, games=games_cache)
                if not m or m["n"] < rules.get("min_picks", 60):
                    res["verdict"] = f"insufficient_data (n={m['n'] if m else 0})"
                else:
                    res.update(metrics=m)
                    res["verdict"] = ("PASS — " + cp["on_pass"]
                                      if m["roi"] > rules.get("pass_if_roi_gt", 0.0)
                                      else "FAIL — " + cp["on_fail"])
            elif cp["type"] == "health":
                clv = check_budget_clv(since=win)
                pat = check_promoted_pattern(since=win)
                res.update(budget_clv=clv, promoted=pat)
                if (pat.get("n_settled") or 0) < rules.get("min_promoted_n", 25):
                    res["verdict"] = f"insufficient_data (promoted n={pat.get('n_settled')})"
                else:
                    wr = pat["wr_pct"] or 0
                    clv_ok = (clv["avg_clv_pct"] or 0) >= rules["budget_clv_pct_min"]
                    if wr >= rules["promoted_wr_keep_pct"] and clv_ok:
                        res["verdict"] = "PASS — " + cp["on_pass"]
                    elif wr <= rules["promoted_wr_demote_pct"] or not clv_ok:
                        res["verdict"] = "FAIL — " + cp["on_fail"]
                    else:
                        res["verdict"] = "borderline — keep watching (wr between demote and keep thresholds)"
            else:   # note
                res["verdict"] = "manual note — see rationale"
                res["rationale"] = cp.get("rationale", "")
        except Exception as e:
            res["verdict"] = f"error: {e}"
        results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser(description="Model health report")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report = {
        "generated_for": date.today().isoformat(),
        "bankroll": check_bankroll(),
        "budget_performance": check_budget_perf(),
        "budget_clv": check_budget_clv(),
        "promoted_pattern": check_promoted_pattern(),
        "governors": check_governors(),
        "checkpoints": evaluate_checkpoints(),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"=== MODEL HEALTH REPORT — {report['generated_for']} ===\n")
    b = report["bankroll"]
    print(f"[{'OK' if b.get('ok') else '⚠ DRIFT'}] Bankroll: ${b.get('bankroll')} "
          f"vs ledger ${b.get('ledger_value')} (drift {b.get('drift')})")
    p = report["budget_performance"]
    for k in ("since_package", "last_7d"):
        if k in p:
            print(f"[--] Budget {k}: {p[k]['record']}  P&L ${p[k]['pnl']:+.2f}")
    c = report["budget_clv"]
    print(f"[--] Budget CLV since package: {c.get('avg_clv_pct')}% (n={c.get('n')})")
    pp = report["promoted_pattern"]
    print(f"[--] Promoted pattern: {pp.get('record')} ({pp.get('wr_pct')}%) "
          f"CLV {pp.get('avg_clv_pct')}% (flagged {pp.get('n_flagged')})")
    g = report["governors"]
    print(f"[--] CLV gates: {g.get('clv_gates') or 'none'}")
    print(f"[--] Calibration phases beyond 0: {g.get('calibration_phases') or 'none'}")
    print(f"[--] MLB caps: {g.get('mlb_caps')}")
    print("\n=== CHECKPOINTS ===")
    for r in report["checkpoints"]:
        print(f"  {r['id']:22} {r['verdict']}")
        if "roi_delta_pp" in r:
            print(f"  {'':22}   live roi {r['live']['roi']:+.1f}% vs variant {r['variant']['roi']:+.1f}% "
                  f"(Δ{r['roi_delta_pp']:+.1f}pp, n={r['live']['n']})")
        if "metrics" in r:
            m = r["metrics"]
            print(f"  {'':22}   n={m['n']} wr={m['wr']}% roi={m['roi']:+.1f}% over {m['days']} days")


if __name__ == "__main__":
    main()
