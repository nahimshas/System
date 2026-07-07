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


def check_log_liveness(recent_days=3, clv_window_days=14):
    """The improvement engine dies silently if logging breaks — check pulse.

    - decision log: rows written in the last `recent_days` (feature stamping alive)
    - shadow log:   rows written in the last `recent_days`
    - settlement:   share of shadow entries >=2 days old that have an outcome
    - CLV coverage: share of settled entries in the last `clv_window_days` with
      a stamped clv (collapsing coverage = stamping off/broken/out of credits)
    """
    out = {"name": "log_liveness"}
    try:
        cutoff_recent = (date.today() - timedelta(days=recent_days)).isoformat()
        cutoff_settle = (date.today() - timedelta(days=2)).isoformat()
        cutoff_clv = (date.today() - timedelta(days=clv_window_days)).isoformat()

        dl_recent = 0
        for p in sorted(glob.glob(os.path.join(_ROOT, "state/decision_log/*.json"))):
            data = json.load(open(p))
            e = data.get("entries", data) if isinstance(data, dict) else data
            if isinstance(e, dict):
                e = list(e.values())
            dl_recent += sum(1 for r in e if r.get("date", "") >= cutoff_recent)

        shadow = _shadow_entries()
        sl_recent = sum(1 for r in shadow if r.get("date", "") >= cutoff_recent)
        old_enough = [r for r in shadow if cutoff_clv <= r.get("date", "") <= cutoff_settle]
        settled = [r for r in old_enough if r.get("outcome") in ("win", "loss", "push")]
        with_clv = [r for r in settled if r.get("clv") is not None]

        out.update(
            decision_rows_recent=dl_recent,
            shadow_rows_recent=sl_recent,
            settlement_rate_pct=round(len(settled) / len(old_enough) * 100, 1) if old_enough else None,
            clv_coverage_pct=round(len(with_clv) / len(settled) * 100, 1) if settled else None,
            ok=dl_recent > 0 and sl_recent > 0,
        )
    except Exception as e:
        out.update(ok=False, error=str(e))
    return out


# Deterministic ACTION NEEDED triggers beyond checkpoints — the routine treats
# a non-empty alerts list as ACTION NEEDED (or DEGRADED for liveness failures).
DRAWDOWN_ALERT_7D = -40.0     # last-7-days budget P&L below this → alert
SETTLEMENT_RATE_MIN = 60.0    # % of 2+ day-old shadow entries settled
CLV_COVERAGE_MIN = 40.0       # % of settled entries carrying CLV (14d window)


def compute_alerts(report):
    alerts = []
    b = report["bankroll"]
    if not b.get("ok"):
        alerts.append(f"bankroll drift: ${b.get('drift')} vs ledger — investigate before next sizing run")
    p7 = report["budget_performance"].get("last_7d", {})
    if p7 and p7.get("pnl") is not None and p7["pnl"] < DRAWDOWN_ALERT_7D:
        alerts.append(f"drawdown: last-7d budget P&L ${p7['pnl']:+.2f} below ${DRAWDOWN_ALERT_7D} floor ({p7['record']})")
    lv = report["log_liveness"]
    if not lv.get("ok"):
        alerts.append("logging stopped: no decision/shadow rows in the last 3 days — feature stamping or the morning run is broken")
    if lv.get("settlement_rate_pct") is not None and lv["settlement_rate_pct"] < SETTLEMENT_RATE_MIN:
        alerts.append(f"settlement lag: only {lv['settlement_rate_pct']}% of mature shadow entries settled")
    if lv.get("clv_coverage_pct") is not None and lv["clv_coverage_pct"] < CLV_COVERAGE_MIN:
        alerts.append(f"CLV coverage {lv['clv_coverage_pct']}% (14d) — stamping disabled/broken/out of credits (informational if deliberately disabled)")
    return alerts


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


def classify(report):
    """Deterministic weekly status. ACTION_NEEDED when a human decision is due."""
    if any(not report[k].get("ok", True) and report[k].get("error")
           for k in ("bankroll", "budget_performance", "budget_clv",
                     "promoted_pattern", "log_liveness", "governors")):
        return "DEGRADED"
    if not report["log_liveness"].get("ok"):
        return "DEGRADED"
    if report["alerts"]:
        return "ACTION_NEEDED"
    for c in report["checkpoints"]:
        v = c.get("verdict", "")
        if v.startswith("PASS") or v.startswith("FAIL"):
            return "ACTION_NEEDED"
    return "ALL_NORMAL"


_STATUS_STYLE = {
    "ALL_NORMAL":    ("#22c55e", "✅ All normal — no decisions needed this week"),
    "ACTION_NEEDED": ("#f59e0b", "⚠ Action needed — a decision is due (see below)"),
    "DEGRADED":      ("#ef4444", "❌ Degraded — health checks could not run cleanly"),
}


def _esc(s):
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_health_html(report, status):
    col, banner = _STATUS_STYLE[status]
    p = report["budget_performance"]
    c = report["budget_clv"]
    pp = report["promoted_pattern"]
    b = report["bankroll"]
    lv = report["log_liveness"]
    g = report["governors"]

    def chip(k, v):
        return (f'<div style="background:#1a1d27;border-radius:8px;padding:8px 10px;text-align:center;">'
                f'<div style="color:#64748b;font-size:0.6rem;text-transform:uppercase;">{k}</div>'
                f'<div style="color:#e2e8f0;font-weight:800;font-size:0.88rem;">{v}</div></div>')

    sp = p.get("since_package", {})
    l7 = p.get("last_7d", {})
    chips = (chip("Bankroll", f"${b.get('bankroll')}")
             + chip("Last 7d", f"{l7.get('record','?')} (${l7.get('pnl',0):+.0f})")
             + chip("Since pkg", f"{sp.get('record','?')} (${sp.get('pnl',0):+.0f})")
             + chip("Budget CLV", f"{c.get('avg_clv_pct')}%" if c.get("avg_clv_pct") is not None else "n/a")
             + chip("Pattern", f"{pp.get('record','?')}"))

    alerts_html = ""
    if report["alerts"]:
        items = "".join(f"<li>{_esc(a)}</li>" for a in report["alerts"])
        alerts_html = (f'<div style="background:#1a1d27;border-left:3px solid #ef4444;'
                       f'border-radius:10px;padding:12px 14px;margin:12px 0;">'
                       f'<div style="color:#ef4444;font-weight:700;font-size:0.8rem;">🚨 ALERTS</div>'
                       f'<ul style="color:#e2e8f0;font-size:0.78rem;margin:6px 0 0;padding-left:18px;">{items}</ul></div>')

    cp_rows = ""
    for r in report["checkpoints"]:
        v = r.get("verdict", "")
        vcol = "#22c55e" if v.startswith("PASS") else ("#ef4444" if v.startswith("FAIL") else "#94a3b8")
        extra = ""
        if "roi_delta_pp" in r:
            extra = (f'<div style="color:#64748b;font-size:0.7rem;margin-top:2px;">live '
                     f'{r["live"]["roi"]:+.1f}% vs variant {r["variant"]["roi"]:+.1f}% '
                     f'(Δ{r["roi_delta_pp"]:+.1f}pp, n={r["live"]["n"]})</div>')
        if "metrics" in r:
            m = r["metrics"]
            extra = (f'<div style="color:#64748b;font-size:0.7rem;margin-top:2px;">n={m["n"]} '
                     f'wr={m["wr"]}% roi={m["roi"]:+.1f}%</div>')
        cp_rows += (f'<div style="background:#1a1d27;border-radius:8px;padding:9px 12px;margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;gap:10px;">'
                    f'<span style="color:#38bdf8;font-size:0.76rem;font-weight:600;">{_esc(r["id"])}</span>'
                    f'<span style="color:{vcol};font-size:0.72rem;text-align:right;">{_esc(v)}</span></div>'
                    f'{extra}</div>')

    gov = (f"CLV gates: {', '.join(g.get('clv_gates') or []) or 'none'} · "
           f"Phases: {', '.join(f'{k}={v}' for k, v in (g.get('calibration_phases') or {}).items()) or 'none'} · "
           f"Logs: decision(3d)={lv.get('decision_rows_recent')} settled={lv.get('settlement_rate_pct')}% "
           f"clv_cov={lv.get('clv_coverage_pct')}%")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model Health — {report['generated_for']}</title></head>
<body style="background:#050507;color:#e2e8f0;font-family:-apple-system,system-ui,sans-serif;margin:0;padding:16px;">
<div style="max-width:600px;margin:0 auto;">
<div style="background:#1a1d27;border-left:4px solid {col};border-radius:12px;padding:16px;margin-bottom:14px;">
<div style="font-size:1.02rem;font-weight:800;">🩺 Model Health — {report['generated_for']}</div>
<div style="color:{col};font-weight:700;font-size:0.84rem;margin-top:6px;">{banner}</div>
<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:12px;">{chips}</div>
</div>
{alerts_html}
<div style="color:#64748b;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:18px 0 8px;">Checkpoints</div>
{cp_rows}
<div style="color:#64748b;font-size:0.7rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin:18px 0 8px;">Governors & Logs</div>
<div style="background:#1a1d27;border-radius:8px;padding:10px 12px;color:#94a3b8;font-size:0.74rem;line-height:1.6;">{_esc(gov)}</div>
<div style="color:#64748b;font-size:0.66rem;text-align:center;margin:22px 0 8px;">
Generated automatically (deterministic) · <a href="/System/index_spa.html" style="color:#38bdf8;">← Back to Picks</a></div>
</div></body></html>"""


def publish(report):
    """Write docs/health_latest.html + append docs/health_history.json.
    Idempotent per date. Prints the status so the workflow can gate the
    notification step on it."""
    status = classify(report)
    docs = os.path.join(_ROOT, "docs")
    with open(os.path.join(docs, "health_latest.html"), "w") as f:
        f.write(_render_health_html(report, status))

    hist_path = os.path.join(docs, "health_history.json")
    try:
        with open(hist_path) as f:
            hist = json.load(f)
    except Exception:
        hist = {"entries": []}
    entries = hist.get("entries", hist) if isinstance(hist, dict) else hist
    d = report["generated_for"]
    entries = [e for e in entries if e.get("date") != d]
    entries.append({
        "date": d,
        "status": status,
        "headline": (report["alerts"][0] if report["alerts"] else
                     "Bankroll ok, logs alive, no checkpoint decisions due."),
        "checkpoint_verdicts": {r["id"]: r.get("verdict", "") for r in report["checkpoints"]},
        "budget_clv_pct": report["budget_clv"].get("avg_clv_pct"),
        "promoted_record": report["promoted_pattern"].get("record"),
        "bankroll_drift": report["bankroll"].get("drift"),
        "generated_by": "workflow",
    })
    hist = {"entries": entries[-26:]}
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"HEALTH_STATUS={status}")
    return status


def main():
    ap = argparse.ArgumentParser(description="Model health report")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--publish", action="store_true",
                    help="write docs/health_latest.html + docs/health_history.json")
    args = ap.parse_args()

    report = {
        "generated_for": date.today().isoformat(),
        "bankroll": check_bankroll(),
        "budget_performance": check_budget_perf(),
        "budget_clv": check_budget_clv(),
        "promoted_pattern": check_promoted_pattern(),
        "log_liveness": check_log_liveness(),
        "governors": check_governors(),
        "checkpoints": evaluate_checkpoints(),
    }
    report["alerts"] = compute_alerts(report)
    if args.publish:
        publish(report)
        return
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"=== MODEL HEALTH REPORT — {report['generated_for']} ===\n")
    if report["alerts"]:
        print("🚨 ALERTS:")
        for a in report["alerts"]:
            print(f"  - {a}")
        print()
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
    lv = report["log_liveness"]
    print(f"[{'OK' if lv.get('ok') else '⚠ DEAD'}] Logs: decision rows(3d)={lv.get('decision_rows_recent')} "
          f"shadow(3d)={lv.get('shadow_rows_recent')} settled={lv.get('settlement_rate_pct')}% "
          f"clv_cov={lv.get('clv_coverage_pct')}%")
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
