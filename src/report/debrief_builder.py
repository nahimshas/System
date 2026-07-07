"""
Deterministic nightly debrief publisher.

Builds docs/debrief_latest.html + appends docs/debrief_history.json from
state/results_snapshot.json + state/picks_{date}.json — pure Python, no LLM.

WHY THIS EXISTS (Jul 7 2026): the Claude routine that used to publish the
debrief lost ALL write access — the platform now redacts secrets embedded in
routine prompts and policy-blocks git/MCP/HTTP writes from routine sandboxes.
This module runs inside the Results Snapshot workflow (GitHub's runners, full
write access) so the PWA debrief page, the Signal Scorecard history, and the
push notification (via notify_debrief.yml, which fires when debrief_latest.html
is pushed) never depend on the routine again. The routine still runs and can
add narrative color as an in-app artifact, but nothing requires it.

Everything here is derivable without judgment: results/scores/CLV come from
the snapshot, stakes from the picks file, signal_verdicts from the fixed
result×CLV templates, canonical signal names from a regex table, and the
patterns/observations from simple aggregation. The only field the LLM version
did better is game narratives (what_happened) — replaced by a score-based
sentence.

Usage (from repo root): python -m src.report.debrief_builder
Exit 0 always intended for workflow use; failures print and exit 0 (non-fatal).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = Path("state/results_snapshot.json")
DOCS = Path("docs")
HISTORY = DOCS / "debrief_history.json"
LATEST = DOCS / "debrief_latest.html"

# ── Canonical signal mapping (mirrors the routine's vocabulary) ──────────────
_SIGNAL_MAP = [
    (re.compile(r"Validated pattern", re.I), "Starter Edge"),
    (re.compile(r"ERA trap", re.I), "ERA Trap"),
    (re.compile(r"injury impact|lineup impact|injuries benefit|⚕", re.I), "Injury Impact"),
    (re.compile(r"Park factor", re.I), "Park Factor"),
    (re.compile(r"Recent form|momentum", re.I), "Recent Form"),
    (re.compile(r"schedule load", re.I), "Schedule Load"),
    (re.compile(r"K matchup", re.I), "Pitcher Matchup"),
    (re.compile(r"Umpire|👨‍⚖️", re.I), "Umpire Factor"),
    (re.compile(r"Weather|🌤|wind", re.I), "Weather Factor"),
    (re.compile(r"Platoon", re.I), "Platoon Edge"),
    (re.compile(r"NetRtg|net rating|Rating edge", re.I), "Rating Edge"),
    (re.compile(r"rest advantage|back-to-back|b2b", re.I), "Rest Advantage"),
    (re.compile(r"Elo", re.I), "Elo Edge"),
    (re.compile(r"altitude", re.I), "Altitude Edge"),
    (re.compile(r"host nation", re.I), "Host Nation Advantage"),
    (re.compile(r"\bxG\b", re.I), "xG Edge"),
    (re.compile(r"dead rubber", re.I), "Dead Rubber"),
    (re.compile(r"goalie", re.I), "Goalie Edge"),
    (re.compile(r"home advantage|home edge", re.I), "Home Advantage"),
]
# Raw lines that are stats/debug, never signals.
_SIGNAL_SKIP = re.compile(
    r"Model projected score|Model expected|FIP [\d.]|xFIP [\d.]|tanh cap", re.I)


def _canonical_signals(raw_signals):
    out = []
    for s in raw_signals or []:
        if _SIGNAL_SKIP.search(s or ""):
            continue
        for rx, name in _SIGNAL_MAP:
            if rx.search(s or ""):
                if name not in out:
                    out.append(name)
                break
        if len(out) >= 4:
            break
    return out or ["Model Projection"]


def _verdict(result, clv_pct):
    """Fixed result × CLV templates (identical to the routine's rules)."""
    if result not in ("WON", "LOST"):
        return ""
    if clv_pct is None:
        return ("CORRECT — won (no closing line captured)" if result == "WON"
                else "INCORRECT — lost (no closing line captured)")
    c = clv_pct
    if result == "WON":
        if c >= 1.0:
            return f"CORRECT — won and beat the close (+{c:.1f}%)"
        if c > -1.0:
            return f"CORRECT — won; line movement flat ({c:+.1f}%, noise range)"
        return f"MIXED — won, but the market moved {c:.1f}% against this pick"
    if c >= 1.0:
        return f"CORRECT — beat the close (+{c:.1f}%), lost on variance"
    if c > -1.0:
        return f"INCORRECT — lost; line movement flat ({c:+.1f}%)"
    return f"INCORRECT — lost AND the market moved {c:.1f}% against us"


def _pnl_str(v):
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"


def _why(pick_meta, canon):
    lead = canon[0] if canon else "Model Projection"
    raw = ""
    for s in pick_meta.get("signals", []):
        if not _SIGNAL_SKIP.search(s or ""):
            raw = s
            break
    return f"Primary signal: {lead}." + (f" ({raw})" if raw else "")


def build(today=None):
    if not SNAPSHOT.exists():
        print("no results_snapshot.json — nothing to build")
        return

    snap = json.loads(SNAPSHOT.read_text())
    date = snap.get("date")
    if not date:
        print("snapshot has no date — skipping")
        return
    picks_path = Path(f"state/picks_{date}.json")
    if not picks_path.exists():
        print(f"no picks file for {date} — skipping")
        return
    state = json.loads(picks_path.read_text())

    # Lookups from snapshot
    res_lookup = {(r["game"], r["bet_type"], r["pick"]): r
                  for r in snap.get("singles", []) + snap.get("watchlist", [])}
    parlay_lookup = {p.get("label", ""): p for p in snap.get("parlays", [])}

    def rget(p):
        return res_lookup.get((p.get("game", ""), p.get("bet_type", ""), p.get("pick", "")),
                              {"result": "PENDING", "score": "Score not available"})

    singles = state.get("singles", [])
    parlays = state.get("parlays", [])
    budget_keys = {(p.get("game"), p.get("bet_type"), p.get("pick")) for p in singles}
    display_pools = (state.get("singles_display", []) + state.get("wnba_display", []) +
                     state.get("mls_display", []) + state.get("wc_display", []) +
                     state.get("ipl_display", []))
    watchlist = [p for p in display_pools
                 if (p.get("game"), p.get("bet_type"), p.get("pick")) not in budget_keys]

    picks_out = []
    budget_results = []
    budget_clvs = []

    for p in singles:
        r = rget(p)
        result = r.get("result", "PENDING")
        clv_pct = r.get("clv_pct")
        canon = _canonical_signals(p.get("signals"))
        pnl = ""
        if result == "WON":
            pnl = _pnl_str(float(p.get("profit_if_win", 0) or 0))
        elif result == "LOST":
            pnl = _pnl_str(-float(p.get("total_cost", 0) or 0))
        elif result in ("PUSH", "VOID"):
            pnl = "$0.00"
        budget_results.append((result, float(p.get("profit_if_win", 0) or 0),
                               float(p.get("total_cost", 0) or 0)))
        if clv_pct is not None:
            budget_clvs.append(clv_pct)
        picks_out.append({
            "game": p.get("game", ""), "sport": p.get("sport", ""),
            "pick": p.get("pick", ""), "bet_type": p.get("bet_type", ""),
            "confidence": p.get("confidence", ""),
            "result": result, "score": r.get("score", ""), "pnl": pnl,
            **({"clv_pct": clv_pct} if clv_pct is not None else {}),
            "why_picked": _why(p, canon),
            "what_happened": (f"Final: {r.get('score','')}." if result in ("WON", "LOST", "PUSH")
                              else ""),
            "signals": canon,
            "signal_verdict": _verdict(result, clv_pct),
            "key_insight": "",
            "in_budget": True,
        })

    for par in parlays:
        sp = parlay_lookup.get(par.get("label", ""), {})
        result = sp.get("result", "PENDING")
        pnl = ""
        if result == "WON":
            pnl = _pnl_str(float(par.get("profit_if_win", 0) or 0))
        elif result == "LOST":
            pnl = _pnl_str(-float(par.get("total_cost", 0) or 0))
        elif result in ("PUSH", "VOID"):
            pnl = "$0.00"
        budget_results.append((result, float(par.get("profit_if_win", 0) or 0),
                               float(par.get("total_cost", 0) or 0)))
        leg_sigs = []
        for leg in par.get("legs", []):
            for s in _canonical_signals(leg.get("signals")):
                if s not in leg_sigs:
                    leg_sigs.append(s)
        legs_out = [{"pick": l.get("pick", ""), "sport": l.get("sport", "MLB"),
                     "result": (l2.get("result", "PENDING") if l2 else "PENDING")}
                    for l, l2 in zip(par.get("legs", []),
                                     (sp.get("legs") or [None] * len(par.get("legs", []))))]
        picks_out.append({
            "game": " / ".join(l.get("game", "") for l in par.get("legs", [])),
            "pick": par.get("label", ""), "sport": "PARLAY", "bet_type": "Parlay",
            "result": result, "pnl": pnl, "legs": legs_out,
            "signals": leg_sigs[:4] or ["Model Projection"],
            "signal_verdict": ("CORRECT — all legs hit" if result == "WON"
                               else ("INCORRECT — a leg missed" if result == "LOST" else "")),
            "key_insight": ("Settled as a reduced parlay (postponed leg)"
                            if result == "WON" and sp.get("void_legs") else ""),
            "in_budget": True,
        })

    wl_w = wl_l = 0
    for p in watchlist:
        r = rget(p)
        result = r.get("result", "PENDING")
        if result == "WON":
            wl_w += 1
        elif result == "LOST":
            wl_l += 1
        clv_pct = r.get("clv_pct")
        canon = _canonical_signals(p.get("signals"))
        picks_out.append({
            "game": p.get("game", ""), "sport": p.get("sport", ""),
            "pick": p.get("pick", ""), "bet_type": p.get("bet_type", ""),
            "confidence": p.get("confidence", ""),
            "result": result, "score": r.get("score", ""), "pnl": "",
            **({"clv_pct": clv_pct} if clv_pct is not None else {}),
            "why_picked": _why(p, canon),
            "what_happened": (f"Final: {r.get('score','')}." if result in ("WON", "LOST") else ""),
            "signals": canon,
            "signal_verdict": _verdict(result, clv_pct),
            "key_insight": "", "in_budget": False,
        })

    won = sum(1 for r, _, _ in budget_results if r == "WON")
    lost = sum(1 for r, _, _ in budget_results if r == "LOST")
    pending = sum(1 for r, _, _ in budget_results if r == "PENDING")
    if won + lost == 0:
        print(f"nothing settled yet for {date} — not publishing (morning run will cover)")
        return
    budget_pnl = round(sum((profit if r == "WON" else -cost if r == "LOST" else 0.0)
                           for r, profit, cost in budget_results), 2)
    avg_clv = round(sum(budget_clvs) / len(budget_clvs), 2) if budget_clvs else None
    clv_str = f"{avg_clv:+.2f}%" if avg_clv is not None else "n/a"

    headline = (f"{won}-{lost} on budget ({_pnl_str(budget_pnl)})"
                + (f", avg CLV {clv_str}" if avg_clv is not None else "")
                + (f"; {pending} pending" if pending else ""))

    # Patterns + per-signal observations (settled budget singles only)
    settled_singles = [p for p in picks_out
                       if p["in_budget"] and p.get("bet_type") != "Parlay"
                       and p["result"] in ("WON", "LOST")]
    patterns = []
    for conf in ("HIGH", "MEDIUM"):
        seg = [p for p in settled_singles if p.get("confidence") == conf]
        if seg:
            w = sum(1 for p in seg if p["result"] == "WON")
            patterns.append(f"{conf} confidence: {w}-{len(seg) - w} today")
    by_market = {}
    for p in settled_singles:
        by_market.setdefault(f"{p['sport']} {p['bet_type']}", []).append(p)
    for mk, seg in by_market.items():
        w = sum(1 for p in seg if p["result"] == "WON")
        patterns.append(f"{mk}: {w}-{len(seg) - w}")

    observations = []
    sig_groups = {}
    for p in settled_singles:
        for s in p.get("signals", []):
            sig_groups.setdefault(s, []).append(p)
    for sig, seg in sorted(sig_groups.items(), key=lambda kv: -len(kv[1])):
        if len(seg) < 2 and sig != "Starter Edge":
            continue
        w = sum(1 for p in seg if p["result"] == "WON")
        clvs = [p["clv_pct"] for p in seg if p.get("clv_pct") is not None]
        obs = f"{sig} picks: {w}-{len(seg) - w}"
        if clvs:
            obs += f" with avg CLV {sum(clvs) / len(clvs):+.1f}%"
        observations.append(obs)

    entry = {
        "date": date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "workflow",   # deterministic builder, not the LLM routine
        "headline": headline,
        "budget_record": f"{won}-{lost}",
        "watchlist_record": f"{wl_w}-{wl_l}",
        "budget_pnl": budget_pnl,
        "avg_budget_clv": avg_clv,
        "snapshot_timeout": False,
        "picks": picks_out,
        "patterns": patterns,
        "model_observations": observations,
    }

    try:
        hist = json.loads(HISTORY.read_text())
    except Exception:
        hist = {"entries": []}
    hist["entries"] = [e for e in hist.get("entries", []) if e.get("date") != date]
    hist["entries"].append(entry)
    hist["entries"] = hist["entries"][-60:]
    HISTORY.write_text(json.dumps(hist, indent=2))

    LATEST.write_text(_render_html(entry, clv_str))
    print(f"debrief built for {date}: {won}-{lost} ({_pnl_str(budget_pnl)}), "
          f"{pending} pending, {len(picks_out)} picks total")


# ── HTML rendering ────────────────────────────────────────────────────────────
_RESULT_COLOR = {"WON": "#22c55e", "LOST": "#ef4444", "PENDING": "#f59e0b",
                 "PUSH": "#94a3b8", "VOID": "#94a3b8"}


def _clv_color(c):
    if c is None:
        return "#94a3b8"
    return "#22c55e" if c > 0 else ("#ef4444" if c <= -1.0 else "#94a3b8")


def _esc(s):
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _card(p, brief=False):
    col = _RESULT_COLOR.get(p["result"], "#94a3b8")
    clv = p.get("clv_pct")
    clv_tag = (f'<span style="font-size:0.7rem;font-weight:700;color:{_clv_color(clv)};'
               f'margin-left:8px;">CLV {clv:+.1f}%</span>' if clv is not None else "")
    h = [f'<div style="background:#1a1d27;border-left:3px solid {col};border-radius:10px;'
         f'padding:12px 14px;margin-bottom:10px;">']
    h.append(f'<div style="display:flex;justify-content:space-between;align-items:center;">'
             f'<span style="color:#38bdf8;font-weight:700;">{_esc(p["pick"])}</span>'
             f'<span><span style="color:{col};font-weight:800;font-size:0.78rem;">'
             f'{p["result"]}</span>{clv_tag}</span></div>')
    sub = _esc(p.get("game", ""))
    if p.get("pnl"):
        sub += f' &nbsp;·&nbsp; {p["pnl"]}'
    h.append(f'<div style="color:#64748b;font-size:0.76rem;margin-top:3px;">{sub}</div>')
    if p.get("score"):
        h.append(f'<div style="color:#e2e8f0;font-size:0.8rem;margin-top:5px;">{_esc(p["score"])}</div>')
    if p.get("legs"):
        for l in p["legs"]:
            lc = _RESULT_COLOR.get(l.get("result", "PENDING"), "#94a3b8")
            h.append(f'<div style="font-size:0.78rem;margin-top:4px;">• {_esc(l["pick"])} '
                     f'<span style="color:{lc};font-weight:700;">{l.get("result","")}</span></div>')
    if not brief:
        if p.get("why_picked"):
            h.append(f'<div style="color:#94a3b8;font-size:0.76rem;margin-top:6px;">{_esc(p["why_picked"])}</div>')
        if p.get("signal_verdict"):
            h.append(f'<div style="color:#e2e8f0;font-size:0.76rem;margin-top:4px;">'
                     f'<strong>Verdict:</strong> {_esc(p["signal_verdict"])}</div>')
    elif p.get("signal_verdict"):
        h.append(f'<div style="color:#94a3b8;font-size:0.74rem;margin-top:4px;">{_esc(p["signal_verdict"])}</div>')
    if p.get("signals"):
        chips = "".join(f'<span style="background:#0f1219;border:1px solid #2a3040;color:#94a3b8;'
                        f'border-radius:20px;padding:1px 8px;font-size:0.66rem;margin-right:5px;">'
                        f'{_esc(s)}</span>' for s in p["signals"])
        h.append(f'<div style="margin-top:7px;">{chips}</div>')
    h.append("</div>")
    return "".join(h)


def _render_html(e, clv_str):
    budget = [p for p in e["picks"] if p["in_budget"]]
    wl = [p for p in e["picks"] if not p["in_budget"]]
    wl_by_sport = {}
    for p in wl:
        wl_by_sport.setdefault(p.get("sport", "?"), []).append(p)

    chips = [
        ("Today's Card", e["budget_record"]),
        ("Watchlist", e["watchlist_record"]),
        ("Budget P&L", _pnl_str(e["budget_pnl"])),
        ("Avg CLV", clv_str),
    ]
    chip_html = "".join(
        f'<div style="background:#1a1d27;border-radius:8px;padding:8px 12px;text-align:center;">'
        f'<div style="color:#64748b;font-size:0.62rem;text-transform:uppercase;">{k}</div>'
        f'<div style="color:#e2e8f0;font-weight:800;font-size:0.95rem;">{v}</div></div>'
        for k, v in chips)

    def section(title, inner):
        return (f'<div style="color:#64748b;font-size:0.7rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin:20px 0 10px;">{title}</div>'
                + inner)

    wl_html = ""
    for sport, ps in wl_by_sport.items():
        wl_html += (f'<div style="color:#38bdf8;font-size:0.72rem;font-weight:700;'
                    f'margin:10px 0 6px;">{_esc(sport)}</div>'
                    + "".join(_card(p, brief=True) for p in ps))

    bullets = lambda items: ("<ul style='color:#e2e8f0;font-size:0.8rem;line-height:1.7;"
                             "padding-left:18px;margin:0;'>"
                             + "".join(f"<li>{_esc(i)}</li>" for i in items) + "</ul>") if items else \
        "<div style='color:#64748b;font-size:0.78rem;'>—</div>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nightly Debrief — {e['date']}</title></head>
<body style="background:#050507;color:#e2e8f0;font-family:-apple-system,system-ui,sans-serif;
margin:0;padding:16px;">
<div style="max-width:600px;margin:0 auto;">
<div style="background:#1a1d27;border-radius:12px;padding:16px;margin-bottom:14px;">
<div style="font-size:1.05rem;font-weight:800;">📊 Nightly Debrief — {e['date']}</div>
<div style="color:#94a3b8;font-size:0.82rem;margin-top:6px;">{_esc(e['headline'])}</div>
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px;">{chip_html}</div>
<div style="color:#64748b;font-size:0.64rem;margin-top:10px;">Generated automatically from the
Results Snapshot (deterministic — scores, results and CLV are official).</div>
</div>
{section("Today's Card — Budget Picks", "".join(_card(p) for p in budget))}
{section("Watchlist — All Sport Tabs", wl_html)}
{section("Patterns Today", bullets(e['patterns']))}
{section("Model Observations", bullets(e['model_observations']))}
<div style="color:#64748b;font-size:0.68rem;text-align:center;margin:24px 0 8px;">
Generated {e['generated_at']} · <a href="/System/index_spa.html" style="color:#38bdf8;">← Back to Picks</a></div>
</div></body></html>"""


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"debrief build failed (non-fatal): {exc}")
    sys.exit(0)
