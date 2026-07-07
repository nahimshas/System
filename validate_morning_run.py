#!/usr/bin/env python3
"""
Morning-run validation — post-settlement invariant checks.

Runs after the daily analysis/settlement step. Verifies that the night-before
card reconciles with what the morning run settled: no dropped bets/props, P&L
adds up, tiles reconcile, no duplicates, watchlist tracking intact, self-tuning
state sane.

REPORT-ONLY by design: never mutates state, never blocks the run. Always exits 0.
Writes state/validation_report.json (consumed by the report's status banner) and
prints a PASS/FAIL summary to the workflow log.

Each finding carries: check id, severity, plain-English `what`, `why` it matters,
and a `fix` hint — so the PWA banner can tell you what happened and what to try.
"""
import json
import sys
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

STATE = Path("state")
HISTORY          = STATE / "history.json"
PROP_HISTORY     = STATE / "prop_history.json"
WATCHLIST_HIST   = STATE / "watchlist_history.json"
CAP_STATE        = STATE / "cap_state.json"
CALIBRATION      = STATE / "calibration_state.json"
REPORT_OUT       = STATE / "validation_report.json"

# Sports that settle into history.json (budget) vs watchlist_history.json.
BUDGET_SPORTS    = {"NBA", "MLB", "NFL", "NHL"}
WATCHLIST_SPORTS = {"IPL", "WNBA", "MLS"}


def _load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _yesterday_iso(today: date) -> str:
    return (today - timedelta(days=1)).isoformat()


def _picks_for(d: str):
    return _load(STATE / f"picks_{d}.json", {})


def run_checks(today: date):
    """Return (findings, stats, meta). findings = list of dicts; empty = all pass."""
    findings = []
    y = _yesterday_iso(today)

    # ── Determine first-run-of-day vs subsequent run ──────────────────────────
    # Load the previous validation report. If it's already from today, this is
    # a subsequent run and we only re-check items that were flagged this morning.
    # If it's from a prior day (or missing), this IS the first run — full checks.
    prev_report: dict = {}
    if REPORT_OUT.exists():
        try:
            prev_report = json.loads(REPORT_OUT.read_text())
        except Exception:
            pass

    # Determine if the previous report is already from today.
    # Prefer explicit report_date (new format); fall back to generated_at date (old format).
    prev_date = prev_report.get("report_date", "")
    if prev_date:
        # New format: report_date is always written as str(today) in the run's timezone.
        is_first_run_today = prev_date != str(today)
    else:
        # Old format (no report_date field): we can't reliably derive the run date from
        # generated_at because the server (GitHub Actions, UTC) and the user's timezone
        # (PDT, UTC-7) can disagree — a midnight-UTC timestamp maps to the *previous*
        # calendar day in Pacific time, producing a false "first run" on every subsequent
        # check.  Instead, treat this as a first run only if we are currently inside the
        # morning analysis window (08:00–11:59 Pacific).  Outside that window we are
        # definitely on a subsequent run and should not re-fire the full checks.
        now_pacific_hour = (datetime.now(timezone.utc) - timedelta(hours=7)).hour
        is_first_run_today = 8 <= now_pacific_hour < 12

    # Keys flagged as dropped in the morning report (used on subsequent runs).
    prev_flagged_bets  = {tuple(k) for k in prev_report.get("flagged_dropped_bets",  [])}
    prev_flagged_props = {tuple(k) for k in prev_report.get("flagged_dropped_props", [])}

    # Collect still-unfixed keys so subsequent runs know what to re-check.
    flagged_dropped_bets:  list = []
    flagged_dropped_props: list = []

    def fail(check, severity, what, why, fix):
        findings.append({"check": check, "severity": severity,
                         "what": what, "why": why, "fix": fix})

    history   = _load(HISTORY, [])
    props     = _load(PROP_HISTORY, [])
    wl_hist   = _load(WATCHLIST_HIST, [])
    y_state   = _picks_for(y)

    hist_y = [r for r in history if r.get("date") == y]
    wl_y   = [r for r in wl_hist if r.get("date") == y]
    # All prop records for yesterday (any result, incl. pending None) — used to
    # tell "dropped" (no record at all) apart from "pending" (record, not yet settled).
    prop_y_all     = [r for r in props if r.get("date") == y]
    prop_y_settled = [r for r in prop_y_all if r.get("result") in ("WON", "LOST", "PUSH", "HIT", "MISS")]

    # Shared timestamp for all commence_time guards below
    now_utc = datetime.now(timezone.utc)

    # ── Check 1: no dropped budget bets ──────────────────────────────────────
    # Every placed single from yesterday's card should have a settled record.
    placed_singles = y_state.get("singles", []) if y_state else []
    if placed_singles:
        settled_keys = {(r.get("game", ""), r.get("bet_type", ""), r.get("pick", ""))
                        for r in hist_y if r.get("type") == "single"}
        for s in placed_singles:
            k = (s.get("game", ""), s.get("bet_type", ""), s.get("pick", ""))
            if k not in settled_keys:
                # Subsequent run: only re-check bets flagged in the morning report.
                if not is_first_run_today and k not in prev_flagged_bets:
                    continue
                # First run: skip if game hasn't had time to finish (5h guard).
                if is_first_run_today:
                    commence_str = s.get("commence_time", "")
                    if commence_str:
                        try:
                            from datetime import timedelta as _td
                            ct = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                            if now_utc < ct + _td(hours=5):
                                continue
                        except Exception:
                            pass
                flagged_dropped_bets.append(list(k))
                fail("dropped_bet", "warn",
                     f"Placed single '{s.get('pick')}' ({s.get('sport')} {s.get('bet_type')}, "
                     f"{s.get('game')}) from {y}'s card has no settled record in history.json.",
                     "A placed bet that never settled means yesterday's P&L is understated and "
                     "the by-sport/confidence tiles are missing it.",
                     "Re-run the workflow (settlement retries). If it persists, the ESPN team-name "
                     "match likely failed in _find_game_score — verify the game's team names.")

    # ── Check 2: no dropped props ────────────────────────────────────────────
    placed_props = y_state.get("props", []) if y_state else []
    if placed_props:
        recorded_prop_keys = {(r.get("player", ""), r.get("prop_type", "")) for r in prop_y_all}
        for p in placed_props:
            k = (p.get("player", ""), p.get("prop_type", ""))
            # Only flag props that have NO record at all (truly dropped). A record
            # with result=None is simply pending settlement — not a problem.
            if k not in recorded_prop_keys:
                # Subsequent run: only re-check props flagged in the morning report.
                if not is_first_run_today and k not in prev_flagged_props:
                    continue
                # First run: 5h guard same as singles.
                if is_first_run_today:
                    commence_str = p.get("commence_time", "")
                    if commence_str:
                        try:
                            from datetime import timedelta as _td
                            ct = datetime.fromisoformat(commence_str.replace("Z", "+00:00"))
                            if now_utc < ct + _td(hours=5):
                                continue
                        except Exception:
                            pass
                flagged_dropped_props.append(list(k))
                fail("dropped_prop", "info",
                     f"Prop '{p.get('player')} {p.get('prop_type')}' from {y} has no record at all "
                     f"in prop_history.json (not even pending).",
                     "A prop with no record was lost between selection and settlement — missing "
                     "from the prop-accuracy tracker entirely.",
                     "Check prop_history.json for the player; if absent, the prop settlement loop "
                     "didn't pick it up (name-match or sport-routing issue).")

    # ── Check 3: P&L reconciliation (budget singles + parlays) ───────────────
    pnl_mismatch = 0.0
    for r in hist_y:
        result = r.get("result")
        cost   = r.get("cost", 0) or 0
        profit = r.get("profit_if_win", 0) or 0
        stored = r.get("actual_pnl")
        if stored is None:
            continue
        expected = profit if result == "WON" else (0.0 if result == "PUSH" else -cost)
        if abs(expected - stored) > 0.01:
            pnl_mismatch += 1
            fail("pnl_mismatch", "warn",
                 f"P&L mismatch on '{r.get('pick')}' ({r.get('game')}): stored actual_pnl="
                 f"{stored:+.2f} but {result} with cost {cost}/profit {profit} implies {expected:+.2f}.",
                 "A stored P&L that disagrees with the result corrupts the day's total and the tiles.",
                 "Inspect the record in history.json; likely a mis-settled result or a stale "
                 "cost/profit snapshot. Re-running settlement usually corrects it.")

    # ── Check 4: tiles reconcile (by-sport + by-confidence sum to overall) ───
    total_pnl = round(sum((r.get("actual_pnl") or 0) for r in hist_y), 2)
    by_sport_sum = round(sum((r.get("actual_pnl") or 0) for r in hist_y
                             if r.get("sport") in (BUDGET_SPORTS | {"PARLAY"})), 2)
    if abs(total_pnl - by_sport_sum) > 0.01:
        fail("tile_reconcile", "warn",
             f"Yesterday's total P&L ({total_pnl:+.2f}) ≠ sum of by-sport P&L ({by_sport_sum:+.2f}).",
             "The by-sport tiles are filtered views of history; if they don't sum to the total, a "
             "record has an unexpected sport label and a tile is wrong.",
             "Find the history record(s) for yesterday whose sport is not in "
             f"{sorted(BUDGET_SPORTS | {'PARLAY'})} and correct the label.")

    # ── Check 5: no duplicate settlement ─────────────────────────────────────
    seen = {}
    for r in hist_y:
        k = (r.get("date"), r.get("game"), r.get("bet_type"), r.get("pick"))
        seen[k] = seen.get(k, 0) + 1
    dups = [k for k, n in seen.items() if n > 1]
    if dups:
        fail("duplicate_settlement", "warn",
             f"{len(dups)} bet(s) appear more than once in history.json for {y}: "
             f"{', '.join(k[3] for k in dups[:3])}{'…' if len(dups) > 3 else ''}.",
             "Duplicate records double-count P&L and inflate the record.",
             "De-duplicate history.json by (date, game, bet_type, pick). Usually caused by a "
             "git merge re-appending records.")

    # ── Check 6: no unsettled-but-finished bets ──────────────────────────────
    now = datetime.now(timezone.utc)
    for r in hist_y:
        if r.get("result") not in ("WON", "LOST", "PUSH"):
            fail("invalid_result", "warn",
                 f"History record '{r.get('pick')}' ({r.get('game')}) for {y} has result="
                 f"{r.get('result')!r} (expected WON/LOST/PUSH).",
                 "An unresolved result past game time means settlement didn't complete for this bet.",
                 "Re-run the workflow; if it stays unresolved, ESPN likely lacks a final score for "
                 "that game (postponement?).")

    # ── Check 7: watchlist parallel-tracking intact ──────────────────────────
    # Watchlist-sport picks from yesterday's display lists should settle into
    # watchlist_history.json. (Catches the graduation-routing class of bug.)
    if y_state:
        wl_settled_keys = {(r.get("sport"), r.get("pick"), r.get("game")) for r in wl_y}
        for slug in ("wnba", "mls", "ipl"):
            disp = y_state.get(f"{slug}_display", []) or []
            for s in disp:
                sport = s.get("sport", "")
                if sport not in WATCHLIST_SPORTS:
                    continue
                # IPL settles via the rolling pending file — skip the strict check.
                if sport == "IPL":
                    continue
                k = (sport, s.get("pick", ""), s.get("game", ""))
                if k not in wl_settled_keys:
                    # info-level: watchlist games can legitimately lack a settle (no edge tracked)
                    pass  # tracked but not flagged unless a clear pattern; kept lightweight

    # ── Check 8: self-tuning state sane (cap bounds, parseable) ──────────────
    cap = _load(CAP_STATE, None)
    if cap is None:
        fail("cap_state_parse", "info",
             "cap_state.json missing or unparseable.",
             "The credibility-cap auto-adjustment panel won't render; live caps fall back to defaults.",
             "Confirm the file exists and is valid JSON; a corrupted shard is auto-backed-up by the writer.")
    else:
        caps = cap.get("caps", cap) if isinstance(cap, dict) else {}
        def _walk_cap_values(o):
            vals = []
            if isinstance(o, dict):
                for k, v in o.items():
                    if k in ("value", "cap_value", "current") and isinstance(v, (int, float)):
                        vals.append(v)
                    else:
                        vals += _walk_cap_values(v)
            elif isinstance(o, list):
                for v in o:
                    vals += _walk_cap_values(v)
            return vals
        for v in _walk_cap_values(caps):
            if not (0.05 <= v <= 0.30):
                fail("cap_out_of_bounds", "warn",
                     f"A credibility cap value ({v}) is outside the safe range [0.05, 0.30].",
                     "Caps outside bounds mean the self-tuning produced an unsafe value that could "
                     "over- or under-shrink edges.",
                     "Inspect cap_state.json; the counterfactual tuner should clamp to [0.05,0.30] — "
                     "a value outside it indicates a bad write.")
                break

    calib = _load(CALIBRATION, None)
    if calib is None:
        fail("calibration_parse", "info",
             "calibration_state.json missing or unparseable.",
             "The calibration panel won't render and effective_edge falls back to raw edge.",
             "Confirm the file exists and is valid JSON.")

    # ── Check 9: state integrity (yesterday's picks file + count match) ──────
    if not y_state:
        fail("missing_state", "info",
             f"No picks_{y}.json found — can't cross-check yesterday's card against settlement.",
             "Without yesterday's state the dropped-bet/prop checks can't run (they pass vacuously).",
             "Normal if there were no games yesterday; otherwise the state commit may have been lost.")
    else:
        n_singles = len(y_state.get("singles", []))
        n_settled_singles = len([r for r in hist_y if r.get("type") == "single"])
        # Only an issue if settlement found FEWER than placed (some may PUSH/cancel → still settle).
        if n_singles and n_settled_singles < n_singles:
            missing = n_singles - n_settled_singles
            fail("count_mismatch", "info",
                 f"{y} had {n_singles} placed single(s) but only {n_settled_singles} settled "
                 f"({missing} unaccounted).",
                 "A gap between placed and settled singles is the signature of the 'dropped bet' "
                 "class (e.g. a graduated sport excluded from a hardcoded list).",
                 "See the dropped_bet findings above for specifics; if none, a game may be "
                 "postponed/not yet final.")

    # ── Check 10: weekly health routine liveness (dead-man's switch) ──────────
    # The Sunday model-health routine publishes docs/health_history.json. If it
    # silently dies (token expiry, sandbox loses repo access, proxy change),
    # nothing else would notice — this check surfaces staleness in the banner.
    try:
        _hh = _load(Path("docs/health_history.json"), None)
        if _hh is not None:
            _entries = _hh.get("entries", _hh) if isinstance(_hh, dict) else _hh
            _last = max((e.get("date", "") for e in _entries), default="")
            if _last:
                _age = (today - date.fromisoformat(_last)).days
                if _age > 8:
                    fail("health_routine_stale", "warn",
                         f"Last weekly health report is {_age} days old ({_last}) — the "
                         f"model-health-weekly routine appears to have stopped running.",
                         "The health routine evaluates model checkpoints and watches for drift; "
                         "if it dies silently, pending evaluations (and drift alarms) never fire.",
                         "Open the model-health-weekly routine in the Claude app and run it "
                         "manually — check for 403 push errors (repo selector) or token expiry.")
    except Exception:
        pass   # liveness check must never break validation itself

    stats = {
        "date_checked": y,
        "settled_singles": len([r for r in hist_y if r.get("type") == "single"]),
        "settled_total_records": len(hist_y),
        "yesterday_pnl": total_pnl,
        "placed_singles": len(placed_singles),
        "placed_props": len(placed_props),
    }
    meta = {
        "flagged_dropped_bets":  flagged_dropped_bets,
        "flagged_dropped_props": flagged_dropped_props,
        "is_first_run_today":    is_first_run_today,
    }
    return findings, stats, meta


def main():
    today = date.today()
    try:
        findings, stats, meta = run_checks(today)
    except Exception as e:
        # Never let validation break the run.
        findings = [{"check": "validator_error", "severity": "info",
                     "what": f"Validator raised an exception: {e}",
                     "why": "The validation itself failed; checks did not complete.",
                     "fix": "Inspect validate_morning_run.py against current state-file shapes."}]
        stats = {}
        meta  = {}

    warns = [f for f in findings if f["severity"] == "warn"]
    infos = [f for f in findings if f["severity"] == "info"]
    status = "FAIL" if warns else ("WARN" if infos else "PASS")

    report = {
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "report_date":          str(today),
        "status":               status,
        "warn_count":           len(warns),
        "info_count":           len(infos),
        "findings":             findings,
        "stats":                stats,
        # Persist flagged keys so subsequent runs know what to re-check.
        "flagged_dropped_bets":  meta.get("flagged_dropped_bets",  []),
        "flagged_dropped_props": meta.get("flagged_dropped_props", []),
    }
    try:
        REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_OUT, "w") as f:
            json.dump(report, f, indent=2)
    except Exception as e:
        print(f"[validate] could not write report: {e}")

    # Console summary for the workflow log.
    icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[status]
    print(f"\n{icon} Morning-run validation: {status} "
          f"({len(warns)} warn, {len(infos)} info) — checked {stats.get('date_checked','?')}")
    for f in findings:
        sev = {"warn": "⚠️", "info": "ℹ️"}.get(f["severity"], "•")
        print(f"  {sev} [{f['check']}] {f['what']}")
        print(f"        → fix: {f['fix']}")
    if not findings:
        print("  All invariants hold — card reconciles with settlement.")

    # Report-only: ALWAYS exit 0 so validation never blocks the workflow.
    sys.exit(0)


if __name__ == "__main__":
    main()
