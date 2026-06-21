"""
Results Snapshot — resolves TODAY's picks against ESPN and writes
state/results_snapshot.json for the nightly debrief routine to consume.

Run via the Results Snapshot workflow (workflow_dispatch), triggered by the
nightly debrief agent right before it builds the report. Unlike the morning
outcome checker (which settles YESTERDAY's picks into history.json), this
script resolves today's picks at debrief time — including in-progress games
(PENDING) and postponed games (VOID) — and never writes to history.json.

Every result the debrief displays comes from this file, so score resolution
is deterministic Python instead of an LLM agent executing code blocks.
"""
import json
import logging
import os
import sys
import requests
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.data.outcome_checker import (
    ESPN_BASE,
    ESPN_SPORT_PATHS,
    ESPN_WATCHLIST_PATHS,
    _determine_outcome,
    _determine_mls_outcome,
    _team_token_overlap_match,
    _soccer_90min_scores,
    _SOCCER_90MIN_SPORTS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SNAPSHOT_PATH = Path("state/results_snapshot.json")

# Combined sport → ESPN path map (budget + watchlist sports)
ALL_ESPN_PATHS = {**ESPN_SPORT_PATHS, **ESPN_WATCHLIST_PATHS}

# IPL games finish after midnight Pacific — always PENDING at debrief time.
SKIP_SPORTS = {"IPL"}


def _today_pacific() -> date:
    now_utc = datetime.now(timezone.utc)
    offset = -7 if 3 <= now_utc.month <= 10 else -8
    return (now_utc + timedelta(hours=offset)).date()


def _load_picks_state() -> tuple:
    """Load today's picks file, falling back to yesterday (after-midnight runs)."""
    for d in [_today_pacific(), _today_pacific() - timedelta(days=1)]:
        path = Path(f"state/picks_{d.isoformat()}.json")
        if path.exists():
            with open(path) as f:
                return json.load(f), d
    return None, None


def _fetch_events(sport: str, game_date: date) -> list:
    """Fetch ALL events (any status) for a sport on a date. Returns entry list."""
    path = ALL_ESPN_PATHS.get(sport)
    if not path:
        return []
    # Soccer is ESPN-dated by UTC, so an evening-Pacific match rolls into the
    # next date — query a 2-day window so those games settle (see outcome_checker).
    if sport in ("MLS", "WC"):
        date_str = f"{game_date.strftime('%Y%m%d')}-{(game_date + timedelta(days=1)).strftime('%Y%m%d')}"
    else:
        date_str = game_date.strftime("%Y%m%d")
    url = f"{ESPN_BASE}/{path}/scoreboard"
    try:
        r = requests.get(url, params={"dates": date_str}, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"ESPN fetch failed ({sport}, {game_date}): {e}")
        return []

    entries = []
    for event in data.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        status = comp.get("status", {}).get("type", {})
        completed = bool(status.get("completed", False))
        status_text = (status.get("name", "") + " " + status.get("description", "")).lower()
        postponed = "postponed" in status_text or "canceled" in status_text or "cancelled" in status_text

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        try:
            home_score = float(home.get("score", 0) or 0)
            away_score = float(away.get("score", 0) or 0)
        except (TypeError, ValueError):
            home_score = away_score = 0.0

        # Soccer knockouts: grade 90-minute markets on the 90-minute score.
        # If the game went to ET/pens and the half breakdown is unavailable,
        # treat as not-completed → the pick stays PENDING (honest PENDING
        # beats grading against the wrong number).
        if sport in _SOCCER_90MIN_SPORTS and completed:
            h90, a90, ok90 = _soccer_90min_scores(comp, home, away)
            if ok90:
                home_score, away_score = h90, a90
            else:
                completed = False

        entries.append({
            "home_name":  home.get("team", {}).get("displayName", ""),
            "away_name":  away.get("team", {}).get("displayName", ""),
            "home_score": home_score,
            "away_score": away_score,
            "completed":  completed,
            "postponed":  postponed,
            "event_date": event.get("date", ""),
        })
    return entries


def _names_match(query: str, key: str) -> bool:
    q, k = query.lower(), key.lower()
    if not q or not k:
        return False
    return q in k or k in q or _team_token_overlap_match(query, key)


def _within_hours(iso_a: str, iso_b: str, hours: float = 8.0) -> bool:
    """True if two ISO timestamps are within `hours` of each other (or unparseable)."""
    if not iso_a or not iso_b:
        return True
    try:
        a = datetime.fromisoformat(iso_a.replace("Z", "+00:00"))
        b = datetime.fromisoformat(iso_b.replace("Z", "+00:00"))
        return abs((a - b).total_seconds()) <= hours * 3600
    except Exception:
        return True


def _find_event(events: list, home_team: str, away_team: str, commence_time: str):
    """Find the event matching a pick's teams AND start time (8h guard)."""
    for e in events:
        if (_names_match(home_team, e["home_name"]) and _names_match(away_team, e["away_name"])) or \
           (_names_match(home_team, e["away_name"]) and _names_match(away_team, e["home_name"])):
            if _within_hours(e["event_date"], commence_time):
                return e
    return None


def _resolve(sport: str, pick: str, bet_type: str, home_team: str, away_team: str,
             commence_time: str, events_by_sport: dict) -> dict:
    """Resolve one pick → {result, score} with result in WON/LOST/PUSH/PENDING/VOID."""
    if sport in SKIP_SPORTS:
        return {"result": "PENDING", "score": "Settles next morning (IPL)"}

    event = _find_event(events_by_sport.get(sport, []), home_team, away_team, commence_time)
    if event is None:
        return {"result": "PENDING", "score": "Score not available"}
    if event["postponed"]:
        return {"result": "VOID", "score": "Postponed"}

    score_str = (f"{event['away_name']} {int(event['away_score'])}, "
                 f"{event['home_name']} {int(event['home_score'])}")
    if not event["completed"]:
        return {"result": "PENDING", "score": f"In progress ({score_str})"}

    # ESPN home/away orientation may be flipped vs the pick's metadata — score
    # the outcome with ESPN's own team names so spreads/totals stay correct.
    if sport in ("MLS", "WC"):
        result = _determine_mls_outcome(pick, bet_type, event["home_name"], event["away_name"],
                                        event["home_score"], event["away_score"])
    else:
        result = _determine_outcome(pick, bet_type, event["home_name"], event["away_name"],
                                    event["home_score"], event["away_score"])
    if result not in ("WON", "LOST", "PUSH"):
        return {"result": "PENDING", "score": score_str}
    return {"result": result, "score": score_str}


def main() -> int:
    state, picks_date = _load_picks_state()
    if state is None:
        logger.error("No picks state file found for today or yesterday")
        # Still write a snapshot so the routine sees a definitive "no picks" answer
        SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump({
                "date": None,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "error": "no_picks_state",
                "singles": [], "parlays": [], "watchlist": [],
            }, f, indent=2)
        return 0

    logger.info(f"Resolving picks for {picks_date}")

    # ── CLV capture (self-healing) ────────────────────────────────────────
    # Stamp closing-line probabilities onto shadow log entries whose games
    # have started, then build a lookup so each resolved pick below carries
    # its CLV. Historical archive data persists, so a failed night here is
    # automatically repaired on the next run (or the morning run). Non-fatal.
    clv_lookup = {}
    _clv_disabled = os.environ.get("DISABLE_CLV_STAMPING", "").lower() in ("1", "true", "yes")
    if _clv_disabled:
        logger.info("CLV stamping disabled via DISABLE_CLV_STAMPING — skipping capture")
    else:
        try:
            from src.data.closing_lines import (
                update_all_clv,
                repair_missing_commence_times, clv_lookup_for_date,
            )
            repair_missing_commence_times(max_credits=20)
            clv_summary = update_all_clv(max_credits=1000, lookback_days=7)
            logger.info(f"CLV capture summary: {clv_summary}")
            clv_lookup = clv_lookup_for_date(picks_date)
        except Exception as e:
            logger.warning(f"CLV capture failed (non-fatal): {e}")

    # Collect every pick list the debrief shows
    singles = state.get("singles", [])
    parlays = state.get("parlays", [])
    display_pools = (
        state.get("singles_display", []) +
        state.get("wnba_display", []) +
        state.get("mls_display", []) +
        state.get("wc_display", []) +
        state.get("ipl_display", [])
    )

    # Sports needed
    sports = set()
    for p in singles + display_pools:
        sports.add(p.get("sport", "").upper())
    for par in parlays:
        for leg in par.get("legs", []):
            sports.add(leg.get("sport", "").upper())

    events_by_sport = {}
    for sport in sports & set(ALL_ESPN_PATHS.keys()):
        if sport in SKIP_SPORTS:
            continue
        events_by_sport[sport] = _fetch_events(sport, picks_date)
        logger.info(f"ESPN {sport}: {len(events_by_sport[sport])} event(s) for {picks_date}")

    def resolve_pick(p: dict) -> dict:
        sport = p.get("sport", "").upper()
        res = _resolve(sport, p.get("pick", ""), p.get("bet_type", ""),
                       p.get("home_team", ""), p.get("away_team", ""),
                       p.get("commence_time", ""), events_by_sport)
        out = {
            "game":      p.get("game", ""),
            "sport":     sport,
            "bet_type":  p.get("bet_type", ""),
            "pick":      p.get("pick", ""),
            "result":    res["result"],
            "score":     res["score"],
        }
        # Attach closing-line value when the shadow log has it. clv is the
        # probability delta (close − open): positive = we beat the close.
        clv_info = clv_lookup.get((out["game"], out["bet_type"], out["pick"]))
        if clv_info and clv_info.get("clv") is not None:
            out["clv"] = clv_info["clv"]
            out["clv_pct"] = round(clv_info["clv"] * 100, 1)
            out["market_prob_at_close"] = clv_info["market_prob_at_close"]
        return out

    budget_keys = {(p.get("game"), p.get("bet_type"), p.get("pick")) for p in singles}

    singles_out = [resolve_pick(p) for p in singles]

    parlays_out = []
    for par in parlays:
        legs_out = []
        for leg in par.get("legs", []):
            lr = resolve_pick(leg)
            legs_out.append(lr)
        leg_results = [l["result"] for l in legs_out]
        active = [r for r in leg_results if r != "VOID"]
        if any(r == "LOST" for r in active):
            result = "LOST"
        elif any(r == "PENDING" for r in active):
            result = "PENDING"
        elif active and all(r == "WON" for r in active):
            result = "WON"
        else:
            result = "PUSH"   # all void, or active legs all pushed
        parlays_out.append({
            "label":      par.get("label", ""),
            "result":     result,
            "void_legs":  leg_results.count("VOID"),
            "legs":       legs_out,
        })

    # Update rolling bankroll from today's settled singles AND parlays.
    # P&L compounds day-over-day: tomorrow's budget = today's bankroll ± results.
    try:
        from src.state.bankroll import load_bankroll, save_bankroll
        bankroll = load_bankroll()
        pnl = 0.0
        settled_count = 0
        for orig, resolved in zip(singles, singles_out):
            result = resolved["result"]
            if result == "WON":
                pnl += float(orig.get("profit_if_win", 0))
                settled_count += 1
            elif result == "LOST":
                pnl -= float(orig.get("loss_if_lose", 0))
                settled_count += 1
        for orig, resolved in zip(parlays, parlays_out):
            result = resolved["result"]
            if result == "WON":
                pnl += float(orig.get("profit_if_win", 0))
                settled_count += 1
            elif result == "LOST":
                pnl -= float(orig.get("total_cost", 0))
                settled_count += 1
        if settled_count > 0:
            new_bankroll = max(10.0, bankroll + pnl)
            save_bankroll(new_bankroll,
                          note=f"After {picks_date}: P&L {pnl:+.2f} ({settled_count} settled)")
            logger.info(f"Bankroll updated: ${bankroll:.2f} → ${new_bankroll:.2f} "
                        f"(P&L {pnl:+.2f}, {settled_count} settled)")
    except Exception as _br_err:
        logger.warning(f"Bankroll update failed (non-fatal): {_br_err}")

    watchlist_out = [
        resolve_pick(p) for p in display_pools
        if (p.get("game"), p.get("bet_type"), p.get("pick")) not in budget_keys
    ]

    snapshot = {
        "date":         picks_date.isoformat(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "singles":      singles_out,
        "parlays":      parlays_out,
        "watchlist":    watchlist_out,
    }

    SNAPSHOT_PATH.parent.mkdir(exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(snapshot, f, indent=2)

    n_settled = sum(1 for r in singles_out + watchlist_out
                    if r["result"] in ("WON", "LOST", "PUSH", "VOID"))
    n_total = len(singles_out) + len(watchlist_out)
    logger.info(f"Snapshot written: {n_settled}/{n_total} picks settled, "
                f"{len(parlays_out)} parlay(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
