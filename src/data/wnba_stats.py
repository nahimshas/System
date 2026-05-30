"""
WNBA statistics fetcher.
All data from ESPN public APIs (no key required, CORS-open).
"""
import logging
import re
import requests
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"

# WNBA team name normalisation (Odds API names → ESPN display names)
_WNBA_ALIASES: Dict[str, str] = {
    "atlanta dream": "Atlanta Dream",
    "chicago sky": "Chicago Sky",
    "connecticut sun": "Connecticut Sun",
    "dallas wings": "Dallas Wings",
    "golden state valkyries": "Golden State Valkyries",
    "indiana fever": "Indiana Fever",
    "las vegas aces": "Las Vegas Aces",
    "los angeles sparks": "Los Angeles Sparks",
    "minnesota lynx": "Minnesota Lynx",
    "new york liberty": "New York Liberty",
    "phoenix mercury": "Phoenix Mercury",
    "seattle storm": "Seattle Storm",
    "toronto tempo": "Toronto Tempo",
    "washington mystics": "Washington Mystics",
}

def normalize(name: str) -> str:
    """Lowercase + strip for flexible team matching."""
    return name.lower().strip()


def _name_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    return a == b or a in b or b in a


def _score_val(competitor: dict) -> float:
    """
    Extract a numeric score from an ESPN schedule competitor.

    The schedule endpoint returns score as a dict {"value": 112.0,
    "displayValue": "112"}, NOT a scalar. The original code did
    float(competitor["score"]) which threw on the dict and was silently
    swallowed — leaving recent-form (and the new season margin) empty. This
    handles both the dict and legacy scalar shapes.
    """
    s = competitor.get("score", 0)
    if isinstance(s, dict):
        s = s.get("value", 0)
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def get_wnba_context(today: date, team_names: List[str]) -> Dict:
    """
    Returns {season_stats, recent_form, rest_days} for all requested teams.
    season_stats[team] = {ppg, opp_ppg, fg_pct, ast_to, net_rtg}
    recent_form[team]  = {recent_ppg, recent_opp_ppg, recent_net_rtg, recent_w_pct, games}
    rest_days[team]    = int (days since last game, capped at 4)
    """
    ctx: Dict = {"season_stats": {}, "recent_form": {}, "rest_days": {}}

    # Fetch all teams to build name→id map
    try:
        r = requests.get(f"{ESPN_BASE}/teams", timeout=12)
        r.raise_for_status()
        data = r.json()
        teams_list = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
    except Exception as e:
        logger.error(f"WNBA teams fetch failed: {e}")
        return ctx

    # Build name→id map
    team_id_map: Dict[str, str] = {}
    for entry in teams_list:
        tm = entry.get("team", {})
        tid = tm.get("id", "")
        display = tm.get("displayName", "")
        if tid and display:
            team_id_map[normalize(display)] = tid
            if tm.get("abbreviation"):
                team_id_map[normalize(tm["abbreviation"])] = tid

    # Resolve requested team names to IDs
    resolved: Dict[str, str] = {}  # canonical_name → team_id
    for name in team_names:
        n = normalize(name)
        tid = team_id_map.get(n)
        if not tid:
            for k, v in team_id_map.items():
                if n in k or k in n:
                    tid = v
                    break
        if tid:
            # Find canonical display name
            for entry in teams_list:
                tm = entry.get("team", {})
                if tm.get("id") == tid:
                    resolved[tm.get("displayName", name)] = tid
                    break

    # Fetch stats + schedule for each resolved team
    for display_name, tid in resolved.items():
        _fetch_team_stats(display_name, tid, today, ctx)

    return ctx


def _fetch_team_stats(display_name: str, tid: str, today: date, ctx: Dict) -> None:
    """Fetch season stats and schedule for one team, populate ctx in-place."""
    # Season stats
    try:
        r = requests.get(f"{ESPN_BASE}/teams/{tid}/statistics", timeout=12)
        r.raise_for_status()
        data = r.json()
        cats = data.get("results", {}).get("stats", {}).get("categories", [])
        stats_map: Dict[str, float] = {}
        for cat in cats:
            for s in cat.get("stats", []):
                v = s.get("value")
                if v is not None:
                    try:
                        stats_map[s["name"]] = float(v)
                    except (TypeError, ValueError):
                        pass

        ppg     = stats_map.get("avgPoints", 0.0)
        # opp_ppg not directly in team stats — approximate from pts allowed proxy
        fg_pct  = stats_map.get("fieldGoalPct", 0.0)
        ast_to  = stats_map.get("assistTurnoverRatio", 1.0)
        reb     = stats_map.get("avgRebounds", 0.0)
        blk     = stats_map.get("avgBlocks", 0.0)
        stl     = stats_map.get("avgSteals", 0.0)
        # wins/losses are NOT in the stats categories — parse from recordSummary
        # e.g. team.recordSummary = "2-0" → wins=2, losses=0
        record_str = data.get("team", {}).get("recordSummary", "")
        try:
            _w, _l = record_str.split("-")
            wins   = int(_w)
            losses = int(_l)
        except Exception:
            wins   = int(stats_map.get("wins",   0))
            losses = int(stats_map.get("losses", 0))
        # Fallback net_rtg proxy, used ONLY if the schedule fetch below fails.
        # The schedule block overrides this with a real points-for-minus-against
        # season margin. This offense-only proxy ignores defense and is a poor
        # strength estimate — kept solely as a last resort.
        net_rtg_proxy = ppg + (blk + stl) * 1.5 - 82.0  # centered around ~82 PPG league avg

        ctx["season_stats"][display_name] = {
            "ppg":     round(ppg, 1),
            "fg_pct":  round(fg_pct, 1),
            "ast_to":  round(ast_to, 2),
            "blk":     round(blk, 1),
            "stl":     round(stl, 1),
            "net_rtg": round(net_rtg_proxy, 2),
            "wins":    wins,
            "losses":  losses,
        }
    except Exception as e:
        logger.warning(f"WNBA stats fetch failed ({display_name}): {e}")

    # Schedule for rest days + recent form
    try:
        r2 = requests.get(
            f"{ESPN_BASE}/teams/{tid}/schedule",
            params={"season": today.year},
            timeout=12,
        )
        r2.raise_for_status()
        sched = r2.json()
        events = sched.get("events", [])

        today_dt = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
        past_games = []
        for ev in events:
            try:
                ev_dt = datetime.fromisoformat(ev.get("date", "").replace("Z", "+00:00"))
                if ev_dt < today_dt:
                    # Only count completed games
                    comp = (ev.get("competitions") or [{}])[0]
                    status = comp.get("status", {}).get("type", {})
                    if status.get("completed") or status.get("state") == "post":
                        past_games.append({"date": ev_dt, "event": ev})
            except Exception:
                pass

        past_games.sort(key=lambda x: x["date"])

        # Rest days
        if past_games:
            last_game_dt = past_games[-1]["date"]
            rest = (today_dt - last_game_dt).days
            ctx["rest_days"][display_name] = min(rest, 5)
        else:
            ctx["rest_days"][display_name] = 3  # default

        # Recent form: last 8 completed games (enough for 40-game season context)
        recent = past_games[-8:]
        if recent:
            w, l, ppg_sum, opp_ppg_sum = 0, 0, 0.0, 0.0
            for g in recent:
                comp = (g["event"].get("competitions") or [{}])[0]
                competitors = comp.get("competitors", [])
                our_comp = next(
                    (c for c in competitors if c.get("team", {}).get("id") == tid), None
                )
                opp_comp = next(
                    (c for c in competitors if c.get("team", {}).get("id") != tid), None
                )
                if our_comp and opp_comp:
                    our_score = _score_val(our_comp)
                    opp_score = _score_val(opp_comp)
                    if our_score or opp_score:
                        ppg_sum     += our_score
                        opp_ppg_sum += opp_score
                        if our_comp.get("winner"):
                            w += 1
                        else:
                            l += 1

            total = w + l
            if total > 0:
                recent_net = (ppg_sum - opp_ppg_sum) / total
                ctx["recent_form"][display_name] = {
                    "recent_ppg":     round(ppg_sum / total, 1),
                    "recent_opp_ppg": round(opp_ppg_sum / total, 1),
                    "recent_net_rtg": round(recent_net, 2),
                    "recent_w_pct":   round(w / total, 3),
                    "games":          total,
                }

        # ── Real season net rating (points for − against over ALL games) ──────
        # Replaces the old offense-only proxy (ppg + (blk+stl)×1.5 − 82), which
        # ignored opponent scoring and over-rated high-scoring/leaky teams. The
        # margin data is already here in past_games — we just aggregate the full
        # season, not only the recent window. Also stores season opp_ppg for the
        # projected-score fallback.
        s_pf = s_pa = s_n = 0.0
        for g in past_games:
            comp = (g["event"].get("competitions") or [{}])[0]
            competitors = comp.get("competitors", [])
            our_c = next((c for c in competitors if c.get("team", {}).get("id") == tid), None)
            opp_c = next((c for c in competitors if c.get("team", {}).get("id") != tid), None)
            if our_c and opp_c:
                ocs, ops = _score_val(our_c), _score_val(opp_c)
                if ocs or ops:
                    s_pf += ocs
                    s_pa += ops
                    s_n  += 1
        if s_n > 0 and display_name in ctx["season_stats"]:
            # Derive ppg, opp_ppg, and net_rtg all from the same schedule
            # aggregation so the card's "PPG − opp_ppg = NetRtg" reconciles.
            ctx["season_stats"][display_name]["ppg"]     = round(s_pf / s_n, 1)
            ctx["season_stats"][display_name]["opp_ppg"] = round(s_pa / s_n, 1)
            ctx["season_stats"][display_name]["net_rtg"] = round((s_pf - s_pa) / s_n, 2)

    except Exception as e:
        logger.warning(f"WNBA schedule fetch failed ({display_name}): {e}")


def get_wnba_injuries() -> Dict[str, List[Dict]]:
    """
    Returns {team_display_name: [{player, status, points_share, minutes_share}]}.
    points_share = player_ppg / team_ppg (dynamically fetched per injured player).
    """
    result: Dict[str, List[Dict]] = {}

    try:
        r = requests.get(f"{ESPN_BASE}/injuries", timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"WNBA injuries fetch failed: {e}")
        return result

    for team_entry in data.get("injuries", []):
        team_name = team_entry.get("displayName", "")
        team_injs = []

        # Get team PPG for points-share calculation
        team_ppg = _get_team_ppg(team_entry.get("id", ""))

        for inj in team_entry.get("injuries", []):
            ath = inj.get("athlete", {})
            player_name = ath.get("displayName", "")
            status = inj.get("status", "Out")

            # Extract athlete ID from ESPN player card link
            athlete_id = None
            for link in ath.get("links", []):
                href = link.get("href", "")
                m = re.search(r'/id/(\d+)/', href)
                if m:
                    athlete_id = m.group(1)
                    break

            # Fetch player PPG for points-share (with fast timeout — non-critical)
            player_ppg = 0.0
            player_mpg = 0.0
            if athlete_id:
                try:
                    pr = requests.get(
                        f"{ESPN_CORE}/athletes/{athlete_id}/statistics/0",
                        timeout=6,
                    )
                    if pr.ok:
                        pdata = pr.json()
                        pcats = pdata.get("splits", {}).get("categories", [])
                        for cat in pcats:
                            for s in cat.get("stats", []):
                                if s.get("name") == "avgPoints":
                                    try: player_ppg = float(s.get("value", 0))
                                    except: pass
                                if s.get("name") == "avgMinutes":
                                    try: player_mpg = float(s.get("value", 0))
                                    except: pass
                except Exception:
                    pass  # non-critical — falls back to 0 (minimal impact)

            # Points share: player PPG / team PPG
            points_share = round(player_ppg / team_ppg, 3) if team_ppg > 0 else 0.0
            # Minutes share (proxy for defensive/facilitating value): mpg / 40 minutes per game
            minutes_share = round(player_mpg / 40.0, 3)
            # Weight = max of points share and 60% of minutes share
            player_weight = max(points_share, minutes_share * 0.6)

            team_injs.append({
                "player":        player_name,
                "status":        status,
                "ppg":           round(player_ppg, 1),
                "mpg":           round(player_mpg, 1),
                "points_share":  points_share,
                "minutes_share": minutes_share,
                "player_weight": round(player_weight, 3),
            })

        if team_injs:
            result[team_name] = team_injs

    return result


def _get_team_ppg(team_id: str) -> float:
    """Fetch team PPG for points-share denominator."""
    if not team_id:
        return 82.0  # WNBA league average fallback
    try:
        r = requests.get(f"{ESPN_BASE}/teams/{team_id}/statistics", timeout=8)
        r.raise_for_status()
        data = r.json()
        cats = data.get("results", {}).get("stats", {}).get("categories", [])
        for cat in cats:
            for s in cat.get("stats", []):
                if s.get("name") == "avgPoints":
                    return float(s.get("value", 82.0))
    except Exception:
        pass
    return 82.0
