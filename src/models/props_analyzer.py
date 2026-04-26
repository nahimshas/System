"""
Statistical prop analysis — overs only (Robinhood only offers over props).
All props are player-specific. Generates model lines from ESPN team stats +
player leaders. User must verify Robinhood's actual line before betting.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List
from src.models.edge_finder import _is_nba_playoff, _is_mlb_playoff
from src.config import NBA_PLAYOFF_RECENT_WEIGHT, MLB_PLAYOFF_STARTER_IP

logger = logging.getLogger(__name__)


@dataclass
class PropPick:
    sport: str
    player: str       # Real player name (e.g. "Anthony Edwards") or "Team leader" fallback
    team: str
    opponent: str
    prop_type: str
    model_line: float
    confidence: str
    note: str
    model_margin: float = 0.0  # how far model line is above the show threshold — used for sorting
    signals: List[str] = field(default_factory=list)
    research: List[str] = field(default_factory=list)
    commence_time: str = ""   # raw UTC ISO — for game-started detection


# ---------------------------------------------------------------------------
# NBA Props — points / rebounds / assists overs (player-specific)
# ---------------------------------------------------------------------------

def nba_player_props(games: List[Dict], nba_ctx: Dict) -> List[PropPick]:
    from src.data.nba_stats import normalize as nba_normalize

    season_stats  = nba_ctx.get("season_stats", {})
    recent_form   = nba_ctx.get("recent_form", {})
    rest_days     = nba_ctx.get("rest_days", {})
    team_leaders  = nba_ctx.get("team_leaders", {})  # {espn_name: {cat: {name, value}}}
    playoff       = _is_nba_playoff()

    if not season_stats:
        logger.warning("NBA season stats unavailable — skipping NBA props")
        return []

    picks: List[PropPick] = []

    for game in games:
        # Normalize to ESPN names (same key scheme as season_stats)
        home = nba_normalize(game["home_team"])
        away = nba_normalize(game["away_team"])
        game_commence = game.get("commence_time", "")

        for team, opp in [(home, away), (away, home)]:
            team_stats = season_stats.get(team, {})
            opp_stats  = season_stats.get(opp, {})
            if not team_stats or not opp_stats:
                continue

            off_rtg     = team_stats.get("off_rtg", 110.0)
            opp_def_rtg = opp_stats.get("def_rtg", 110.0)
            team_net    = team_stats.get("net_rtg", 0.0)
            opp_net     = opp_stats.get("net_rtg", 0.0)
            team_wins   = team_stats.get("wins", 0)
            team_losses = team_stats.get("losses", 1)
            opp_wins    = opp_stats.get("wins", 0)
            opp_losses  = opp_stats.get("losses", 1)

            team_recent  = recent_form.get(team, {})
            opp_recent   = recent_form.get(opp, {})
            recent_off   = team_recent.get("recent_off_rtg", off_rtg)
            recent_def_opp = opp_recent.get("recent_def_rtg", opp_def_rtg)

            # Blended expected team scoring (60% season / 40% recent)
            blended_off     = off_rtg     * 0.6 + recent_off      * 0.4
            blended_opp_def = opp_def_rtg * 0.6 + recent_def_opp  * 0.4
            expected_team_pts = (blended_off + blended_opp_def) / 2
            league_avg = 112.0

            team_rest = rest_days.get(team, 1)
            opp_rest  = rest_days.get(opp, 1)

            # Shared research block for all props in this matchup
            base_research = [
                f"{team}: {team_wins}W-{team_losses}L | PPG {off_rtg:.1f} | OPPG {opp_def_rtg:.1f} (opp) | Net {team_net:+.1f}",
                f"{opp}: {opp_wins}W-{opp_losses}L | Net {opp_net:+.1f}",
            ]
            if team_recent:
                base_research.append(
                    f"{team} recent (14d): PPG {recent_off:.1f} | OPPG {team_recent.get('recent_def_rtg', '?'):.1f} | "
                    f"Win% {team_recent.get('recent_w_pct', 0)*100:.0f}%"
                )
            base_research.append(
                f"Model projected team pts: {expected_team_pts:.1f} (league avg {league_avg})"
            )
            base_research.append(f"Rest days — {team}: {team_rest}d | {opp}: {opp_rest}d")

            leaders = team_leaders.get(team, {})

            # ── Points over ──────────────────────────────────────────────────
            pts_leader = leaders.get("points", {})
            pts_name   = pts_leader.get("name", f"{team} leading scorer")
            pts_season = pts_leader.get("value", 0.0)   # season PPG

            # Adjust season PPG for today's matchup vs opponent defense
            def_adj = (opp_def_rtg - league_avg) / league_avg * pts_season * 0.08
            model_pts = round(pts_season + def_adj, 1) if pts_season > 0 else round(expected_team_pts * 0.30, 1)
            # Playoff: stars (≥23 PPG) get higher usage (+5%), role players stay flat
            if playoff and model_pts >= 23:
                model_pts = round(model_pts * 1.05, 1)

            if model_pts >= 18:
                pts_conf = "HIGH" if (
                    model_pts >= 23
                    and opp_def_rtg > 111
                    and expected_team_pts > league_avg + 4
                    and team_rest > 0
                ) else "MEDIUM"
                pts_margin = round(model_pts - 18.0, 1)
                pts_signals = [
                    f"{pts_name} season avg: {pts_season:.1f} PPG"
                    + (f" (adjusted to {model_pts} vs {opp}'s defense)" if abs(def_adj) > 0.5 else ""),
                    f"{opp} allows {opp_def_rtg:.1f} PPG — "
                    + ("above avg (weak defense)" if opp_def_rtg > 111 else "average defense"),
                ]
                if playoff:
                    if model_pts >= 23:
                        pts_signals.append("🏆 Playoffs: star usage boost applied (+5%)")
                    else:
                        pts_signals.append("🏆 Playoffs: tighter defense — verify line carefully")
                if team_rest == 0:
                    pts_signals.append(f"⚠ {team} on B2B — consider reducing line or fading")
                picks.append(PropPick(
                    sport="NBA",
                    player=pts_name,
                    team=team,
                    opponent=opp,
                    prop_type="Points Over",
                    model_line=model_pts,
                    confidence=pts_conf,
                    model_margin=pts_margin,
                    note=(
                        f"Search '{pts_name} points' on Robinhood. "
                        f"Model line: {model_pts} pts. Look for Robinhood Over at or below {int(model_pts)}."
                    ),
                    signals=pts_signals,
                    research=base_research[:],
                    commence_time=game_commence,
                ))

            # ── Rebounds over ────────────────────────────────────────────────
            reb_leader = leaders.get("rebounds", {})
            reb_name   = reb_leader.get("name", "")
            reb_season = reb_leader.get("value", 0.0)

            if reb_name and reb_season >= 7.0:
                reb_conf = "HIGH" if reb_season >= 10.0 else "MEDIUM"
                reb_margin = round(reb_season - 7.0, 1)
                picks.append(PropPick(
                    sport="NBA",
                    player=reb_name,
                    team=team,
                    opponent=opp,
                    prop_type="Rebounds Over",
                    model_line=round(reb_season, 1),
                    confidence=reb_conf,
                    model_margin=reb_margin,
                    note=(
                        f"Search '{reb_name} rebounds' on Robinhood. "
                        f"Model line: {reb_season:.1f} RPG season avg. "
                        f"Look for Over at or below {int(reb_season)}."
                    ),
                    signals=[
                        f"{reb_name} averages {reb_season:.1f} rebounds/game this season",
                        f"Matchup pace supports rebounding volume",
                    ],
                    research=base_research[:],
                    commence_time=game_commence,
                ))

            # ── Assists over ─────────────────────────────────────────────────
            ast_leader = leaders.get("assists", {})
            ast_name   = ast_leader.get("name", "")
            ast_season = ast_leader.get("value", 0.0)

            if ast_name and ast_season >= 7.0:   # tightened from 6.0 → 7.0
                ast_conf   = "HIGH" if ast_season >= 9.0 else "MEDIUM"
                ast_margin = round(ast_season - 7.0, 1)
                picks.append(PropPick(
                    sport="NBA",
                    player=ast_name,
                    team=team,
                    opponent=opp,
                    prop_type="Assists Over",
                    model_line=round(ast_season, 1),
                    confidence=ast_conf,
                    model_margin=ast_margin,
                    note=(
                        f"Search '{ast_name} assists' on Robinhood. "
                        f"Model line: {ast_season:.1f} APG season avg. "
                        f"Look for Over at or below {int(ast_season)}."
                    ),
                    signals=[
                        f"{ast_name} averages {ast_season:.1f} assists/game this season",
                        f"High-scoring expected game supports assist volume",
                    ],
                    research=base_research[:],
                    commence_time=game_commence,
                ))

    # Deduplicate (same player could appear from home+away loop), sort, then cap
    seen: set = set()
    deduped: List[PropPick] = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    deduped.sort(key=lambda p: (0 if p.confidence == "HIGH" else 1, -p.model_margin))
    return deduped[:6]


# ---------------------------------------------------------------------------
# MLB Props — pitcher strikeouts + batter hits (player-specific, overs only)
# ---------------------------------------------------------------------------

def mlb_player_props(games: List[Dict], pitcher_stats_map: Dict) -> List[PropPick]:
    playoff = _is_mlb_playoff()
    picks: List[PropPick] = []

    for game in games:
        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        venue = game.get("venue", "")
        game_commence = game.get("commence_time", "")

        from src.data.mlb_stats import get_park_factor
        pf = get_park_factor(venue)

        for side, opp_side in [("home", "away"), ("away", "home")]:
            pitcher_name = game.get(f"{side}_pitcher_name", "TBD")
            team         = game.get(f"{side}_team", "")
            opp_team     = game.get(f"{opp_side}_team", "")

            if pitcher_name == "TBD" or not game.get(f"{side}_pitcher_id"):
                continue

            stats = pitcher_stats_map.get(pitcher_name, {})
            if not stats:
                continue

            k9  = stats.get("k_per_9", 7.0)
            era = stats.get("era", 4.50)
            fip = stats.get("fip", 4.20)
            bb9 = stats.get("bb_per_9", 3.0)
            hr9 = stats.get("hr_per_9", 1.2)
            ip  = stats.get("innings_pitched", 0)
            expected_innings = MLB_PLAYOFF_STARTER_IP if playoff else 5.5

            if ip < 20:
                continue

            pitcher_research = [
                f"{pitcher_name}: ERA {era:.2f} | FIP {fip:.2f} | K/9 {k9:.1f} | BB/9 {bb9:.1f} | HR/9 {hr9:.2f}",
                f"Season IP: {ip:.0f} | Projected start: ~{expected_innings} inn",
                f"Venue: {venue} (park factor {pf:.2f})" if venue else "Venue unknown",
            ]

            # ── Pitcher strikeouts over ──────────────────────────────────────
            expected_ks = round(k9 / 9 * expected_innings, 1)
            k_conf      = "HIGH" if (k9 > 9.5 and ip > 40 and fip < 4.0) else "MEDIUM"
            k_margin    = round(k9 - 7.0, 1)   # K/9 above league-average strikeout rate
            picks.append(PropPick(
                sport="MLB",
                player=pitcher_name,
                team=team,
                opponent=opp_team,
                prop_type="Strikeouts Over",
                model_line=expected_ks,
                confidence=k_conf,
                model_margin=k_margin,
                note=(
                    f"Search '{pitcher_name} strikeouts' on Robinhood. "
                    f"Model: ~{expected_ks} Ks in {expected_innings} inn. "
                    f"Look for Over at or below {int(expected_ks)}."
                ),
                signals=[
                    f"K/9: {k9:.1f} → projects {expected_ks} Ks over {expected_innings} inn",
                    f"FIP {fip:.2f} ({'above' if fip > 4.20 else 'below'} league avg 4.20)",
                    f"Season IP: {ip:.0f} — {'solid' if ip > 50 else 'limited'} sample",
                ],
                research=pitcher_research[:],
                commence_time=game_commence,
            ))

            # ── Batter 1+ hits over (vs hittable pitcher, FIP > 4.50) ───────
            if fip > 4.50:
                expected_hits = round((era / 9) * expected_innings * 1.1, 1)
                hits_conf     = "HIGH" if fip > 5.20 else "MEDIUM"
                hits_margin   = round(fip - 4.50, 2)   # how hittable above threshold
                picks.append(PropPick(
                    sport="MLB",
                    player=f"{opp_team} top-order batters",
                    team=opp_team,
                    opponent=team,
                    prop_type="Hits Over (1+)",
                    model_line=1.0,
                    confidence=hits_conf,
                    model_margin=hits_margin,
                    note=(
                        f"{pitcher_name} (FIP {fip:.2f}) is hittable. "
                        f"On Robinhood, search 1+ hits over props for {opp_team}'s "
                        f"#1–#3 batters in the lineup. "
                        f"Model projects ~{expected_hits} total hits in {expected_innings} inn."
                    ),
                    signals=[
                        f"Pitcher FIP {fip:.2f} — above avg (4.20), favorable for hitters",
                        f"Model projects ~{expected_hits} hits in {expected_innings} inn",
                        f"Target: {opp_team} leadoff, #2, and #3 batters for 1+ hit props",
                    ],
                    research=pitcher_research + [
                        f"FIP {fip:.2f} > 4.50 — hitter-favorable matchup",
                        f"ERA-based hit estimate: ~{expected_hits} hits in {expected_innings} inn",
                    ],
                    commence_time=game_commence,
                ))

    # Deduplicate, sort, then cap
    seen: set = set()
    deduped: List[PropPick] = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    deduped.sort(key=lambda p: (0 if p.confidence == "HIGH" else 1, -p.model_margin))
    return deduped[:6]
