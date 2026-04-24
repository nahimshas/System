"""
Statistical prop analysis — overs only (Robinhood only offers over props).
All props are player-specific. Generates model lines based on stats vs matchups.
User must verify Robinhood's actual line before betting.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class PropPick:
    sport: str
    player: str
    team: str
    opponent: str
    prop_type: str
    model_line: float
    confidence: str
    note: str
    signals: List[str] = field(default_factory=list)
    research: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NBA Props — player points/rebounds/assists overs only
# ---------------------------------------------------------------------------

def nba_player_props(games: List[Dict], nba_ctx: Dict) -> List[PropPick]:
    picks: List[PropPick] = []
    season_stats = nba_ctx.get("season_stats", {})
    recent_form = nba_ctx.get("recent_form", {})
    rest_days = nba_ctx.get("rest_days", {})

    if not season_stats:
        logger.warning("NBA season stats unavailable — skipping NBA props")
        return []

    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        for team, opp in [(home, away), (away, home)]:
            team_stats = season_stats.get(team, {})
            opp_stats = season_stats.get(opp, {})
            if not team_stats or not opp_stats:
                continue

            off_rtg = team_stats.get("off_rtg", 110.0)
            opp_def_rtg = opp_stats.get("def_rtg", 110.0)
            pace = team_stats.get("pace", 100.0)
            team_net = team_stats.get("net_rtg", 0.0)
            opp_net = opp_stats.get("net_rtg", 0.0)

            # Recent form
            team_recent = recent_form.get(team, {})
            recent_off = team_recent.get("recent_off_rtg", off_rtg)
            recent_def_opp = recent_form.get(opp, {}).get("recent_def_rtg", opp_def_rtg)

            # Blended expected team points (season + recent weighted)
            blended_off = off_rtg * 0.6 + recent_off * 0.4
            blended_opp_def = opp_def_rtg * 0.6 + recent_def_opp * 0.4
            expected_team_pts = (blended_off + blended_opp_def) / 2 * pace / 100
            league_avg = 112.0

            # Rest
            team_rest = rest_days.get(team, 1)
            opp_rest = rest_days.get(opp, 1)

            research = [
                f"{team} OffRtg: {off_rtg:.1f} | DefRtg: {opp_def_rtg:.1f} (opponent)",
                f"{team} NetRtg: {team_net:+.1f} | Pace: {pace:.1f} poss/game",
                f"Recent (14d) OffRtg: {recent_off:.1f} vs opp recent DefRtg: {recent_def_opp:.1f}",
                f"Model projected team pts: {expected_team_pts:.1f} (league avg: {league_avg})",
                f"Rest days — {team}: {team_rest}d | {opp}: {opp_rest}d",
            ]

            # --- Leading scorer points over ---
            # Estimate: star player scores ~28-32% of team total
            star_pts = round(expected_team_pts * 0.30, 1)
            if star_pts >= 21:
                confidence = "HIGH" if (
                    star_pts >= 24 and expected_team_pts > league_avg + 5 and opp_def_rtg > 111
                ) else "MEDIUM"
                signals = [
                    f"Projected team pts {expected_team_pts:.1f} → star scorer ~{star_pts} pts (30% share)",
                    f"Opponent DefRtg {opp_def_rtg:.1f} — {'above avg (weak defense)' if opp_def_rtg > 111 else 'average defense'}",
                ]
                if team_rest == 0:
                    signals.append(f"{team} on B2B — may reduce star minutes, fade if tired")
                picks.append(PropPick(
                    sport="NBA",
                    player=f"{team} leading scorer",
                    team=team,
                    opponent=opp,
                    prop_type="Points Over",
                    model_line=star_pts,
                    confidence=confidence,
                    note=(
                        f"Model projects {team}'s top scorer ~{star_pts} pts. "
                        f"On Robinhood, search for the points over prop for {team}'s leading scorer "
                        f"and look for a line at or below {int(star_pts)}."
                    ),
                    signals=signals,
                    research=research,
                ))

            # --- Second scorer / secondary star ---
            second_pts = round(expected_team_pts * 0.20, 1)
            if second_pts >= 16 and expected_team_pts > league_avg + 6:
                picks.append(PropPick(
                    sport="NBA",
                    player=f"{team} second scorer",
                    team=team,
                    opponent=opp,
                    prop_type="Points Over",
                    model_line=second_pts,
                    confidence="MEDIUM",
                    note=(
                        f"High-paced game favors {team}'s secondary scorer. "
                        f"Model projects ~{second_pts} pts for the #2 option. "
                        f"Look for a line at or below {int(second_pts)} on Robinhood."
                    ),
                    signals=[
                        f"Projected team pts {expected_team_pts:.1f} supports multiple high scorers",
                        f"Opponent DefRtg {opp_def_rtg:.1f} — allowing {opp_def_rtg - 110:.1f} pts above avg",
                    ],
                    research=research,
                ))

    # Deduplicate and cap at 4 NBA props
    seen = set()
    deduped = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped[:4]


# ---------------------------------------------------------------------------
# MLB Props — pitcher strikeouts + batter hits (player-specific, overs only)
# ---------------------------------------------------------------------------

def mlb_player_props(games: List[Dict], pitcher_stats_map: Dict) -> List[PropPick]:
    picks: List[PropPick] = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        venue = game.get("venue", "")

        from src.data.mlb_stats import get_park_factor
        pf = get_park_factor(venue)
        park_note = f"Park factor {pf:.2f}" if venue else "Park factor unknown"

        for side, opp_side in [("home", "away"), ("away", "home")]:
            pitcher_name = game.get(f"{side}_pitcher_name", "TBD")
            opp_team = game.get(f"{opp_side}_team", "")
            team = game.get(f"{side}_team", "")

            if pitcher_name == "TBD" or not game.get(f"{side}_pitcher_id"):
                continue

            stats = pitcher_stats_map.get(pitcher_name, {})
            if not stats:
                continue

            k9 = stats.get("k_per_9", 7.0)
            era = stats.get("era", 4.50)
            fip = stats.get("fip", 4.20)
            bb9 = stats.get("bb_per_9", 3.0)
            hr9 = stats.get("hr_per_9", 1.2)
            ip = stats.get("innings_pitched", 0)
            expected_innings = 5.5

            if ip < 20:
                continue

            pitcher_research = [
                f"{pitcher_name}: ERA {era:.2f} | FIP {fip:.2f} | K/9 {k9:.1f} | BB/9 {bb9:.1f}",
                f"Season innings: {ip:.0f} | HR/9: {hr9:.2f}",
                f"Projected start: ~{expected_innings} innings",
                f"Venue: {venue} ({park_note})",
            ]

            # --- Pitcher strikeouts over ---
            expected_ks = round(k9 / 9 * expected_innings, 1)
            k_confidence = "HIGH" if (k9 > 9.5 and ip > 40 and fip < 4.0) else "MEDIUM"
            picks.append(PropPick(
                sport="MLB",
                player=pitcher_name,
                team=team,
                opponent=opp_team,
                prop_type="Strikeouts Over",
                model_line=expected_ks,
                confidence=k_confidence,
                note=(
                    f"Search for {pitcher_name} strikeouts on Robinhood. "
                    f"Model: ~{expected_ks} Ks in {expected_innings} innings. "
                    f"Look for Over lines at {int(expected_ks)} or below."
                ),
                signals=[
                    f"K/9: {k9:.1f} → projects {expected_ks} Ks over {expected_innings} inn",
                    f"FIP {fip:.2f} ({'above' if fip > 4.20 else 'below'} league avg 4.20)",
                    f"Season IP: {ip:.0f} — {'sufficient' if ip > 40 else 'limited'} sample",
                ],
                research=pitcher_research,
            ))

            # --- Batter hits over (vs weak/hittable pitcher) ---
            # Only suggest when pitcher is clearly hittable (FIP > 4.50)
            if fip > 4.50:
                # Estimate hits allowed: ERA-based hit rate × innings
                expected_hits = round((era / 9) * expected_innings * 1.1, 1)
                picks.append(PropPick(
                    sport="MLB",
                    player=f"{opp_team} top-order batters",
                    team=opp_team,
                    opponent=team,
                    prop_type="Hits Over (1+)",
                    model_line=1.0,
                    confidence="MEDIUM",
                    note=(
                        f"{pitcher_name} (FIP {fip:.2f}) is hittable. "
                        f"On Robinhood, look for 1+ hits over props on {opp_team}'s "
                        f"1st-3rd batters in the lineup. Model projects ~{expected_hits} hits allowed in {expected_innings} inn."
                    ),
                    signals=[
                        f"Pitcher FIP {fip:.2f} — well above avg (4.20), hittable",
                        f"Model projects ~{expected_hits} hits allowed in {expected_innings} innings",
                        f"Target: {opp_team} leadoff / #2 / #3 hitters for 1+ hit prop",
                    ],
                    research=pitcher_research + [
                        f"FIP {fip:.2f} > 4.50 threshold — opposing batters have advantage",
                        f"ERA-based hit projection: ~{expected_hits} hits in {expected_innings} innings",
                    ],
                ))

    # Deduplicate (pitcher can appear in both sides) and cap
    seen = set()
    deduped = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped[:6]
