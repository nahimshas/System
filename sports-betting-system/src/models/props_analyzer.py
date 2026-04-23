"""
Statistical prop analysis (no market price comparison — user verifies Robinhood price).
Generates model lines for player props based on rolling averages vs defensive matchups.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PropPick:
    sport: str
    player: str
    team: str
    prop_type: str          # e.g. "Points", "Strikeouts", "Home Runs"
    model_line: float       # our model's expected value
    direction: str          # "Over" or "Under" (vs Robinhood's line)
    confidence: str         # "HIGH" / "MEDIUM"
    note: str               # instruction for user to check Robinhood
    signals: List[str] = field(default_factory=list)


def nba_player_props(games: List[Dict], nba_ctx: Dict) -> List[PropPick]:
    """
    Generates NBA player prop model lines.
    Uses team-level offensive stats as proxy when individual player data is unavailable.
    For full player stat integration, connect a paid stats API.
    """
    picks: List[PropPick] = []
    season_stats = nba_ctx.get("season_stats", {})
    recent_form = nba_ctx.get("recent_form", {})

    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        for team, opp in [(home, away), (away, home)]:
            team_stats = season_stats.get(team, {})
            opp_stats = season_stats.get(opp, {})

            off_rtg = team_stats.get("off_rtg", 110.0)
            opp_def_rtg = opp_stats.get("def_rtg", 110.0)
            pace = team_stats.get("pace", 100.0)

            # Estimate team's expected points in this game
            expected_team_pts = (off_rtg + opp_def_rtg) / 2 * pace / 100

            # Team points flag — if expected_team_pts differs significantly from average
            avg_pts = 112.0  # NBA league average
            if abs(expected_team_pts - avg_pts) > 5:
                direction = "Over" if expected_team_pts > avg_pts else "Under"
                signals = [
                    f"Team OffRtg: {off_rtg:.1f} vs Opp DefRtg: {opp_def_rtg:.1f}",
                    f"Expected team pts: {expected_team_pts:.1f} (league avg: {avg_pts})",
                ]
                picks.append(PropPick(
                    sport="NBA",
                    player=f"{team} (team scoring)",
                    team=team,
                    prop_type="Team Points",
                    model_line=round(expected_team_pts, 1),
                    direction=direction,
                    confidence="MEDIUM",
                    note=f"Check Robinhood for {team} team total prop. Model projects {expected_team_pts:.1f} pts.",
                    signals=signals,
                ))

    return picks[:4]   # cap at top 4 prop notes


def mlb_player_props(games: List[Dict], pitcher_stats_map: Dict) -> List[PropPick]:
    """
    Generates MLB pitcher strikeout model lines.
    pitcher_stats_map: {pitcher_name: stats_dict}
    """
    picks: List[PropPick] = []

    for game in games:
        for side in ["home", "away"]:
            name_key = f"{side}_pitcher_name"
            id_key = f"{side}_pitcher_id"
            pitcher_name = game.get(name_key, "TBD")
            if pitcher_name == "TBD" or not game.get(id_key):
                continue

            stats = pitcher_stats_map.get(pitcher_name, {})
            if not stats:
                continue

            k9 = stats.get("k_per_9", 7.0)
            fip = stats.get("fip", 4.20)
            ip = stats.get("innings_pitched", 0)

            # Expected K's in ~5.5 innings (typical starter today)
            expected_ks = round(k9 / 9 * 5.5, 1)
            signals = [
                f"K/9: {k9} | FIP: {fip} | Season IP: {ip:.0f}",
                f"Projected {expected_ks} Ks over ~5.5 innings",
            ]

            confidence = "HIGH" if k9 > 9.5 and ip > 30 else "MEDIUM"

            picks.append(PropPick(
                sport="MLB",
                player=pitcher_name,
                team=game[f"{side}_team"],
                prop_type="Strikeouts",
                model_line=expected_ks,
                direction="Over" if k9 > 8.5 else "Under",
                confidence=confidence,
                note=f"Check Robinhood for {pitcher_name} strikeout prop. Model projects {expected_ks} Ks.",
                signals=signals,
            ))

        # HR / hits props using team-level OPS as indicator
        home_ops = game.get("home_ops", 0.720)
        away_ops = game.get("away_ops", 0.720)

    return picks
