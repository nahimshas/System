"""
Statistical prop analysis — overs only (Robinhood only offers over props).
Generates model lines for player props based on stats vs defensive matchups.
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


# ---------------------------------------------------------------------------
# NBA Props (Overs only)
# ---------------------------------------------------------------------------

def nba_player_props(games: List[Dict], nba_ctx: Dict) -> List[PropPick]:
    picks: List[PropPick] = []
    season_stats = nba_ctx.get("season_stats", {})

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

            expected_team_pts = (off_rtg + opp_def_rtg) / 2 * pace / 100
            avg_pts = 112.0

            signals = [
                f"Team OffRtg: {off_rtg:.1f} vs Opp DefRtg: {opp_def_rtg:.1f}",
                f"Pace: {pace:.1f} possessions/game",
                f"Model expected team pts: {expected_team_pts:.1f} (league avg: {avg_pts})",
            ]

            if expected_team_pts > avg_pts + 5:
                confidence = "HIGH" if expected_team_pts > avg_pts + 8 else "MEDIUM"
                picks.append(PropPick(
                    sport="NBA",
                    player=f"{team} (team total)",
                    team=team,
                    opponent=opp,
                    prop_type="Team Points Over",
                    model_line=round(expected_team_pts, 1),
                    confidence=confidence,
                    note=f"Check Robinhood for {team} team total prop. Model: {expected_team_pts:.1f} pts — look for Over lines below {expected_team_pts:.0f}.",
                    signals=signals,
                ))

            # Star player points estimate (~30% of team scoring)
            star_pts = round(expected_team_pts * 0.30, 1)
            if star_pts >= 22:
                picks.append(PropPick(
                    sport="NBA",
                    player=f"{team} leading scorer",
                    team=team,
                    opponent=opp,
                    prop_type="Points Over",
                    model_line=star_pts,
                    confidence="MEDIUM",
                    note=f"Model projects {team}'s top scorer ~{star_pts} pts. Check Robinhood for leading scorer points prop — look for lines below {star_pts:.0f}.",
                    signals=signals,
                ))

    return picks[:4]


# ---------------------------------------------------------------------------
# MLB Props (Overs only — strikeouts, hits, HRs, total bases)
# ---------------------------------------------------------------------------

def mlb_player_props(games: List[Dict], pitcher_stats_map: Dict) -> List[PropPick]:
    picks: List[PropPick] = []

    for game in games:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

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
            ip = stats.get("innings_pitched", 0)
            expected_innings = 5.5

            if ip < 20:
                continue

            # --- Strikeouts Over ---
            expected_ks = round(k9 / 9 * expected_innings, 1)
            k_confidence = "HIGH" if k9 > 9.5 and ip > 40 else "MEDIUM"
            picks.append(PropPick(
                sport="MLB",
                player=pitcher_name,
                team=team,
                opponent=opp_team,
                prop_type="Strikeouts Over",
                model_line=expected_ks,
                confidence=k_confidence,
                note=f"Check Robinhood for {pitcher_name} strikeouts prop. Model: {expected_ks} Ks (~{expected_innings} inn). Look for Over lines at or below {int(expected_ks)}.",
                signals=[
                    f"K/9: {k9} | FIP: {fip} | ERA: {era} | Season IP: {ip:.0f}",
                    f"Projected {expected_ks} Ks over ~{expected_innings} innings",
                ],
            ))

            # --- Hits Over (opposing batters vs weak pitcher) ---
            if fip > 4.50:
                expected_hits = round((era / 9) * expected_innings * 1.1, 1)
                picks.append(PropPick(
                    sport="MLB",
                    player=f"{opp_team} batters vs {pitcher_name}",
                    team=opp_team,
                    opponent=team,
                    prop_type="Hits Over",
                    model_line=expected_hits,
                    confidence="MEDIUM",
                    note=f"{pitcher_name} (FIP {fip}) is hittable. Check Robinhood for 1+ hit overs on {opp_team}'s top of lineup batters.",
                    signals=[
                        f"Opposing pitcher FIP: {fip} (above avg 4.20) | ERA: {era}",
                        f"Model projects ~{expected_hits} hits allowed in {expected_innings} innings",
                    ],
                ))

        # --- HR prop for hitter-friendly parks ---
        venue = game.get("venue", "")
        from src.data.mlb_stats import get_park_factor
        pf = get_park_factor(venue)
        if pf >= 1.08:
            picks.append(PropPick(
                sport="MLB",
                player=f"Power hitters — {home} vs {away}",
                team=f"{home}/{away}",
                opponent="",
                prop_type="Home Run Over",
                model_line=0.5,
                confidence="MEDIUM",
                note=f"{venue} is HR-friendly (park factor {pf:.2f}). Check Robinhood for 1+ HR props on power hitters from both lineups.",
                signals=[
                    f"Park factor: {pf:.2f} — well above average (1.00)",
                    "Elevated HR environment raises 1+ HR probability for power hitters",
                ],
            ))

    # Deduplicate and cap
    seen = set()
    deduped = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped[:6]
