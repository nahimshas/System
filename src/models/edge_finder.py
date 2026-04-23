"""
Core probability model. Takes market odds + stats + injuries, returns edge analysis.

Approach:
  1. Use market implied probability (no-vig) as base
  2. Apply statistical adjustments for factors the market may have mispriced
  3. Compare adjusted probability to market — flag positive edges
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from scipy.stats import norm

from src.config import (
    NBA_HOME_ADVANTAGE, NBA_BACK_TO_BACK_PENALTY, NBA_REST_BONUS_PER_DAY,
    NBA_RECENT_FORM_WEIGHT, MLB_HOME_ADVANTAGE, MLB_BACK_TO_BACK_PENALTY,
    MLB_REST_BONUS_PER_DAY, MIN_EDGE, ROBINHOOD_COMMISSION
)
from src.data.injuries import injury_adjustment
from src.data.nba_stats import normalize as nba_normalize
from src.models.kelly import robinhood_kelly, has_positive_ev, BetSizing

logger = logging.getLogger(__name__)

NBA_SPREAD_STD = 12.0    # std dev of NBA score margins
MLB_SPREAD_STD = 1.8     # std dev of MLB run differentials


@dataclass
class BetRecommendation:
    sport: str
    game: str                        # "Away @ Home"
    bet_type: str                    # "Moneyline", "Total Over", "Total Under", "Spread"
    pick: str                        # e.g. "Los Angeles Lakers" or "Over 224.5"
    market_prob: float               # market's implied probability (no vig)
    model_prob: float                # our adjusted probability
    edge: float                      # model_prob - market_prob
    contract_price: float            # estimated Robinhood price (= market_prob)
    sizing: BetSizing
    confidence: str                  # "HIGH", "MEDIUM"
    signals: List[str] = field(default_factory=list)
    home_team: str = ""
    away_team: str = ""


def _confidence_label(edge: float) -> str:
    if edge >= 0.07:
        return "HIGH"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# NBA Edge Finder
# ---------------------------------------------------------------------------

def _nba_team_strength(team: str, ctx: Dict) -> float:
    """Returns adjusted net rating for a team."""
    season = ctx["season_stats"].get(team, {})
    recent = ctx["recent_form"].get(team, {})

    season_net = season.get("net_rtg", 0.0)
    recent_net = recent.get("recent_net_rtg", season_net)

    # Blend season and recent form
    return (1 - NBA_RECENT_FORM_WEIGHT) * season_net + NBA_RECENT_FORM_WEIGHT * recent_net


def _nba_margin_to_prob(expected_margin: float) -> float:
    """Converts expected point margin to home win probability."""
    return float(norm.cdf(expected_margin, 0, NBA_SPREAD_STD))


def analyze_nba_game(game: Dict, nba_ctx: Dict, nba_injuries: Dict) -> List[BetRecommendation]:
    home = game["home_team"]
    away = game["away_team"]
    home = nba_normalize(home)
    away = nba_normalize(away)
    label = f"{away} @ {home}"
    recs = []

    # --- Moneyline ---
    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_strength = _nba_team_strength(home, nba_ctx)
        away_strength = _nba_team_strength(away, nba_ctx)
        base_margin = home_strength - away_strength

        # Adjustments
        signals = []
        adj = 0.0

        adj += NBA_HOME_ADVANTAGE
        signals.append(f"Home court advantage (+{NBA_HOME_ADVANTAGE*100:.0f}%)")

        home_rest = nba_ctx["rest_days"].get(home, 1)
        away_rest = nba_ctx["rest_days"].get(away, 1)

        if home_rest == 0:
            adj -= NBA_BACK_TO_BACK_PENALTY
            signals.append(f"{home} on back-to-back (-{NBA_BACK_TO_BACK_PENALTY*100:.0f}%)")
        if away_rest == 0:
            adj += NBA_BACK_TO_BACK_PENALTY
            signals.append(f"{away} on back-to-back (favors {home} +{NBA_BACK_TO_BACK_PENALTY*100:.0f}%)")

        rest_diff = min(home_rest - away_rest, 3)
        if abs(rest_diff) >= 1:
            rest_adj = rest_diff * NBA_REST_BONUS_PER_DAY
            adj += rest_adj
            direction = home if rest_diff > 0 else away
            signals.append(f"Rest advantage: {direction} (+{abs(rest_adj)*100:.1f}%)")

        home_inj = injury_adjustment(home, nba_injuries, "nba")
        away_inj = injury_adjustment(away, nba_injuries, "nba")
        if home_inj > 0.005:
            adj -= home_inj
            signals.append(f"{home} injury drag (-{home_inj*100:.1f}%)")
        if away_inj > 0.005:
            adj += away_inj
            signals.append(f"{away} injuries benefit {home} (+{away_inj*100:.1f}%)")

        # Strength differential signal
        if abs(home_strength - away_strength) > 3:
            stronger = home if home_strength > away_strength else away
            signals.append(f"Net rating edge: {stronger} (season net rtg diff {home_strength-away_strength:+.1f})")

        # Convert adjusted margin to probability
        adjusted_home_prob = min(0.90, max(0.10, _nba_margin_to_prob(base_margin) + adj))
        adjusted_away_prob = 1 - adjusted_home_prob

        # Check home edge
        home_edge = adjusted_home_prob - market_home_prob
        if home_edge >= MIN_EDGE and has_positive_ev(adjusted_home_prob, market_home_prob):
            sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=adjusted_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=_confidence_label(home_edge),
                    signals=signals, home_team=home, away_team=away,
                ))

        # Check away edge
        away_edge = adjusted_away_prob - market_away_prob
        if away_edge >= MIN_EDGE and has_positive_ev(adjusted_away_prob, market_away_prob):
            away_signals = [s.replace(home, "[home]").replace(away, "[away]") for s in signals]
            sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=adjusted_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=_confidence_label(away_edge),
                    signals=away_signals, home_team=home, away_team=away,
                ))

    # --- Total ---
    total = game.get("total")
    if total and nba_ctx["season_stats"]:
        home_stats = nba_ctx["season_stats"].get(home, {})
        away_stats = nba_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            avg_pace = (home_stats.get("pace", 100) + away_stats.get("pace", 100)) / 2
            expected_home_pts = (home_stats.get("off_rtg", 110) + away_stats.get("def_rtg", 110)) / 2 * avg_pace / 100
            expected_away_pts = (away_stats.get("off_rtg", 110) + home_stats.get("def_rtg", 110)) / 2 * avg_pace / 100
            expected_total = expected_home_pts + expected_away_pts

            market_line = total["line"]
            total_std = 14.0
            model_over_prob = float(1 - norm.cdf(market_line, expected_total, total_std))
            market_over_prob = total["over_prob"]
            market_under_prob = total["under_prob"]

            over_edge = model_over_prob - market_over_prob
            under_edge = (1 - model_over_prob) - market_under_prob

            total_signals = [f"Model expected total: {expected_total:.1f} vs market line {market_line}"]
            if home_rest == 0 or away_rest == 0:
                total_signals.append("B2B team reduces pace/scoring")

            if over_edge >= MIN_EDGE and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=f"Over {market_line}",
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing, confidence=_confidence_label(over_edge),
                        signals=total_signals, home_team=home, away_team=away,
                    ))
            if under_edge >= MIN_EDGE and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=f"Under {market_line}",
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing, confidence=_confidence_label(under_edge),
                        signals=total_signals, home_team=home, away_team=away,
                    ))

    return recs


# ---------------------------------------------------------------------------
# MLB Edge Finder
# ---------------------------------------------------------------------------

def _pitcher_quality_score(stats: Dict) -> float:
    """Converts pitcher stats to a quality score (lower FIP = better)."""
    fip = stats.get("fip", 4.20)
    # Normalize: league avg FIP ~4.20, score 0 = avg, positive = better
    return (4.20 - fip) / 1.50   # 1.5 FIP difference ~ ±1 run per game


def analyze_mlb_game(game: Dict, home_pitcher_stats: Dict, away_pitcher_stats: Dict,
                     home_batting: Dict, away_batting: Dict,
                     home_bullpen: Dict, away_bullpen: Dict,
                     mlb_injuries: Dict) -> List[BetRecommendation]:
    from src.data.mlb_stats import get_park_factor
    home = game["home_team"]
    away = game["away_team"]
    label = f"{away} @ {home}"
    venue = game.get("venue", "")
    park_factor = get_park_factor(venue)
    recs = []

    # Build signal list
    signals = []
    home_pitcher_name = game.get("home_pitcher_name", "TBD")
    away_pitcher_name = game.get("away_pitcher_name", "TBD")

    if home_pitcher_stats:
        signals.append(f"{home_pitcher_name} FIP: {home_pitcher_stats.get('fip', '?')} | K/9: {home_pitcher_stats.get('k_per_9', '?')}")
    if away_pitcher_stats:
        signals.append(f"{away_pitcher_name} FIP: {away_pitcher_stats.get('fip', '?')} | K/9: {away_pitcher_stats.get('k_per_9', '?')}")

    home_ops = home_batting.get("ops", 0.720)
    away_ops = away_batting.get("ops", 0.720)
    home_rpg = home_batting.get("runs_per_game", 4.5)
    away_rpg = away_batting.get("runs_per_game", 4.5)

    home_bullpen_era = home_bullpen.get("bullpen_era", 4.20)
    away_bullpen_era = away_bullpen.get("bullpen_era", 4.20)

    # Expected runs using simplified model
    league_avg_runs = 4.5
    home_sp_score = _pitcher_quality_score(home_pitcher_stats) if home_pitcher_stats else 0
    away_sp_score = _pitcher_quality_score(away_pitcher_stats) if away_pitcher_stats else 0

    home_offense_adj = (home_ops - 0.720) / 0.080   # normalized to league avg
    away_offense_adj = (away_ops - 0.720) / 0.080

    home_bullpen_adj = (4.20 - home_bullpen_era) / 2.0
    away_bullpen_adj = (4.20 - away_bullpen_era) / 2.0

    # Expected runs each team scores
    expected_home_runs = max(1.5, league_avg_runs
                             - away_sp_score * 0.8
                             + home_offense_adj * 0.6
                             + away_bullpen_adj * 0.3
                             + MLB_HOME_ADVANTAGE * 5)
    expected_away_runs = max(1.5, league_avg_runs
                             - home_sp_score * 0.8
                             + away_offense_adj * 0.6
                             + home_bullpen_adj * 0.3)

    # Apply park factor
    expected_home_runs *= park_factor
    expected_away_runs *= park_factor

    if park_factor != 1.0:
        direction = "hitter-friendly" if park_factor > 1.02 else "pitcher-friendly"
        signals.append(f"Park factor {park_factor:.2f} ({direction}: {venue})")

    # Injury adjustments
    home_inj = injury_adjustment(home, mlb_injuries, "mlb")
    away_inj = injury_adjustment(away, mlb_injuries, "mlb")
    if home_inj > 0.01:
        signals.append(f"{home} injury impact (-{home_inj*100:.1f}%)")
    if away_inj > 0.01:
        signals.append(f"{away} injury impact (-{away_inj*100:.1f}%)")

    # Win probability using expected run margin + normal approximation
    run_diff = expected_home_runs - expected_away_runs
    model_home_prob = float(norm.cdf(run_diff, 0, MLB_SPREAD_STD))
    model_home_prob = min(0.85, max(0.15, model_home_prob - home_inj + away_inj))
    model_away_prob = 1 - model_home_prob

    signals.append(f"Model expected score: {home} {expected_home_runs:.1f} — {away} {expected_away_runs:.1f}")

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_edge = model_home_prob - market_home_prob
        away_edge = model_away_prob - market_away_prob

        if home_edge >= MIN_EDGE and has_positive_ev(model_home_prob, market_home_prob):
            sizing = robinhood_kelly(model_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=model_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=_confidence_label(home_edge),
                    signals=signals[:], home_team=home, away_team=away,
                ))

        if away_edge >= MIN_EDGE and has_positive_ev(model_away_prob, market_away_prob):
            sizing = robinhood_kelly(model_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=model_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=_confidence_label(away_edge),
                    signals=signals[:], home_team=home, away_team=away,
                ))

    # --- Total ---
    total = game.get("total")
    if total:
        expected_total = expected_home_runs + expected_away_runs
        market_line = total["line"]
        model_over_prob = float(1 - norm.cdf(market_line, expected_total, MLB_SPREAD_STD * 1.5))
        market_over_prob = total["over_prob"]
        market_under_prob = total["under_prob"]

        total_signals = signals[:] + [f"Model expected total: {expected_total:.1f} vs line {market_line}"]

        over_edge = model_over_prob - market_over_prob
        under_edge = (1 - model_over_prob) - market_under_prob

        if over_edge >= MIN_EDGE and has_positive_ev(model_over_prob, market_over_prob):
            sizing = robinhood_kelly(model_over_prob, market_over_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=f"Over {market_line}",
                    market_prob=market_over_prob, model_prob=model_over_prob,
                    edge=over_edge, contract_price=market_over_prob,
                    sizing=sizing, confidence=_confidence_label(over_edge),
                    signals=total_signals, home_team=home, away_team=away,
                ))
        if under_edge >= MIN_EDGE and has_positive_ev(1 - model_over_prob, market_under_prob):
            sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=f"Under {market_line}",
                    market_prob=market_under_prob, model_prob=1 - model_over_prob,
                    edge=under_edge, contract_price=market_under_prob,
                    sizing=sizing, confidence=_confidence_label(under_edge),
                    signals=total_signals, home_team=home, away_team=away,
                ))

    return recs
