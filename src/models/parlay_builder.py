"""
Builds 2-leg parlays from top single-game recommendations.

Robinhood parlay rules enforced here:
  - Totals (Over/Under) can only be combined with bets from the SAME game.
  - Moneyline and Spread bets can be parlayed across different games/leagues.
  - Same-game parlays (any bet type) are valid.
  - Cross-game parlays require BOTH legs to be ML or Spread (no Totals).
"""
from dataclasses import dataclass, field
from itertools import combinations
from typing import List
from src.config import MIN_PARLAY_LEG_EDGE, MAX_PARLAYS, ROBINHOOD_COMMISSION
from src.models.edge_finder import BetRecommendation
from src.models.kelly import parlay_kelly, has_positive_ev, BetSizing

_TOTAL_TYPE = "Total"   # bet_type value used in BetRecommendation for Over/Under bets


@dataclass
class ParlayRecommendation:
    legs: List[BetRecommendation]
    combined_prob: float
    contract_price: float           # estimated Robinhood parlay price
    edge: float
    sizing: BetSizing
    confidence: str
    expected_value: float

    @property
    def label(self) -> str:
        return " + ".join(f"{l.pick} ({l.bet_type})" for l in self.legs)

    @property
    def game_labels(self) -> List[str]:
        return [l.game for l in self.legs]


def _parlay_valid(leg_a: BetRecommendation, leg_b: BetRecommendation) -> bool:
    """
    Returns True only if the leg combination is allowed on Robinhood.

    Rules:
      1. Same game  → always valid (ML+Total, Spread+Total, ML+ML, etc.)
      2. Cross game → valid ONLY if neither leg is a Total (Over/Under).
         ML + ML across games/leagues ✓
         ML + Spread across games ✓
         Total + anything cross-game ✗
    """
    same_game = (leg_a.game == leg_b.game)

    if same_game:
        return True

    # Cross-game: block if either leg is a Total
    if leg_a.bet_type == _TOTAL_TYPE or leg_b.bet_type == _TOTAL_TYPE:
        return False

    return True


def build_parlays(singles: List[BetRecommendation]) -> List[ParlayRecommendation]:
    """
    Takes the top single-game recommendations and builds valid 2-leg parlays.
    Returns up to MAX_PARLAYS sorted by edge descending.
    """
    eligible = [s for s in singles if s.edge >= MIN_PARLAY_LEG_EDGE]
    parlays: List[ParlayRecommendation] = []

    for leg_a, leg_b in combinations(eligible, 2):
        if not _parlay_valid(leg_a, leg_b):
            continue

        combined_true_prob   = leg_a.model_prob  * leg_b.model_prob
        combined_market_prob = leg_a.market_prob * leg_b.market_prob

        parlay_price = round(combined_market_prob, 4)
        edge         = combined_true_prob - combined_market_prob

        if not has_positive_ev(combined_true_prob, parlay_price):
            continue

        sizing = parlay_kelly(combined_true_prob, parlay_price)
        if sizing.num_contracts == 0:
            continue

        # HIGH only when both legs are HIGH confidence
        both_high  = (leg_a.confidence == "HIGH" and leg_b.confidence == "HIGH")
        confidence = "HIGH" if (both_high and edge >= 0.03) else "MEDIUM"

        parlays.append(ParlayRecommendation(
            legs=[leg_a, leg_b],
            combined_prob=round(combined_true_prob, 4),
            contract_price=parlay_price,
            edge=round(edge, 4),
            sizing=sizing,
            confidence=confidence,
            expected_value=sizing.expected_value,
        ))

    # Sort: best leg-quality tier first, then edge within each tier.
    # Tier 0 = both legs HIGH, Tier 1 = one HIGH + one MEDIUM, Tier 2 = both MEDIUM.
    def _tier(p: ParlayRecommendation) -> int:
        high = sum(1 for l in p.legs if l.confidence == "HIGH")
        return 2 - high   # 2 HIGH → 0, 1 HIGH → 1, 0 HIGH → 2

    parlays.sort(key=lambda p: (_tier(p), -p.edge))
    return parlays[:MAX_PARLAYS]
