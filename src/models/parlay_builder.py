"""Builds 2-leg parlays from top single-game recommendations."""
from dataclasses import dataclass, field
from itertools import combinations
from typing import List
from src.config import MIN_PARLAY_LEG_EDGE, MAX_PARLAYS, ROBINHOOD_COMMISSION
from src.models.edge_finder import BetRecommendation
from src.models.kelly import parlay_kelly, has_positive_ev, BetSizing


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


def build_parlays(singles: List[BetRecommendation]) -> List[ParlayRecommendation]:
    """
    Takes the top single-game recommendations and builds 2-leg parlays.

    Parlay rules:
    - Both legs must have >= MIN_PARLAY_LEG_EDGE individual edge
    - Legs must come from different games
    - Combined true probability must yield positive EV vs combined market price
    - Parlay contract price ≈ product of individual market probabilities
      (Robinhood prices parlay legs multiplicatively like a correlated market)
    """
    eligible = [s for s in singles if s.edge >= MIN_PARLAY_LEG_EDGE]
    parlays: List[ParlayRecommendation] = []

    for leg_a, leg_b in combinations(eligible, 2):
        # Must be different games
        if leg_a.game == leg_b.game:
            continue

        combined_true_prob = leg_a.model_prob * leg_b.model_prob
        combined_market_prob = leg_a.market_prob * leg_b.market_prob

        # Robinhood parlay price = combined market prob
        parlay_price = round(combined_market_prob, 4)

        edge = combined_true_prob - combined_market_prob

        if not has_positive_ev(combined_true_prob, parlay_price):
            continue

        sizing = parlay_kelly(combined_true_prob, parlay_price)
        if sizing.num_contracts == 0:
            continue

        ev = sizing.expected_value
        confidence = "HIGH" if edge >= 0.05 else "MEDIUM"

        parlays.append(ParlayRecommendation(
            legs=[leg_a, leg_b],
            combined_prob=round(combined_true_prob, 4),
            contract_price=parlay_price,
            edge=round(edge, 4),
            sizing=sizing,
            confidence=confidence,
            expected_value=ev,
        ))

    # Sort by edge descending, return top N
    parlays.sort(key=lambda p: p.edge, reverse=True)
    return parlays[:MAX_PARLAYS]
