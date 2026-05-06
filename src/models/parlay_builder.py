"""
Builds 2-leg parlays from top single-game recommendations.

Robinhood parlay rules enforced here:

  Cross-game parlays (different games / leagues):
    - ML + ML only.  Spread + anything cross-game is not allowed.
      Total + anything cross-game is not allowed.

  Same-game parlays (SGP — both legs from the same game):
    - Any combination is valid EXCEPT ML + Spread.
    - Valid: ML+Total, ML+Prop, Spread+Total, Spread+Prop, Total+Prop.
    - Invalid: ML+Spread (same game or cross-game).
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
        def _leg_label(l) -> str:
            if l.bet_type == "Moneyline":
                return f"{l.pick} (ML)"
            return l.pick  # Spread/Total: pick text already contains the line
        return " + ".join(_leg_label(l) for l in self.legs)

    @property
    def game_labels(self) -> List[str]:
        return [l.game for l in self.legs]


def _parlay_valid(leg_a: BetRecommendation, leg_b: BetRecommendation) -> bool:
    """
    Returns True only if the leg combination is allowed on Robinhood.

    ML + Spread is never allowed (same game or cross-game).
    Cross-game: only ML + ML is valid.
    Same-game:  any combo except ML + Spread is valid.
    """
    type_a = leg_a.bet_type
    type_b = leg_b.bet_type

    # ML + Spread is invalid in all contexts
    types = {type_a, type_b}
    if types == {"Moneyline", "Spread"}:
        return False

    same_game = (leg_a.game == leg_b.game)

    if same_game:
        # All remaining combos valid as SGP:
        # ML+Total, ML+Prop, Spread+Total, Spread+Prop, Total+Prop
        return True

    # Cross-game: only ML + ML is allowed
    return type_a == "Moneyline" and type_b == "Moneyline"


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

    # Greedy dedup: ensure no single leg appears in more than one parlay.
    # Without this, the two best parlays often share their strongest leg
    # (e.g. A+B and A+C) giving the appearance of the same bet being doubled up.
    selected: List[ParlayRecommendation] = []
    used_leg_ids: set = set()
    for p in parlays:
        leg_ids = {id(l) for l in p.legs}
        if leg_ids & used_leg_ids:   # any leg already in a selected parlay → skip
            continue
        selected.append(p)
        used_leg_ids |= leg_ids
        if len(selected) >= MAX_PARLAYS:
            break

    return selected
