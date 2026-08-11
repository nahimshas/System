"""
Builds 2-leg parlays from top single-game recommendations.

Leg-combination rules (updated Jul 2026 — Robinhood lifted its parlay
restrictions, so any combination is now PLATFORM-legal; the remaining
rules here are the MODEL's own):

  Cross-game parlays (different games / leagues):
    - Any bet-type combination is allowed (ML+ML, ML+Spread, Spread+Spread,
      etc.). Legs from different games are treated as independent, so the
      combined probability math (p_a × p_b) is sound.

  Same-game parlays (SGP — both legs from the same game):
    - BLOCKED ENTIRELY (Aug 10 2026). This is a model-correctness rule, not a
      platform rule. Two legs from one game are never independent, so p_a × p_b
      is simply the wrong number — and wrong in a direction that varies with
      the pair (a spread and an under move together; an ML and an under often
      oppose), so we cannot even say whether it flatters or penalises the
      parlay. Previously only ML+Spread was blocked, which left Spread+Total
      and ML+Total priced as if independent. Extends the same reasoning as
      MAX_BUDGET_BETS_PER_GAME=1 for singles: one game, one bet.
      Rare in practice — 33 of the 35 historical same-game parlays predate
      June 2026, and the one July case was the duplicate-leg bug. This is a
      safety rail, not a change in day-to-day behaviour. Revisit only with a
      real correlation model, never by relaxing the rule alone.
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
    Returns True if the leg combination is allowed.

    Robinhood no longer restricts combinations (Jul 2026), so the only
    remaining rule is the model's own: BOTH LEGS MUST BE FROM DIFFERENT GAMES.
    Legs from one game are correlated, and the combined-probability math
    (p_a × p_b) assumes independence — so the resulting edge is wrong by an
    unknown amount in an unknown direction. Cross-game, any combination is
    valid, including Spread + Spread, which lets two dog-with-better-starter
    picks (the model's best-performing bet type) be parlayed together.
    """
    return leg_a.game != leg_b.game


def build_parlays(singles: List[BetRecommendation]) -> List[ParlayRecommendation]:
    """
    Takes the top single-game recommendations and builds valid 2-leg parlays.
    Returns up to MAX_PARLAYS sorted by edge descending.
    """
    # Dedupe identical bets first (the raw budget pool can carry the same pick
    # twice after a line move — keep the higher-edge copy). Without this, a
    # duplicated pick could parlay with itself (bit us Jul 7 2026:
    # "Cardinals +1.5 + Cardinals +1.5").
    _best: dict = {}
    for s in singles:
        k = (s.game, s.bet_type, s.pick)
        if k not in _best or s.edge > _best[k].edge:
            _best[k] = s
    eligible = [s for s in _best.values() if s.edge >= MIN_PARLAY_LEG_EDGE]
    parlays: List[ParlayRecommendation] = []

    for leg_a, leg_b in combinations(eligible, 2):
        if not _parlay_valid(leg_a, leg_b):
            continue

        # Use calibrated leg probabilities when available (stamped in main.py)
        # — a parlay compounds any per-leg overconfidence, so it must be the
        # first consumer of the corrected numbers. Falls back to raw model
        # probs for legs without a calibration stamp (Phase 0 = identical).
        _pa = getattr(leg_a, "model_prob_calibrated", None) or leg_a.model_prob
        _pb = getattr(leg_b, "model_prob_calibrated", None) or leg_b.model_prob
        combined_true_prob   = _pa * _pb
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
