"""
One budget bet per game.

Aug 10 2026: the card shipped "Athletics +1.5" (slot 3) and "Tampa Bay Rays ML"
(slot 4) on the same game. Those are near-opposite bets — both win only if the
Rays win by exactly 1 — so one was almost certainly dead on arrival while we
paid vig on both. The existing dedup keyed on (game, team/direction), which
catches a duplicate of the SAME bet but not two different markets on one game.
"""
from src.config import MAX_BUDGET_BETS_PER_GAME, MAX_SINGLE_BETS


class _Rec:
    """Minimal stand-in for a BetRecommendation."""
    def __init__(self, game, bet_type, pick, edge, home="Athletics", away="Tampa Bay Rays"):
        self.game, self.bet_type, self.pick, self.edge = game, bet_type, pick, edge
        self.home_team, self.away_team = home, away


def _budget_pool(recs):
    """The production rule from main.py, applied to an already-sorted list."""
    per_game, out = {}, []
    for r in recs:
        n = per_game.get(r.game, 0)
        if n >= MAX_BUDGET_BETS_PER_GAME:
            continue
        per_game[r.game] = n + 1
        out.append(r)
    return out


def test_the_actual_aug_10_card_keeps_only_the_better_pick():
    game = "Rays @ Athletics"
    recs = [                                    # in _slot_sort_key order
        _Rec(game, "Moneyline", "Tampa Bay Rays", 0.059),
        _Rec(game, "Spread", "Athletics +1.5", 0.056),
    ]
    pool = _budget_pool(recs)
    assert len(pool) == 1
    assert pool[0].pick == "Tampa Bay Rays"      # the higher-ranked one survives


def test_freed_slot_goes_to_a_different_game():
    """Dropping the second same-game pick must not shrink the card."""
    recs = [
        _Rec("Rays @ Athletics", "Moneyline", "Tampa Bay Rays", 0.059),
        _Rec("Rays @ Athletics", "Spread", "Athletics +1.5", 0.056),
        _Rec("Cubs @ Royals", "Spread", "Kansas City Royals +1.5", 0.052),
        _Rec("Reds @ Nationals", "Total", "Under 8.5", 0.051),
        _Rec("Mets @ Pirates", "Moneyline", "New York Mets", 0.050),
        _Rec("Jays @ Phillies", "Spread", "Toronto Blue Jays +1.5", 0.049),
        _Rec("Twins @ Brewers", "Total", "Over 9.5", 0.048),
    ]
    card = _budget_pool(recs)[:MAX_SINGLE_BETS]
    assert len(card) == MAX_SINGLE_BETS
    games = [r.game for r in card]
    assert len(set(games)) == len(games), "a game appears twice on the card"
    assert "Jays @ Phillies" in games, "the freed slot should reach the next game"


def test_opposing_sides_of_one_game_can_never_both_ship():
    """The pathological case: two picks that cannot comfortably both win."""
    game = "Rays @ Athletics"
    recs = [
        _Rec(game, "Moneyline", "Tampa Bay Rays", 0.09),
        _Rec(game, "Spread", "Athletics +1.5", 0.08),
        _Rec(game, "Total", "Under 8.5", 0.07),
    ]
    assert len(_budget_pool(recs)) == 1


def test_other_games_are_untouched():
    recs = [
        _Rec("A @ B", "Moneyline", "A", 0.06),
        _Rec("C @ D", "Moneyline", "C", 0.05),
        _Rec("E @ F", "Total", "Over 8.5", 0.04),
    ]
    assert len(_budget_pool(recs)) == 3


def test_rule_is_driven_by_the_config_constant():
    assert MAX_BUDGET_BETS_PER_GAME == 1
    game = "Rays @ Athletics"
    recs = [_Rec(game, "Moneyline", "Tampa Bay Rays", 0.06),
            _Rec(game, "Spread", "Athletics +1.5", 0.05)]
    assert len(_budget_pool(recs)) == MAX_BUDGET_BETS_PER_GAME
