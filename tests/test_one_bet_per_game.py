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


# ---------------------------------------------------------------------------
# Same rule for parlay legs (Aug 10 2026). Previously only same-game ML+Spread
# was blocked, so Spread+Total and ML+Total on one game were priced as if the
# legs were independent.
# ---------------------------------------------------------------------------

class _Leg:
    def __init__(self, game, bet_type, pick="X", edge=0.06):
        self.game, self.bet_type, self.pick, self.edge = game, bet_type, pick, edge


def test_no_two_parlay_legs_from_the_same_game():
    from src.models.parlay_builder import _parlay_valid
    g = "Rays @ Athletics"
    for a, b in (("Spread", "Total"), ("Moneyline", "Total"),
                 ("Moneyline", "Spread"), ("Spread", "Spread"),
                 ("Total", "Total"), ("Moneyline", "Moneyline")):
        assert _parlay_valid(_Leg(g, a), _Leg(g, b)) is False, f"{a}+{b} same game allowed"


def test_cross_game_combinations_all_still_allowed():
    from src.models.parlay_builder import _parlay_valid
    for a, b in (("Spread", "Spread"), ("Moneyline", "Spread"),
                 ("Total", "Total"), ("Moneyline", "Total")):
        assert _parlay_valid(_Leg("A @ B", a), _Leg("C @ D", b)) is True


def test_builder_produces_no_same_game_parlay():
    """End-to-end: a pool where the two best picks share a game must not
    parlay them together."""
    from src.models.parlay_builder import build_parlays
    from src.models.edge_finder import BetRecommendation
    import inspect

    fields = inspect.signature(BetRecommendation).parameters
    def mk(game, bet_type, pick, edge, home, away):
        kw = {}
        for name in fields:
            kw[name] = None
        kw.update(dict(game=game, bet_type=bet_type, pick=pick, edge=edge,
                       home_team=home, away_team=away))
        for k, v in (("model_prob", 0.60), ("market_prob", 0.54),
                     ("confidence", "MEDIUM"), ("sport", "MLB"),
                     ("signals", []), ("contract_price", 0.54)):
            if k in fields:
                kw[k] = v
        return BetRecommendation(**{k: v for k, v in kw.items() if k in fields})

    pool = [
        mk("Rays @ Athletics", "Moneyline", "Tampa Bay Rays", 0.09, "Athletics", "Tampa Bay Rays"),
        mk("Rays @ Athletics", "Total", "Under 8.5", 0.08, "Athletics", "Tampa Bay Rays"),
    ]
    for par in build_parlays(pool):
        assert len(set(par.game_labels)) == len(par.game_labels), \
            f"same-game parlay built: {par.label}"
