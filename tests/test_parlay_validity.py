"""Parlay leg-combination rules — guards the Jul 7 2026 self-parlay bug
("Cardinals +1.5 + Cardinals +1.5") and the same-game correctness rules."""
from src.models.edge_finder import BetRecommendation
from src.models.kelly import BetSizing
from src.models.parlay_builder import _parlay_valid, build_parlays


def _rec(game, bet_type, pick, edge=0.06, prob=0.60, market=0.54):
    return BetRecommendation(
        sport="MLB", game=game, bet_type=bet_type, pick=pick,
        market_prob=market, model_prob=prob, edge=edge, contract_price=market,
        sizing=BetSizing(dollar_allocation=5.0, num_contracts=10,
                         contract_price=market, total_cost=5.2,
                         profit_if_win=4.0, loss_if_lose=5.2,
                         expected_value=0.5, kelly_fraction=0.05),
        confidence="MEDIUM",
    )


class TestParlayValidity:
    def test_identical_pick_pair_invalid(self):
        a = _rec("MIL @ STL", "Spread", "STL +1.5")
        b = _rec("MIL @ STL", "Spread", "STL +1.5")
        assert not _parlay_valid(a, b)

    def test_same_game_same_market_opposite_sides_invalid(self):
        a = _rec("MIL @ STL", "Spread", "STL +1.5")
        b = _rec("MIL @ STL", "Spread", "MIL -1.5")
        assert not _parlay_valid(a, b)

    def test_same_game_ml_spread_invalid(self):
        a = _rec("MIL @ STL", "Moneyline", "STL")
        b = _rec("MIL @ STL", "Spread", "STL +1.5")
        assert not _parlay_valid(a, b)

    def test_cross_game_spread_spread_valid(self):
        a = _rec("MIL @ STL", "Spread", "STL +1.5")
        b = _rec("COL @ LAD", "Spread", "COL +1.5")
        assert _parlay_valid(a, b)

    def test_cross_game_ml_spread_valid(self):
        a = _rec("MIL @ STL", "Moneyline", "STL")
        b = _rec("COL @ LAD", "Spread", "COL +1.5")
        assert _parlay_valid(a, b)

    def test_build_parlays_dedupes_duplicate_recs(self):
        # Same bet twice (line-move duplicate) + one other pick: the only valid
        # parlay pairs the deduped bet with the other pick — never with itself.
        dup1 = _rec("MIL @ STL", "Spread", "STL +1.5", edge=0.08)
        dup2 = _rec("MIL @ STL", "Spread", "STL +1.5", edge=0.06)
        other = _rec("COL @ LAD", "Spread", "COL +1.5", edge=0.07)
        parlays = build_parlays([dup1, dup2, other])
        for p in parlays:
            legs = [(l.game, l.bet_type, l.pick) for l in p.legs]
            assert len(set(legs)) == len(legs), f"self-parlay produced: {p.label}"
