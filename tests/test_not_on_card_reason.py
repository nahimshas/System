"""Why a sport-tab pick is not on the money card.

The tab and the card legitimately disagree: the card applies gates the display
does not. Before this, a tab card that was not staked carried no sign of it, so
"Detroit Pistons -7.5" sat in the NBA tab while the card held the moneyline with
nothing to explain the difference — and someone reading the tab as a bet list
could stake a pick the governor had refused.
"""

import unittest


def _reason(rec, gated=frozenset(), card=()):
    """Exercise the generator's helper with a controlled environment."""
    from src.config import BUDGET_MIN_EDGE

    # Mirror of generator._not_on_card_reason, parameterised for the test. The
    # ORDER is the contract under test: "cannot" and "refused" before the slot
    # race, because a pick that was never eligible did not lose one.
    blk = rec.get("budget_block")
    if blk:
        return blk
    sport = (rec.get("sport") or "").upper()
    market = rec.get("bet_type") or ""
    if (sport, market) in gated:
        return f"{sport} {market} is gated"
    if (rec.get("num_contracts") or 0) <= 0:
        return "stake rounds to under one contract"
    eff = rec.get("effective_edge", rec.get("edge") or 0)
    if eff < BUDGET_MIN_EDGE:
        return "under the real-money floor"
    for pick, game in card:
        if game == rec.get("game") and pick != rec.get("pick"):
            return f"one bet per game — slot went to {pick}"
    if card:
        return "outside the top slots"
    return None


class TestReasonOrder(unittest.TestCase):
    """Each case below is a real pick from 2026-10-30."""

    def test_a_gated_market_is_reported_as_gated_not_as_a_lost_race(self):
        # Pistons -7.5 had the biggest edge on the board (15%) and sized to 2
        # contracts. It was refused because NBA Spread is CLV-gated. Reporting
        # "one bet per game" here would say it lost a fair contest.
        rec = {"sport": "NBA", "bet_type": "Spread", "pick": "Detroit Pistons -7.5",
               "game": "Detroit Pistons @ Brooklyn Nets",
               "num_contracts": 2, "effective_edge": 0.140969}
        r = _reason(rec, gated={("NBA", "Spread")},
                    card=[("Detroit Pistons", "Detroit Pistons @ Brooklyn Nets")])
        self.assertIn("gated", r)
        self.assertNotIn("one bet per game", r)

    def test_a_zero_stake_is_reported_as_a_stake_problem(self):
        # Panthers -1.5: Kelly asked $0.43 against a $0.465 contract — 92% of
        # a contract, truncated to zero. NHL Spread is NOT gated.
        rec = {"sport": "NHL", "bet_type": "Spread", "pick": "Florida Panthers -1.5",
               "game": "Winnipeg Jets @ Florida Panthers",
               "num_contracts": 0, "effective_edge": 0.066051}
        r = _reason(rec, card=[("Florida Panthers", "Winnipeg Jets @ Florida Panthers")])
        self.assertIn("contract", r)
        self.assertNotIn("one bet per game", r)

    def test_an_eligible_pick_that_lost_its_game_says_so(self):
        rec = {"sport": "NHL", "bet_type": "Spread", "pick": "X -1.5",
               "game": "A @ B", "num_contracts": 3, "effective_edge": 0.09}
        self.assertIn("one bet per game", _reason(rec, card=[("Y", "A @ B")]))

    def test_an_eligible_pick_elsewhere_on_the_slate_is_a_slot_race(self):
        rec = {"sport": "NHL", "bet_type": "Spread", "pick": "X -1.5",
               "game": "A @ B", "num_contracts": 3, "effective_edge": 0.09}
        self.assertEqual(_reason(rec, card=[("Y", "C @ D")]), "outside the top slots")

    def test_a_sub_floor_edge_is_named_before_the_slot_race(self):
        rec = {"sport": "NBA", "bet_type": "Moneyline", "pick": "X",
               "game": "A @ B", "num_contracts": 2, "effective_edge": 0.02}
        self.assertIn("floor", _reason(rec, card=[("Y", "C @ D")]))

    def test_a_stamped_reason_from_the_run_wins(self):
        # budget_block is stamped at routing time and is the most precise.
        rec = {"sport": "NBA", "bet_type": "Spread", "pick": "X", "game": "A @ B",
               "num_contracts": 0, "budget_block": "exact reason from the run"}
        self.assertEqual(_reason(rec), "exact reason from the run")


class TestTemplateScope(unittest.TestCase):
    def test_only_budget_cards_carry_the_not_bet_badge(self):
        # Watchlist sports never route to money, so the badge would be true of
        # every card in those tabs and therefore tell the reader nothing.
        import io
        t = io.open("src/report/templates/report_spa.html", encoding="utf-8").read()
        self.assertEqual(t.count("badge-notbet"), 2)   # one style rule, one use
        # Bound each macro at its own endmacro. Splitting on "{% macro " alone
        # leaves the LAST segment running to end-of-file, swallowing the
        # stylesheet — which made the first draft of this test fail on the
        # style rule rather than on a real badge.
        def macro_body(name):
            start = t.index("{% macro " + name + "(")
            return t[start:t.index("{% endmacro %}", start)]

        bet_card = macro_body("bet_card")
        watchlist = macro_body("watchlist_card")
        prop = macro_body("prop_card")
        self.assertIn("badge-notbet", bet_card)
        self.assertNotIn("badge-notbet", watchlist)
        self.assertNotIn("badge-notbet", prop)

    def test_the_generator_prefers_the_stamped_reason(self):
        import inspect
        from src.report import generator
        src = inspect.getsource(generator)
        self.assertIn("budget_block", src)
        self.assertIn("not_on_card", src)

    def test_main_has_one_definition_of_budget_eligibility(self):
        import inspect
        import src.main as m
        src = inspect.getsource(m._budget_block_reason)
        for gate in ("BUDGET_EXCLUDED_MARKETS", "BUDGET_MIN_EDGE",
                     "clv_gate", "outcome_gate", "num_contracts"):
            self.assertIn(gate, src, gate)


if __name__ == "__main__":
    unittest.main()
