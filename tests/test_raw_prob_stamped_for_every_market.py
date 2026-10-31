"""Every market a sport emits must carry its PRE-CAP probability.

`model_prob_raw` is what the project's own rule means by "read the raw number,
not the post-cap one" — a market pinned at its credibility cap is a broken
input wearing a tidy +10% edge. The soccer Draw market carried `None` from the
day it shipped until Oct 31 2026, because `_stamp_recs_calibration` branched on
`bt in ("Moneyline", "Spread")` and nothing else claimed Draw. 166 rows went
into the shadow log blind, and the raw-gap sanity check could never fire on the
market that pinned at its cap more often than any other we run.

So these tests guard the SHAPE, not the instance: whatever market is added
next, it must land in a branch that stamps a raw prob.
"""

import ast
import unittest

from src.models.edge_finder import _stamp_recs_calibration


class _Rec:
    """Minimal stand-in for BetRecommendation's stamped fields."""

    def __init__(self, pick, bet_type):
        self.pick = pick
        self.bet_type = bet_type
        self.model_prob_raw = None
        self.credibility_cap_fired = False
        self.injury_cap_fired = False
        self.hardcap_fired = False
        self.stats_available = True
        self.game_type = ""
        self.research = []


class TestDrawCarriesItsRawProb(unittest.TestCase):
    def _stamp(self, recs, **kw):
        opts = dict(home_team="FC Cincinnati", away_team="D.C. United",
                    home_raw=0.61, away_raw=0.17, draw_raw=0.22)
        opts.update(kw)
        _stamp_recs_calibration(recs, **opts)
        return recs

    def test_draw_gets_the_pre_cap_prob(self):
        rec = self._stamp([_Rec("Draw", "Draw")])[0]
        self.assertAlmostEqual(rec.model_prob_raw, 0.22)

    def test_the_other_three_way_sides_still_work(self):
        home, away = self._stamp([_Rec("FC Cincinnati", "Moneyline"),
                                  _Rec("D.C. United", "Moneyline")])
        self.assertAlmostEqual(home.model_prob_raw, 0.61)
        self.assertAlmostEqual(away.model_prob_raw, 0.17)

    def test_a_pre_stamped_raw_prob_is_not_overwritten(self):
        # Totals and props stamp themselves before this function runs.
        rec = _Rec("Over 2.5", "Total")
        rec.model_prob_raw = 0.44
        self._stamp([rec])
        self.assertAlmostEqual(rec.model_prob_raw, 0.44)

    def test_omitting_draw_raw_does_not_raise(self):
        # Non-soccer analyzers never pass it; they must stay unaffected.
        rec = self._stamp([_Rec("Draw", "Draw")], draw_raw=None)[0]
        self.assertIsNone(rec.model_prob_raw)


class TestNoMarketIsLoggedBlind(unittest.TestCase):
    """The decision log's 4th tuple element is the pre-cap prob.

    A literal `None` there makes that market permanently invisible to future
    analysis — there is no backfill, a feature only has history from the day it
    is wired in. Every Draw tuple in all three soccer analyzers was `None`.
    """

    def test_no_candidate_tuple_passes_a_literal_none_raw_prob(self):
        tree = ast.parse(open("src/models/edge_finder.py").read())
        offenders = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "id", "") == "_stamp_decision"):
                continue
            if len(node.args) < 4:
                continue
            candidates = node.args[3]
            if not isinstance(candidates, (ast.List, ast.Tuple)):
                continue
            for el in candidates.elts:
                if not (isinstance(el, (ast.Tuple, ast.List)) and len(el.elts) >= 4):
                    continue
                raw = el.elts[3]
                if isinstance(raw, ast.Constant) and raw.value is None:
                    offenders.append(
                        f"line {node.lineno}: {ast.unparse(el.elts[0])} / "
                        f"{ast.unparse(el.elts[1])}")
        self.assertEqual(offenders, [], "\n".join(
            ["these markets are logged without their pre-cap probability:"]
            + offenders))

if __name__ == "__main__":
    unittest.main()
