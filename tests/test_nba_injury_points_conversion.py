"""One win-probability unit is worth one number of points (Nov 7 2026).

The NBA analyzer converted the SAME injury adjustment into points at two
different rates inside one function: the moneyline path used
`_NBA_PROB_TO_MARGIN` (σ/φ(0) = 30.1), the totals path a hand-written 75.0.

The totals side therefore subtracted ~6.2 pts on an average card, firing on
115 of 116 graded games, and pulled the projected total 5.2 pts below the
actual (z=-3.08). Before penalties the model was unbiased (+2.2, z=+1.30), so
the base was never the problem — only the conversion.

⚠️ These tests pin the RELATIONSHIP, not the number. A test asserting
`== 30.08` would pass while the two conversions silently diverged again the
next time NBA_SPREAD_STD moved.
"""
import math

from scipy.stats import norm

import src.models.edge_finder as ef


class TestTheTwoConversionsAgree:
    def test_injury_points_use_the_models_own_prob_to_margin_rate(self):
        assert ef._NBA_INJ_TO_PTS == ef._NBA_PROB_TO_MARGIN

    def test_that_rate_is_derived_from_the_spread_distribution(self):
        """σ/φ(0) — the slope of the same normal `_score_from_prob` inverts."""
        expected = 1.0 / norm.pdf(0, 0, ef.NBA_SPREAD_STD)
        assert math.isclose(ef._NBA_INJ_TO_PTS, expected, rel_tol=1e-9)

    def test_it_tracks_a_change_to_the_spread_std(self, monkeypatch):
        """Hardcoding today's value would pass this file and still drift."""
        import importlib
        monkeypatch.setattr(ef, "NBA_SPREAD_STD", 10.0, raising=True)
        mod = importlib.reload(ef)
        try:
            assert mod._NBA_INJ_TO_PTS == mod._NBA_PROB_TO_MARGIN
        finally:
            importlib.reload(ef)

    def test_the_old_value_is_gone(self):
        """75.0 was 2.5x the model's own rate."""
        assert abs(ef._NBA_INJ_TO_PTS - 75.0) > 40


class TestMagnitude:
    def test_a_typical_injury_costs_a_believable_number_of_points(self):
        """Measured mean NBA injury adjustment is ~0.02-0.04 win prob."""
        for adj in (0.02, 0.04):
            pts = adj * ef._NBA_INJ_TO_PTS
            assert 0.4 < pts < 1.5, (adj, pts)

    def test_the_hard_six_point_cap_can_no_longer_bind(self):
        """INJURY_DRAG_CAP bounds the input below the points cap.

        Asserted so the cap is known to be slack rather than quietly assumed
        live — if a future change to either bound makes it bite again, this
        fails and the comment beside it gets revisited.
        """
        from src.data.injuries import INJURY_DRAG_CAP
        assert INJURY_DRAG_CAP * ef._NBA_INJ_TO_PTS < 6.0

    def test_both_call_sites_share_the_one_constant(self):
        """Display and pricing must not drift apart — they are the same number.

        The projected score on the card and expected_total behind the
        Over/Under price are computed by two separate blocks; the whole point
        of a shared constant is that a fix cannot land on only one of them.
        """
        import inspect
        src = inspect.getsource(ef.analyze_nba_game)
        assert src.count("_NBA_INJ_TO_PTS") >= 4, src.count("_NBA_INJ_TO_PTS")
        assert "75.0" not in src.split("_NBA_INJ_TO_PTS")[0][-400:]
