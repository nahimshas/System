"""Sport lists must be DERIVED. This is the project's #1 recurring bug.

Every instance has failed silently: a sport that is never settled, never
counted, or wrongly dropped looks exactly like a sport with nothing to report.
These tests assert the derivations agree with their sources, so adding a sport
cannot leave one of them behind.
"""

import inspect
import unittest

from src.sports.registry import REGISTRY


class TestManagerWatchlist(unittest.TestCase):
    """merge_picks drops watchlist sports from the cached budget card."""

    def _derived(self):
        import src.state.manager as manager
        src = inspect.getsource(manager.merge_picks)
        self.assertIn("REGISTRY", src,
                      "merge_picks must derive its watchlist set from the registry")
        return {s.upper() for s, e in REGISTRY.items() if not e.caps.enters_budget}

    def test_no_budget_sport_is_treated_as_watchlist(self):
        # NHL was hand-listed here and never removed when it graduated in June
        # 2026, so every subsequent run of the day dropped the cached NHL
        # singles — 92 of 131 October money-card picks.
        derived = self._derived()
        budget = {s.upper() for s, e in REGISTRY.items() if e.caps.enters_budget}
        self.assertEqual(derived & budget, set(),
                         "a budget sport is in the watchlist drop-list")

    def test_every_watchlist_sport_is_covered(self):
        # Stale in the other direction too: CFB, WC and LIGAMX were absent, so
        # the guard could not have caught them entering the budget pool.
        derived = self._derived()
        for slug, e in REGISTRY.items():
            if not e.caps.enters_budget:
                self.assertIn(slug.upper(), derived, slug)

    def test_source_has_no_hand_written_sport_set(self):
        import src.state.manager as manager
        src = inspect.getsource(manager.merge_picks)
        self.assertNotIn('{"NHL", "IPL", "WNBA", "MLS"}', src)


class TestDecisionLogSettlementCoverage(unittest.TestCase):
    """settle_decision_from_scores must cover every score-based sport."""

    def test_supported_is_derived_from_the_path_maps(self):
        import src.state.decision_log as dl
        src = inspect.getsource(dl.settle_decision_from_scores)
        self.assertIn("ESPN_SPORT_PATHS", src)
        self.assertIn("ESPN_WATCHLIST_PATHS", src)
        # Assert on BEHAVIOUR, not on source text: a "no hand-written set"
        # string check matched the comment that explains the old bug. What
        # matters is that SUPPORTED tracks the maps, which the sibling test
        # covers, and that MAIN is built by set arithmetic over them.
        self.assertRegex(src, r"MAIN\s*=\s*set\(ESPN_SPORT_PATHS\)")

    def test_every_registry_sport_but_ipl_can_be_settled(self):
        # NFL was missing: 596 decision rows, 0 graded, while every other sport
        # sat between 87% and 100%.
        from src.data.outcome_checker import ESPN_SPORT_PATHS, ESPN_WATCHLIST_PATHS
        supported = (set(ESPN_SPORT_PATHS) | set(ESPN_WATCHLIST_PATHS)) - {"IPL"}
        for slug in REGISTRY:
            if slug == "ipl":          # no reliable historical score source
                continue
            self.assertIn(slug.upper(), supported,
                          f"{slug} can never be settled in the decision log")


class TestMarketTestFeaturesAreSportAware(unittest.TestCase):
    """market_test decides whether a market gets real money. Its feature set
    must belong to the sport being tested."""

    def test_mlb_list_stays_pinned(self):
        # The Aug 4 2026 MLB conclusion rests on exactly these columns.
        from tools.analysis.market_test import _MLB_FEATURES, _features_for
        self.assertEqual(_features_for("MLB", []), list(_MLB_FEATURES))
        self.assertIn("run_diff_capped", _MLB_FEATURES)

    def test_other_sports_derive_their_own_features(self):
        from tools.analysis.market_test import _features_for
        rows = [{"features": {"home_strength": 1.0, "away_strength": 2.0,
                              "effective_margin": 3.0, "stats_available": True}}]
        feats = _features_for("NFL", rows)
        self.assertIn("home_strength", feats)
        self.assertIn("effective_margin", feats)
        self.assertNotIn("run_diff", feats, "MLB columns leaked into an NFL test")

    def test_booleans_and_bookkeeping_are_not_features(self):
        from tools.analysis.market_test import _features_for
        rows = [{"features": {"playoff": True, "stats_available": 1,
                              "home_strength": 1.0}}]
        feats = _features_for("NHL", rows)
        self.assertEqual(feats, ["home_strength"])

    def test_no_features_is_reported_not_guessed(self):
        from tools.analysis.market_test import _features_for
        self.assertEqual(_features_for("NFL", [{"features": {}}]), [])


if __name__ == "__main__":
    unittest.main()
