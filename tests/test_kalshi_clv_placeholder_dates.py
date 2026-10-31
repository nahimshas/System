"""CFB tickers carry a date that is not kickoff — and sometimes two of them.

Measured Oct 31 2026 on Pittsburgh @ Boston College, our stored date
2026-10-17:

    KXNCAAFGAME-26OCT17PITTBC   closed 2026-10-06, result "scalar"  <- voided
    KXNCAAFGAME-26OCT16PITTBC   closed 2026-10-17, result yes/no    <- real

`resolve_pick` joins on an EXACT event_date, so it matched the voided ticker,
read no candles, burned all three attempts and gave up permanently. CFB
moneyline CLV coverage sat at 47.5% and CFB spread at 51.4% — half the sport
dark, on Saturdays only, which is every CFB slate that matters. After the fix:
96.7% and 82.9%.

`kalshi_pricing.py` already knew CFB cannot date its own games. This module
never learned it: the fourth instance of a fix landing on N-1 of N call sites.
"""

import unittest

from src.data import kalshi_clv as kc


class TestCandidateDates(unittest.TestCase):
    def test_a_normal_sport_tries_only_its_own_date(self):
        # Nothing may change for sports that date their games correctly.
        for sport in ("NBA", "NHL", "MLB", "NFL", "MLS"):
            self.assertEqual(kc._candidate_dates(sport, "2026-10-17"),
                             ["2026-10-17"], sport)

    def test_cfb_tries_its_own_date_first(self):
        # A correct match must never be displaced by a neighbour.
        self.assertEqual(kc._candidate_dates("CFB", "2026-10-17")[0], "2026-10-17")

    def test_cfb_reaches_the_day_before(self):
        # The real market is filed one day early.
        self.assertIn("2026-10-16", kc._candidate_dates("CFB", "2026-10-17"))

    def test_the_window_is_symmetric_and_bounded(self):
        got = kc._candidate_dates("CFB", "2026-10-17")
        self.assertEqual(len(got), 1 + 2 * kc.PLACEHOLDER_DATE_WINDOW)
        self.assertNotIn("2026-10-14", got)

    def test_a_junk_date_does_not_raise(self):
        self.assertEqual(kc._candidate_dates("CFB", "not-a-date"), ["not-a-date"])


class TestVoidedMarkets(unittest.TestCase):
    def test_the_voided_duplicate_is_rejected(self):
        # result "scalar" on a finalized market: no real outcome, no price.
        self.assertTrue(kc._is_voided(
            {"ticker": "KXNCAAFGAME-26OCT17PITTBC-PITT",
             "status": "finalized", "result": "scalar"}))

    def test_a_really_settled_market_is_kept(self):
        for res in ("yes", "no"):
            self.assertFalse(kc._is_voided(
                {"status": "finalized", "result": res}), res)

    def test_an_open_market_is_never_voided(self):
        # Open books have no result yet; dropping them would blank live CLV.
        self.assertFalse(kc._is_voided({"status": "active", "result": ""}))
        self.assertFalse(kc._is_voided({"status": "open"}))

    def test_junk_does_not_raise(self):
        for junk in ({}, {"status": None}, {"status": 3, "result": object()}):
            kc._is_voided(junk)


class TestResolverVersioning(unittest.TestCase):
    """A resolver fix must reach the rows it was written for.

    The rows most in need of rescue are exactly the ones already past
    MAX_ATTEMPTS — 119 were stranded that way when this fix landed.
    """

    def test_a_row_under_the_retry_limit_is_not_exhausted(self):
        self.assertFalse(kc._exhausted({"kalshi_clv_attempts": 1}))

    def test_a_row_that_gave_up_under_an_older_resolver_gets_another_pass(self):
        self.assertFalse(kc._exhausted(
            {"kalshi_clv_attempts": kc.MAX_ATTEMPTS,
             "kalshi_resolver_version": kc.RESOLVER_VERSION - 1}))

    def test_a_row_with_no_version_counts_as_oldest(self):
        self.assertFalse(kc._exhausted({"kalshi_clv_attempts": kc.MAX_ATTEMPTS}))

    def test_a_row_that_gave_up_under_this_resolver_stays_given_up(self):
        # Otherwise a genuinely unmeasurable line is retried forever.
        self.assertTrue(kc._exhausted(
            {"kalshi_clv_attempts": kc.MAX_ATTEMPTS,
             "kalshi_resolver_version": kc.RESOLVER_VERSION}))

    def test_junk_is_treated_as_exhausted_not_retried_forever(self):
        self.assertTrue(kc._exhausted({"kalshi_clv_attempts": "lots"}))


class TestOneResolverForBothPasses(unittest.TestCase):
    def test_both_passes_go_through_the_same_helper(self):
        """They carried identical copies of the resolve-and-price logic.

        The placeholder-date fix would otherwise have landed on one of them,
        which is the very bug shape that caused this.
        """
        import inspect
        src = inspect.getsource(kc)
        self.assertEqual(src.count("_resolve_and_price("), 3,
                         "expected one definition and two call sites")
        shadow = src.split("def update_shadow_log_kalshi_clv")[1].split("\ndef ")[0]
        decision = src.split("def update_decision_log_kalshi_clv")[1].split("\ndef ")[0]
        for name, body in (("shadow", shadow), ("decision", decision)):
            self.assertIn("_resolve_and_price(", body, name)
            self.assertNotIn("resolve_pick(", body,
                             f"{name} pass still resolves on its own")


if __name__ == "__main__":
    unittest.main()
