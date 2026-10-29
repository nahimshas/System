"""The quarantine must exclude exactly the broken window and nothing else."""

import unittest

from src.state.quarantine import QUARANTINE, is_quarantined, summary


def _row(sport, market, date, outcome="win"):
    return {"sport": sport, "market_type": market, "date": date, "outcome": outcome}


class TestMlsXgWindow(unittest.TestCase):
    """MLS attacking xG read 0.0 from the first logged row until 2026-10-28."""

    def test_inside_the_window_is_quarantined_for_every_market(self):
        # ALL markets, not just Total/Draw: every MLS probability comes off the
        # same two Poisson means, so the moneyline number was wrong too even
        # though the ranking it produced looked fine.
        for market in ("Total", "Draw", "Moneyline", "Spread"):
            self.assertTrue(is_quarantined(_row("MLS", market, "2026-08-15")), market)

    def test_window_edges_are_inclusive(self):
        self.assertTrue(is_quarantined(_row("MLS", "Total", "2026-05-17")))
        self.assertTrue(is_quarantined(_row("MLS", "Total", "2026-10-28")))

    def test_after_the_fix_is_not_quarantined(self):
        # The fix landed after the 2026-10-28 run; the first clean MLS rows will
        # carry a later date and MUST teach, or the sport never relearns.
        self.assertFalse(is_quarantined(_row("MLS", "Total", "2026-10-29")))
        self.assertFalse(is_quarantined(_row("MLS", "Total", "2026-11-14")))

    def test_before_the_window_is_not_quarantined(self):
        self.assertFalse(is_quarantined(_row("MLS", "Total", "2026-05-16")))

    def test_other_sports_are_untouched(self):
        for sport in ("MLB", "NBA", "NFL", "NHL", "CFB", "WNBA", "LIGAMX", "WC", "IPL"):
            self.assertFalse(is_quarantined(_row(sport, "Total", "2026-08-15")), sport)


class TestSafety(unittest.TestCase):
    def test_malformed_rows_never_raise_and_are_not_quarantined(self):
        # Fail-open: a bad row may only leave previous behaviour in place.
        for bad in ({}, {"sport": None}, {"sport": "MLS"},
                    {"sport": "MLS", "date": None, "market_type": None},
                    {"sport": 7, "date": 7, "market_type": 7}):
            self.assertFalse(is_quarantined(bad), bad)

    def test_sport_match_is_case_insensitive(self):
        self.assertTrue(is_quarantined(_row("mls", "Total", "2026-08-15")))

    def test_summary_counts_rows_and_settled_separately(self):
        rows = [_row("MLS", "Total", "2026-08-15"),
                _row("MLS", "Draw", "2026-08-15", outcome=None),
                _row("MLB", "Total", "2026-08-15")]
        s = summary(rows)
        mls = [w for w in s if w["sport"] == "MLS"][0]
        self.assertEqual(mls["rows"], 2)
        self.assertEqual(mls["settled_excluded"], 1)

    def test_summary_covers_every_window(self):
        # Derived from QUARANTINE, so a window added later cannot go unreported
        # — the silent-filter failure mode this module exists to avoid.
        self.assertEqual(len(summary([])), len(QUARANTINE))

    def test_every_window_documents_its_cause_and_its_fix(self):
        for w in QUARANTINE:
            self.assertTrue(w.reason.strip(), w.sport)
            self.assertTrue(w.fixed_by.strip(), w.sport)
            self.assertLessEqual(w.date_from, w.date_to, w.sport)


class TestConsumersAreWired(unittest.TestCase):
    """Both LEARNING consumers must filter; the archives must not."""

    def test_calibration_filters(self):
        import inspect
        from src.state import calibration
        src = inspect.getsource(calibration._load_all_settled_entries)
        self.assertIn("_is_quarantined", src)

    def test_clv_governor_filters(self):
        import inspect
        from src.state import clv_governor
        src = inspect.getsource(clv_governor.compute_clv_stats)
        self.assertIn("_is_quarantined", src)

    def test_decision_log_does_not_filter(self):
        # The decision log is a pure archive. Filtering it would destroy the
        # only record of what the broken model actually saw.
        import inspect
        from src.state import decision_log
        self.assertNotIn("quarantine", inspect.getsource(decision_log).lower())


if __name__ == "__main__":
    unittest.main()
