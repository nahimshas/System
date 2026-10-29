"""The outcome governor gates real money. Conservatism is the requirement."""

import unittest
from datetime import date

from src.state import outcome_governor as og


def _rows(n, win_rate, implied, sport="TEST", market="Moneyline", outcome=None):
    wins = int(round(n * win_rate))
    out = []
    for i in range(n):
        out.append({
            "sport": sport, "market_type": market, "date": "2026-10-01",
            "outcome": outcome or ("win" if i < wins else "loss"),
            "market_prob_at_first_pick": implied,
        })
    return out


def _stats(rows):
    """Summarise + phase/gate without touching the shadow log."""
    st = og._summarise([r for r in rows if og._usable(r)])
    if st is None:
        return None
    phase, gated = og._phase_and_gate(st)
    st["phase"], st["gated_on_own_record"] = phase, gated
    return st


class TestGatesOnlyWhenBothConditionsHold(unittest.TestCase):
    def test_a_large_significant_deficit_gates(self):
        # The NFL shape: 43.4% actual against 50.7% implied over 242 picks.
        st = _stats(_rows(242, 0.434, 0.507))
        self.assertTrue(st["gated_on_own_record"])
        self.assertLessEqual(st["z"], og.Z_GATE)

    def test_small_sample_never_gates_however_bad(self):
        # One terrible fortnight is not a verdict.
        st = _stats(_rows(40, 0.20, 0.55))
        self.assertEqual(st["phase"], 0)
        self.assertFalse(st["gated_on_own_record"])

    def test_statistically_significant_but_trivial_deficit_does_not_gate(self):
        # z alone would fire here. Residual overround makes every market very
        # slightly negative, so a magnitude floor is what stops the governor
        # eventually gating everything.
        st = _stats(_rows(40000, 0.4995, 0.5055))
        self.assertLessEqual(st["z"], og.Z_GATE)
        self.assertGreater(st["deficit_pp"], og.MIN_DEFICIT_PP)
        self.assertFalse(st["gated_on_own_record"])

    def test_large_but_insignificant_deficit_does_not_gate(self):
        st = _stats(_rows(220, 0.47, 0.50))
        self.assertLessEqual(st["deficit_pp"], og.MIN_DEFICIT_PP)
        self.assertGreater(st["z"], og.Z_GATE)
        self.assertFalse(st["gated_on_own_record"])

    def test_beating_the_market_never_gates(self):
        st = _stats(_rows(1000, 0.56, 0.50))
        self.assertFalse(st["gated_on_own_record"])
        self.assertGreater(st["deficit_pp"], 0)


class TestMetric(unittest.TestCase):
    def test_deficit_is_measured_against_price_not_against_fifty_percent(self):
        # 45% wins is good on 35c underdogs and bad on 60c favourites. Raw win
        # rate cannot tell those apart; this metric must.
        good = _stats(_rows(1000, 0.45, 0.35))
        bad = _stats(_rows(1000, 0.45, 0.60))
        self.assertGreater(good["deficit_pp"], 0)
        self.assertLess(bad["deficit_pp"], 0)

    def test_pushes_and_pending_carry_no_information(self):
        for o in ("push", "cancel", None):
            self.assertFalse(og._usable({
                "sport": "X", "market_type": "M", "date": "2026-10-01",
                "outcome": o, "market_prob_at_first_pick": 0.5}))

    def test_a_pick_without_a_recorded_price_is_unusable(self):
        self.assertFalse(og._usable({
            "sport": "X", "market_type": "M", "date": "2026-10-01",
            "outcome": "win", "market_prob_at_first_pick": None}))

    def test_quarantined_rows_are_excluded(self):
        self.assertFalse(og._usable({
            "sport": "MLS", "market_type": "Total", "date": "2026-08-15",
            "outcome": "win", "market_prob_at_first_pick": 0.5}))


class TestExemptions(unittest.TestCase):
    def test_an_exemption_expires(self):
        self.assertTrue(og._is_exempt("NFL", og.SPORT_KEY, date(2026, 12, 1)))
        self.assertIsNone(og._is_exempt("NFL", og.SPORT_KEY, date(2026, 12, 8)))

    def test_a_sport_level_exemption_covers_its_markets(self):
        self.assertTrue(og._is_exempt("NFL", "Total", date(2026, 11, 1)))

    def test_exemptions_do_not_leak_to_other_sports(self):
        self.assertIsNone(og._is_exempt("NHL", "Total", date(2026, 11, 1)))

    def test_every_exemption_is_dated_and_explained(self):
        # An undated or unexplained exemption is a silently disabled governor.
        for e in og.EXEMPTIONS:
            date.fromisoformat(e["until"])
            self.assertTrue(e.get("reason", "").strip(), e)
            self.assertTrue(e.get("sport", "").strip(), e)


class TestFailOpen(unittest.TestCase):
    def test_gate_allows_on_junk_records(self):
        class R:
            pass
        for rec in (R(), None):
            allowed, _ = og.outcome_gate(rec)
            self.assertTrue(allowed)

    def test_summarise_never_raises(self):
        self.assertIsNone(og._summarise([]))
        self.assertIsNone(og._summarise([{"outcome": "win"}]))

    def test_gate_is_budget_only(self):
        # Display, watchlist and the logs must be untouched, or a gated market
        # stops being measured and can never recover.
        import inspect
        src = inspect.getsource(og)
        for forbidden in ("own_display", "watchlist_history", "record_picks"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
