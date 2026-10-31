"""The output-sanity detector: it must catch the known bugs and stay quiet otherwise."""

import unittest
from datetime import date

from src.state.model_sanity import evaluate, findings_for_log


# ⚠️ The default fixture must be a market with NO exclusion of any kind.
# Two have bitten already: MLS dates in 2026 sit inside the xGF quarantine
# window, and "NBA Total" is an ACKNOWLEDGED one-sided finding until Dec 1 —
# either default makes these assertions pass vacuously. NBA Spread is neither.
def _row(sport="NBA", market="Spread", day="2026-10-28", side="under",
         raw=0.90, mkt=0.51, cap=False):
    return {"sport": sport, "market_type": market, "date": day,
            "pick_side": side, "model_prob_raw": raw,
            "market_prob_at_first_pick": mkt, "credibility_cap_fired": cap}


def _codes(res, sport=None):
    return {f["code"] for f in res["findings"] if sport is None or f["sport"] == sport}


class TestCatchesTheKnownBugs(unittest.TestCase):
    """Each assertion below corresponds to a bug that really shipped."""

    def test_one_sided_market(self):
        # MLS totals ran 100% Under for five months and nothing noticed.
        rows = [_row(side="under", raw=0.55, cap=False) for _ in range(20)]
        self.assertIn("one_sided", _codes(evaluate(rows, today=date(2026, 10, 29))))

    def test_cap_pinned_market(self):
        rows = [_row(side="under" if i % 2 else "over", raw=0.60, cap=True)
                for i in range(20)]
        self.assertIn("cap_pinned", _codes(evaluate(rows, today=date(2026, 10, 29))))

    def test_extreme_raw_gap(self):
        # The MLS gap was 39pp. Read model_prob_raw, never the post-cap number.
        rows = [_row(side="under" if i % 2 else "over", raw=0.92, mkt=0.50)
                for i in range(20)]
        self.assertIn("raw_gap_extreme", _codes(evaluate(rows, today=date(2026, 10, 29))))

    def test_a_market_that_went_quiet(self):
        # MLS emitted ZERO moneylines while draws and totals kept coming.
        old = [_row(market="Moneyline", day="2026-08-01") for _ in range(5)]
        new = [_row(market="Total", day="2026-10-28") for _ in range(5)]
        res = evaluate(old + new, today=date(2026, 10, 29))
        self.assertIn("market_went_quiet", _codes(res))

    def test_high_severity_sets_ok_false(self):
        rows = [_row(side="under", raw=0.55) for _ in range(20)]
        self.assertFalse(evaluate(rows, today=date(2026, 10, 29))["ok"])


class TestStaysQuiet(unittest.TestCase):
    """A detector that cries wolf is one the reader learns to skip — which is
    exactly how the earlier silent failures survived for months."""

    def test_a_healthy_two_sided_market_is_silent(self):
        rows = [_row(sport="MLB", market="Total",
                     side="under" if i % 2 else "over",
                     raw=0.53 + (i % 5) * 0.01, mkt=0.50,
                     cap=(i % 10 == 0))
                for i in range(40)]
        res = evaluate(rows, today=date(2026, 10, 29))
        self.assertEqual(res["findings"], [], res["findings"])
        self.assertTrue(res["ok"])

    def test_small_samples_do_not_trip_the_absolute_floors(self):
        # Below the minimum n, a 100% rate is noise, not a signal.
        rows = [_row(side="under", raw=0.95) for _ in range(5)]
        self.assertEqual(evaluate(rows, today=date(2026, 10, 29))["findings"], [])

    def test_an_out_of_season_sport_is_not_flagged_as_quiet(self):
        # Every market gone = out of season. Only a market whose SIBLINGS are
        # still active is suspicious.
        rows = [_row(sport="WNBA", market=m, day="2026-08-01")
                for m in ("Moneyline", "Total") for _ in range(5)]
        self.assertNotIn("market_went_quiet", _codes(evaluate(rows, today=date(2026, 10, 29))))

    def test_quarantined_rows_are_not_re_flagged(self):
        # The MLS window is already recorded as broken, with a cause and a fix.
        # Re-reporting it daily is the noise this check must not generate.
        rows = [_row(sport="MLS", market="Total", day="2026-08-15",
                     side="under", raw=0.95) for _ in range(30)]
        res = evaluate(rows, today=date(2026, 8, 20))
        self.assertEqual(res["findings"], [], res["findings"])


class TestSafety(unittest.TestCase):
    def test_never_raises_on_junk(self):
        for junk in ([], None, [{}], [{"sport": "MLB"}], [{"date": "x"}],
                     [{"sport": "MLB", "date": "2026-10-28", "model_prob_raw": "nope"}]):
            res = evaluate(junk, today=date(2026, 10, 29))
            self.assertIn("findings", res)

    def test_detect_only_never_gates(self):
        # If this module ever imports the budget path, the separation of
        # detection from money routing has been lost.
        import inspect
        import src.state.model_sanity as ms
        src = inspect.getsource(ms)
        for forbidden in ("clv_gate", "enters_budget", "BUDGET_EXCLUDED"):
            self.assertNotIn(forbidden, src)

    def test_log_lines_are_rendered(self):
        rows = [_row(side="under", raw=0.55) for _ in range(20)]
        lines = findings_for_log(evaluate(rows, today=date(2026, 10, 29)))
        self.assertTrue(lines)
        self.assertTrue(all("MODEL SANITY" in l for l in lines))


if __name__ == "__main__":
    unittest.main()


class TestAcknowledgedFindings(unittest.TestCase):
    """An investigated finding stops alerting but stays visible.

    A check that keeps firing on a condition somebody has already explained
    teaches the reader to skip it — which is how the MLS bug survived five
    months of weekly reports that all mentioned it.
    """

    @staticmethod
    def _rows_near(today, n=30, **kw):
        """Rows a few days before `today`, so they land in the recent window.

        RECENT_DAYS is 14 — the first draft of these tests used a fixed October
        date against a November `today`, so every row fell outside the window
        and the assertions passed on an empty finding list.
        """
        from datetime import timedelta
        day = (today - timedelta(days=3)).isoformat()
        opts = {"sport": "NBA", "market": "Total", "side": "under",
                "raw": 0.55, "mkt": 0.50}
        opts.update(kw)
        return [_row(day=day, **opts) for _ in range(n)]

    def _nba_totals(self, today, n=30):
        return self._rows_near(today, n)

    def test_an_acknowledged_finding_is_kept_not_dropped(self):
        t = date(2026, 11, 15)
        res = evaluate(self._nba_totals(t), today=t)
        one = [f for f in res["findings"] if f["code"] == "one_sided"]
        self.assertTrue(one, "the finding must still be reported")
        self.assertEqual(one[0]["acknowledged_until"], "2026-12-01")

    def test_an_acknowledged_high_does_not_set_ok_false(self):
        t = date(2026, 11, 15)
        self.assertTrue(evaluate(self._nba_totals(t), today=t)["ok"])

    def test_the_acknowledgement_expires(self):
        t = date(2026, 12, 2)
        res = evaluate(self._nba_totals(t), today=t)
        one = [f for f in res["findings"] if f["code"] == "one_sided"]
        self.assertTrue(one)
        self.assertIsNone(one[0].get("acknowledged_until"))
        self.assertFalse(res["ok"], "after expiry it must alert again")

    def test_it_does_not_cover_a_different_market(self):
        t = date(2026, 11, 15)
        rows = self._rows_near(t, market="Spread")
        res = evaluate(rows, today=t)
        one = [f for f in res["findings"] if f["code"] == "one_sided"]
        self.assertTrue(one)
        self.assertIsNone(one[0].get("acknowledged_until"))

    def test_it_does_not_cover_a_different_finding_on_the_same_market(self):
        # NBA Total is acknowledged for one-sidedness only. A sudden extreme
        # raw gap there is new information and must still alert.
        t = date(2026, 11, 15)
        from datetime import timedelta
        day = (t - timedelta(days=3)).isoformat()
        rows = [_row(sport="NBA", market="Total", day=day,
                     side="under" if i % 2 else "over", raw=0.95, mkt=0.50)
                for i in range(30)]
        res = evaluate(rows, today=t)
        gaps = [f for f in res["findings"] if f["code"] == "raw_gap_extreme"]
        self.assertTrue(gaps)
        self.assertIsNone(gaps[0].get("acknowledged_until"))
        self.assertFalse(res["ok"])

    def test_every_acknowledgement_is_dated_and_explained(self):
        from src.state.model_sanity import ACKNOWLEDGED
        for a in ACKNOWLEDGED:
            date.fromisoformat(a["until"])
            self.assertTrue(a.get("reason", "").strip(), a)
            self.assertTrue(a.get("code", "").strip(), a)

    def test_log_lines_mark_acknowledged_findings(self):
        t = date(2026, 11, 15)
        lines = findings_for_log(evaluate(self._nba_totals(t), today=t))
        self.assertTrue(any("acknowledged until" in l for l in lines))
