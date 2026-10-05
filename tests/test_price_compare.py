"""Kalshi-vs-OddsAPI pricing comparison (Oct 5 2026).

Deliberately a REPORT over existing data, not a new collection path: every
decision-log row already carries both prices for the same moment
(market_prob_at_first_pick = Odds API no-vig, kalshi_prob_at_pick = Kalshi mid
stamped by the CLV pass). 9,698 rows had both when this was written, so a
parallel logger would have added a failure surface and started from zero.
"""
import pytest

from tools.analysis.price_compare import analyse


class TestReportShape:
    def test_runs_over_real_data(self):
        rep = analyse(since="2026-08-01")
        assert rep["n"] > 100
        assert rep["markets"]

    def test_every_market_has_the_expected_fields(self):
        rep = analyse(since="2026-08-01")
        for m in rep["markets"].values():
            for f in ("n", "mean_delta_pp", "mean_abs_delta_pp",
                      "over_2pp_pct", "flips_in", "flips_out", "flip_pct"):
                assert f in m

    def test_empty_window_does_not_crash(self):
        rep = analyse(since="2099-01-01")
        assert rep["n"] == 0
        assert rep["markets"] == {}

    def test_min_edge_changes_the_flip_count(self):
        lo = analyse(since="2026-08-01", min_edge=0.02)
        hi = analyse(since="2026-08-01", min_edge=0.20)
        assert lo["overall"]["flip_pct"] != hi["overall"]["flip_pct"]


class TestKnownFinding:
    """These pin the Oct 2026 discovery so a silent regression is visible."""

    def test_moneylines_agree_closely(self):
        rep = analyse(since="2026-08-01")
        ml = rep["markets"].get("MLB Moneyline")
        assert ml and ml["n"] > 200
        assert abs(ml["mean_delta_pp"]) < 0.5, "MLB ML should agree within half a point"

    def test_totals_diverge_because_oddsapi_totals_keep_the_vig(self):
        """Not a venue difference — see test_totals_vig below."""
        rep = analyse(since="2026-08-01")
        t = rep["markets"].get("MLB Total")
        assert t and t["n"] > 200
        assert t["mean_delta_pp"] < -1.0


class TestTotalsVig:
    """Both sides of a total priced lower on Kalshi is impossible for one
    binary market — they must sum to 1. The Odds API side sums to ~1.05."""

    def _pairs(self, market_type, sport):
        import glob
        import json
        from collections import defaultdict
        byg = defaultdict(list)
        for f in glob.glob("state/decision_log/2026-*.json"):
            d = json.load(open(f))
            rows = d.get("entries", d)
            rows = list(rows.values()) if isinstance(rows, dict) else rows
            for r in rows:
                if not isinstance(r, dict):
                    continue
                if r.get("sport") != sport or r.get("market_type") != market_type:
                    continue
                p = r.get("market_prob_at_first_pick")
                if p is None:
                    continue
                byg[(r.get("game"), r.get("date"), r.get("line"))].append(p)
        return [v for v in byg.values() if len(v) == 2]

    def test_moneyline_sides_sum_to_one(self):
        pairs = self._pairs("Moneyline", "MLB")
        assert len(pairs) > 100
        mean = sum(sum(p) for p in pairs) / len(pairs)
        assert abs(mean - 1.0) < 0.005, f"MLB ML should be de-vigged, got {mean:.4f}"

    def test_totals_sides_do_NOT_sum_to_one(self):
        """Documents the bug: totals keep ~4.6% vig. If this ever starts
        passing at 1.0, the de-vig fix shipped and this test should be
        inverted along with it."""
        pairs = self._pairs("Total", "MLB")
        assert len(pairs) > 100
        mean = sum(sum(p) for p in pairs) / len(pairs)
        assert mean > 1.02, f"expected vig-inflated totals, got {mean:.4f}"
