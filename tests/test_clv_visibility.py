"""Visibility for silently-dark CLV feeds (Sep 11 2026).

Two failures were invisible for their whole lifetime:
  * NFL spread/total CLV never resolved -- the aggregate coverage number is
    dominated by MLB (~3,500 rows vs NFL's 4), so it never moved.
  * A pick can quote a line Kalshi does not list. That returned the same silent
    None as "game not found", so "unbuyable" and "unlisted" were indistinguishable.
"""
import logging

import pytest

from src.data.kalshi import _ladder_points, resolve_pick

SERIES_MAP = {"Moneyline": "KXNFLGAME", "Spread": "KXNFLSPREAD",
              "Total": "KXNFLTOTAL"}


def _m(ticker, event, sub, date="2026-09-10"):
    return {"ticker": ticker, "event_ticker": event, "yes_sub_title": sub,
            "rules_primary": "", "expiration_time": f"{date}T23:59:00Z",
            "close_time": f"{date}T23:59:00Z"}


def _markets():
    """A real NFL ladder shape: 1.5-7.5 complete, then 8.5 MISSING."""
    ev = "KXNFLSPREAD-26SEP10SFLAR"
    rungs = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 9.5, 10.5, 13.5]
    return {
        "KXNFLGAME": [
            _m("KXNFLGAME-26SEP10SFLAR-SF", "KXNFLGAME-26SEP10SFLAR", "San Francisco"),
            _m("KXNFLGAME-26SEP10SFLAR-LAR", "KXNFLGAME-26SEP10SFLAR", "Los Angeles R"),
        ],
        "KXNFLSPREAD": [
            _m(f"KXNFLSPREAD-26SEP10SFLAR-SF{int(r)}", ev,
               f"San Francisco wins by over {r} points") for r in rungs
        ],
    }


class TestLadderPoints:
    def test_reads_listed_strikes(self):
        pts = _ladder_points(_markets()["KXNFLSPREAD"])
        assert pts[:4] == [1.5, 2.5, 3.5, 4.5]
        assert 8.5 not in pts          # the hole
        assert 9.5 in pts

    def test_empty_pool_is_not_fatal(self):
        assert _ladder_points([]) == []

    def test_ignores_untitled_markets(self):
        assert _ladder_points([{"yes_sub_title": None}]) == []


class TestUnbuyableWarning:
    """A line the exchange will not sell must be loud, not silent."""

    def _resolve(self, pick):
        return resolve_pick(
            {"bet_type": "Spread", "pick": pick,
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)

    def test_missing_rung_warns_and_names_neighbours(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.data.kalshi"):
            assert self._resolve("San Francisco 49ers -8.5") is None
        msg = caplog.text
        assert "NOT BUYABLE" in msg
        assert "8.5" in msg
        assert "7.5" in msg and "9.5" in msg     # the nearest listed rungs

    def test_whole_number_warns(self, caplog):
        """Kalshi is binary — no whole-number spread exists anywhere."""
        with caplog.at_level(logging.WARNING, logger="src.data.kalshi"):
            assert self._resolve("San Francisco 49ers -3.0") is None
        assert "NOT BUYABLE" in caplog.text

    def test_listed_line_is_silent(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.data.kalshi"):
            r = self._resolve("San Francisco 49ers -3.5")
        assert r is not None
        assert "NOT BUYABLE" not in caplog.text

    def test_unknown_game_does_not_claim_unbuyable(self, caplog):
        """Game not found is a DIFFERENT failure — must not be mislabelled."""
        with caplog.at_level(logging.WARNING, logger="src.data.kalshi"):
            r = resolve_pick(
                {"bet_type": "Spread", "pick": "Chicago Bears -3.5",
                 "home_team": "Chicago Bears", "away_team": "Green Bay Packers"},
                _markets(), "2026-09-10", require_quote=False,
                series_map=SERIES_MAP)
        assert r is None
        assert "NOT BUYABLE" not in caplog.text


class TestPerMarketCoverage:
    """One dark market must not hide behind the aggregate."""

    def test_thresholds_are_sane(self):
        from tools.analysis import health_report as H
        assert H.MARKET_CLV_MIN_N >= 3          # tiny n is noise
        assert 0 < H.MARKET_CLV_COVERAGE_MIN <= 100

    def test_alert_fires_on_a_wired_dark_market(self):
        from tools.analysis import health_report as H
        report = {
            "bankroll": {"ok": True}, "budget_performance": {},
            "log_liveness": {"ok": True}, "governors": {},
            "subsystem_liveness": {
                "ok": True, "execution_rows_recent": 5,
                "kalshi_clv_coverage_pct": 84.4,
                "kalshi_clv_by_market": {
                    "MLB Spread": {"n": 145, "covered": 145, "pct": 100.0, "wired": True},
                    "CFB Spread": {"n": 7, "covered": 3, "pct": 42.9, "wired": True},
                },
            },
        }
        alerts = H.compute_alerts(report)
        assert any("CFB Spread" in a for a in alerts)
        assert not any("MLB Spread" in a for a in alerts)

    def test_unwired_market_does_not_alert(self):
        """WNBA/MLS have no Kalshi team map — alerting trains the reader to ignore."""
        from tools.analysis import health_report as H
        report = {
            "bankroll": {"ok": True}, "budget_performance": {},
            "log_liveness": {"ok": True}, "governors": {},
            "subsystem_liveness": {
                "ok": True, "execution_rows_recent": 5,
                "kalshi_clv_coverage_pct": 84.4,
                "kalshi_clv_by_market": {
                    "MLS Total": {"n": 43, "covered": 0, "pct": 0.0, "wired": False},
                },
            },
        }
        assert not any("MLS Total" in a for a in H.compute_alerts(report))

    def test_small_sample_does_not_alert(self):
        from tools.analysis import health_report as H
        report = {
            "bankroll": {"ok": True}, "budget_performance": {},
            "log_liveness": {"ok": True}, "governors": {},
            "subsystem_liveness": {
                "ok": True, "execution_rows_recent": 5,
                "kalshi_clv_coverage_pct": 84.4,
                "kalshi_clv_by_market": {
                    "NFL Spread": {"n": 1, "covered": 0, "pct": 0.0, "wired": True},
                },
            },
        }
        assert not any("NFL Spread" in a for a in H.compute_alerts(report))
