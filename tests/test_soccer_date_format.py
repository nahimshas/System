"""ESPN's soccer scoreboards reject a date RANGE (Sep 19 2026).

User report: "the liga mx games have no team logos in the cards".

`?dates=20260919-20260920` returns HTTP 400 for usa.1, mex.1 AND fifa.world,
even though the US sports accept exactly that form. Both soccer call sites used
it, so every soccer fetch failed outright:

  * PWA      -> no logos, no live scores, no in-page settlement (the reported
                symptom; the .catch() reports zero events, which is
                indistinguishable from a day with no matches)
  * settler  -> _fetch_watchlist_final_scores returned {} for MLS and WC, so
                those picks could never settle. LigaMX escaped only because it
                was never added to the ("MLS","WC") tuple and used a single date.

Fix: two SINGLE-DATE calls, merged — keeping the reason the window existed
(ESPN dates soccer by UTC, so an evening-Pacific kickoff lands on tomorrow).
"""
import re

import pytest


def _js():
    with open("src/report/templates/_report_js.html") as f:
        return f.read()


def _oc():
    with open("src/data/outcome_checker.py") as f:
        return f.read()


class TestPwaUrls:
    def test_no_range_format_anywhere(self):
        """`dates=A-B` is the exact thing ESPN rejects for soccer."""
        js = _js()
        assert "_soccerDates" not in js
        assert not re.search(r'dates="\s*\+\s*_espnDate\s*\+\s*"-"', js)

    def test_soccer_uses_a_two_url_helper(self):
        js = _js()
        assert "function _soccerUrls(" in js
        for league in ("usa.1", "fifa.world", "mex.1"):
            assert f'_soccerUrls("{league}")' in js

    def test_helper_returns_two_single_dates(self):
        js = _js()
        body = js.split("function _soccerUrls(")[1].split("}")[0]
        assert "_espnDate" in body and "_espnDateNext" in body
        assert "-" not in body.split("return")[1].replace("scoreboard?dates=", "")

    def test_sources_flatten_array_values(self):
        js = _js()
        i = js.index("var SCOREBOARD_SOURCES")
        block = js[i:i + 600]
        assert "Array" in block          # array-valued entries are expanded
        assert "Object.keys(ESPN)" in block

    def test_non_soccer_sports_unchanged(self):
        js = _js()
        for sport in ("NBA", "MLB", "NHL", "NFL", "WNBA"):
            assert re.search(rf'{sport}: "https://site\.api\.espn\.com', js)


class TestSettlementFetch:
    def test_no_range_in_the_settler(self):
        assert "strftime('%Y%m%d')}-{" not in _oc()

    def test_soccer_set_covers_all_three(self):
        src = _oc()
        assert "_SOCCER_WATCHLIST" in src
        m = re.search(r'_SOCCER_WATCHLIST\s*=\s*\{([^}]*)\}', src)
        assert m
        for sport in ("MLS", "WC", "LIGAMX"):
            assert f'"{sport}"' in m.group(1)

    def test_ligamx_is_included_now(self):
        """It was omitted before, which is the only reason it still worked."""
        src = _oc()
        m = re.search(r'_SOCCER_WATCHLIST\s*=\s*\{([^}]*)\}', src)
        assert '"LIGAMX"' in m.group(1)

    def test_fetch_loops_over_dates(self):
        src = _oc()
        body = src.split("def _fetch_watchlist_final_scores")[1].split("\ndef ")[0]
        assert "for _d in dates:" in body
        assert 'params={"dates": _d}' in body

    def test_a_failed_day_does_not_lose_the_other(self):
        """One bad date must not discard the events already collected."""
        src = _oc()
        body = src.split("def _fetch_watchlist_final_scores")[1].split("\ndef ")[0]
        # the except sits INSIDE the loop and does not return
        after = body.split("for _d in dates:")[1].split("data = {")[0]
        assert "except Exception" in after
        assert "return {}" not in after


class TestLiveEndpoints:
    """Hits ESPN. Documents the actual API behaviour this fix is built on."""

    @pytest.mark.parametrize("league", ["mex.1", "usa.1", "fifa.world"])
    def test_single_date_ok_range_rejected(self, league):
        requests = pytest.importorskip("requests")
        base = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
        try:
            ok = requests.get(base, params={"dates": "20260919"}, timeout=15)
            bad = requests.get(base, params={"dates": "20260919-20260920"}, timeout=15)
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")
        assert ok.status_code == 200
        assert bad.status_code == 400, "ESPN started accepting ranges — revisit the fix"
