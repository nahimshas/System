"""Kalshi renamed several books in Oct 2026 and matching went dark.

Caught by the per-market CLV alert, which is exactly what it was built for:
NFL Spread fell to 7.1% (1 of 14) while the aggregate still read 92.6%.

Two separate renames:
  * NFL spread titles changed from the city-only form the MONEYLINE book still
    uses ("Los Angeles R") to an abbreviation+nickname form ("LA Chargers").
  * All three MLB F5 books dropped their "first 5 innings" suffixes and now
    read exactly like the full-game books ("Atlanta", "Atlanta wins by over 1.5
    runs", "Over 0.5 runs").

Prose has now broken matching three times (Sep 11 cross-series inconsistency,
Sep 19 soccer's "more than" verb, and this). The durable fix is to lean on the
TICKER, which has survived every rename.
"""
import re

import pytest


def _src():
    with open("src/data/kalshi.py") as f:
        return f.read()


class TestStalePromptsAreGone:
    def _code_only(self):
        """Source with comments stripped — the phrase survives in comments that
        EXPLAIN the rename, and matching on those is a false positive."""
        return "\n".join(l.split("#")[0] for l in _src().split("\n"))

    def test_no_first_5_innings_match_string_remains(self):
        """The F5 books no longer carry that suffix, so no want-string may."""
        assert "first 5 innings" not in self._code_only()

    def test_f5_total_shares_the_full_game_prefix(self):
        assert "runs in the first 5" not in self._code_only()


class TestSpreadMatchesOnTheTicker:
    def test_abbrev_helper_exists(self):
        src = _src()
        assert "def _abbrevs_for_game(" in src

    def test_abbrevs_come_from_the_moneyline_book(self):
        """The ML book is the one whose titles we can still map."""
        body = _src().split("def _abbrevs_for_game(")[1].split("\ndef ")[0]
        assert "anchor_series" in body
        assert 'rsplit("-", 1)[1]' in body

    def test_spread_prefers_the_ticker_suffix(self):
        body = _src().split("def _select(")[1].split("\ndef ")[0]
        assert "startswith(_ab)" in body

    def test_line_number_is_still_read_from_the_title(self):
        """Team from the ticker, number from the text."""
        body = _src().split("def _select(")[1].split("\ndef ")[0]
        assert r"\bover\s+" in body


class TestLiveFormats:
    """Hits Kalshi. These assert the CURRENT prose, so the next rename fails
    loudly here instead of silently zeroing a market's CLV."""

    @pytest.fixture(scope="class")
    def idx(self):
        pytest.importorskip("requests")
        from src.data.kalshi_clv import SPORT_SERIES, build_index
        try:
            return build_index(sorted(set(SPORT_SERIES["MLB"].values())),
                               {"2026-10-04"})
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")

    def test_f5_winner_is_titled_with_a_bare_team(self, idx):
        titles = {str(m.get("yes_sub_title")) for m in idx.get("KXMLBF5", [])}
        if not titles:
            pytest.skip("no F5 markets listed for that date")
        assert not any("first 5" in t.lower() for t in titles)

    def test_f5_spread_reads_like_the_full_game_book(self, idx):
        titles = {str(m.get("yes_sub_title")) for m in idx.get("KXMLBF5SPREAD", [])}
        if not titles:
            pytest.skip("no F5 spread markets listed")
        assert any(re.search(r"wins by over [\d.]+ runs", t) for t in titles)

    def test_nfl_spread_and_moneyline_name_teams_DIFFERENTLY(self, idx):
        """Documents the inconsistency the ticker join exists to survive."""
        from src.data.kalshi_clv import SPORT_SERIES, build_index
        try:
            n = build_index(sorted(set(SPORT_SERIES["NFL"].values())), {"2026-10-04"})
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")
        ml = {str(m.get("yes_sub_title")) for m in n.get("KXNFLGAME", [])}
        sp = {str(m.get("yes_sub_title")) for m in n.get("KXNFLSPREAD", [])}
        if not ml or not sp:
            pytest.skip("no NFL markets listed for that date")
        # The ML book uses city-only tokens; if that ever stops being true the
        # anchor itself is broken and every sport is affected.
        assert any(t in ml for t in ("Philadelphia", "Los Angeles R", "Seattle"))
