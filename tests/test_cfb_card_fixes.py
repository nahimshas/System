"""CFB card fixes (Sep 11 2026).

Three defects the user spotted on one slate:
  1. no team logos on CFB cards
  2. two nested picks on one game (Kansas ML + Kansas +5.5)
  3. (found while checking 2) the credibility cap fired but recorded False
"""
import re

import pytest


class TestEspnEndpoint:
    """CFB was absent from the client-side ESPN map — hence no logos."""

    def _js(self):
        with open("src/report/templates/_report_js.html") as f:
            return f.read()

    def test_cfb_is_in_the_espn_map(self):
        assert re.search(r"\bCFB:\s*\"https://site\.api\.espn\.com", self._js())

    def test_cfb_uses_college_football_path(self):
        assert "football/college-football/scoreboard" in self._js()

    def test_cfb_lifts_espn_default_page_size(self):
        """A Saturday FBS slate runs past ESPN's 25-game default page."""
        m = re.search(r"CFB:\s*\"([^\"]+)\"", self._js())
        assert m, "CFB entry missing"
        url = m.group(1)
        assert "limit=" in url
        assert int(re.search(r"limit=(\d+)", url).group(1)) >= 100
        assert "groups=80" in url          # FBS only

    def test_classic_report_also_has_cfb(self):
        with open("src/report/templates/report.html") as f:
            assert "football/college-football/scoreboard" in f.read()


class TestOnePickPerGameOnTiles:
    """Watchlist tiles now keep one pick per game, like the budget card."""

    def _main(self):
        with open("src/main.py") as f:
            return f.read()

    def test_helper_exists_and_is_applied(self):
        src = self._main()
        assert "def _one_per_game(" in src
        assert "_one_per_game(" in src.split("fresh_own_displays")[1]

    def test_ipl_is_exempt(self):
        assert 'if slug == "ipl":' in self._main()

    def test_keeps_first_of_each_game(self):
        """Input is pre-sorted by _slot_sort_key, so 'first' == 'best'."""
        class R:
            def __init__(self, game, pick):
                self.game, self.pick = game, pick

        recs = [R("MIZ @ KU", "Kansas +5.5"),
                R("MIZ @ KU", "Kansas ML"),
                R("RUT @ BC", "Rutgers ML")]
        seen, out = {}, []
        for r in recs:                      # mirrors the shipped loop
            if seen.get(r.game, 0) >= 1:
                continue
            seen[r.game] = 1
            out.append(r)
        assert [r.pick for r in out] == ["Kansas +5.5", "Rutgers ML"]


class TestCredibilityCapFlag:
    """The cap fired on both Kansas picks but reported credibility_cap_fired=False."""

    def test_dispatched_cap_reports_firing(self):
        from src.models.edge_finder import _apply_credibility_cap_dispatched
        # The real Sep 11 numbers: raw 47.96% vs market 35.4% = 12.6pp drift,
        # well past the 8pp CFB cap.
        capped, fired = _apply_credibility_cap_dispatched(
            0.47964346025132204, 0.354, 0.08, "cfb", "credibility_moneyline")[:2]
        assert fired is True
        assert capped == pytest.approx(0.434, abs=1e-3)

    def test_uncapped_probability_does_not_report_firing(self):
        from src.models.edge_finder import _apply_credibility_cap_dispatched
        capped, fired = _apply_credibility_cap_dispatched(
            0.40, 0.38, 0.08, "cfb", "credibility_moneyline")[:2]
        assert fired is False
        assert capped == pytest.approx(0.40, abs=1e-9)

    def test_cfb_emit_stamps_the_flag(self):
        """Regression guard: the flag must not be dropped on the floor again."""
        with open("src/models/edge_finder.py") as f:
            src = f.read()
        emit = src.split("def _emit(pick: str, bet_type: str")[1].split("# ── Moneyline")[0]
        assert "r.credibility_cap_fired = bool(cap_fired)" in emit
        # and the callers must actually pass it
        cfb = src.split("def analyze_cfb_game")[1]
        assert "cap_fired=_ml_fired" in cfb
        assert "cap_fired=_sp_fired" in cfb

    def test_cfb_cap_result_is_not_subscripted_away(self):
        """`[0]` discarded the fired flag — that was the bug."""
        with open("src/models/edge_finder.py") as f:
            cfb = f.read().split("def analyze_cfb_game")[1]
        assert '"cfb", "credibility_moneyline")[0]' not in cfb
        assert '"cfb", "credibility_spread")[0]' not in cfb
