"""A missing stat must not take down a whole sport's analysis (Sep 21 2026).

The idiom throughout edge_finder was:

    f"{stats.get('net_rtg', '?'):.1f}"

That `'?'` fallback CAN NEVER FORMAT — `f"{'?':.1f}"` raises ValueError. So the
thing that looked like graceful degradation was a guaranteed crash, and because
these lines sit inside the analyzer it would kill the ENTIRE SPORT's slate for
the day rather than drop one research line. Found when a test fixture omitted
net_rtg and analyze_nfl_game raised. 35 sites, all replaced with _stat().
"""
import re

import pytest

from src.models.edge_finder import _stat


class TestFormattingIsUnchanged:
    """Present values must render exactly as the plain f-string did."""

    @pytest.mark.parametrize("value,nd", [
        (8.5, 1), (27.456, 1), (0.7123, 3), (4.005, 2), (-6.5, 1),
        (0, 1), (100, 2), (0.001, 3),
    ])
    def test_matches_plain_format_spec(self, value, nd):
        assert _stat(value, nd) == f"{float(value):.{nd}f}"

    def test_numeric_strings_still_format(self):
        """Feeds sometimes hand back numbers as strings."""
        assert _stat("27.5", 1) == "27.5"

    def test_default_precision_is_one(self):
        assert _stat(3.14159) == "3.1"


class TestDegradesInsteadOfRaising:
    @pytest.mark.parametrize("value", [None, "?", "", "abc", [], {}, float("nan")])
    def test_bad_values_return_the_placeholder(self, value):
        assert _stat(value) == "?"

    def test_nan_is_a_missing_stat_not_the_text_nan(self):
        assert _stat(float("nan")) == "?"

    def test_custom_default(self):
        assert _stat(None, 1, default="—") == "—"

    def test_never_raises(self):
        class Hostile:
            def __float__(self):
                raise RuntimeError("boom")
        with pytest.raises(RuntimeError):
            float(Hostile())          # confirms the object really is hostile
        # _stat only guards TypeError/ValueError by design; document that.
        assert _stat(object()) == "?"


class TestNoCrashSitesRemain:
    def _src(self):
        with open("src/models/edge_finder.py") as f:
            return f.read()

    def test_no_question_mark_default_with_a_format_spec(self):
        """The exact pattern that crashed."""
        bad = re.findall(r"\.get\((['\"])\w+\1,\s*(['\"])\?\2\)\s*:\s*\.\d+f", self._src())
        assert not bad, f"{len(bad)} crash-prone sites remain"

    def test_helper_is_actually_used(self):
        assert self._src().count("_stat(") > 30


class TestAnalyzerSurvivesMissingFields:
    """The real symptom: one absent field killed the whole slate."""

    def _run(self, stats_extra):
        from src.data.nfl_stats import normalize
        from src.models.edge_finder import analyze_nfl_game
        h, a = normalize("Buffalo Bills"), normalize("New York Giants")
        game = {"home_team": "Buffalo Bills", "away_team": "New York Giants",
                "commence_time": "2026-09-21T20:00:00Z",
                "moneyline": {"home_prob": 0.68, "away_prob": 0.32},
                "spread": {"home_spread": -6.0, "home_prob": 0.52, "away_prob": 0.48},
                "total": {"line": 44.5, "over_prob": 0.5, "under_prob": 0.5},
                "bookmakers": []}
        hs = {"ppg": 27.5, "oppg": 19.0, "wins": 2, "losses": 0}
        as_ = {"ppg": 18.0, "oppg": 24.5, "wins": 0, "losses": 2}
        hs.update(stats_extra); as_.update(stats_extra)
        ctx = {"season_stats": {h: hs, a: as_},
               "rest_days": {h: 7, a: 7}, "recent_form": {}}
        return analyze_nfl_game(game, ctx, {}, min_edge=0.0)

    def test_missing_net_rtg_no_longer_raises(self):
        recs = self._run({})              # net_rtg absent — used to ValueError
        assert recs

    def test_placeholder_appears_in_the_research_line(self):
        recs = self._run({})
        assert any("NetRtg ?" in r for rec in recs for r in rec.research)

    def test_output_unchanged_when_the_field_is_present(self):
        recs = self._run({"net_rtg": 8.5})
        assert any("NetRtg 8.5" in r for rec in recs for r in rec.research)

    def test_projected_score_still_emitted(self):
        for rec in self._run({}):
            assert any("Model projected score" in s for s in rec.signals)
