"""The card must never contradict its own pick (Sep 17 2026).

User report: "the lions were selected but the projected score is the bills to
win by one, this contradicts itself."

Root cause: the projected score was built from season scoring averages alone
(`(home.ppg + away.oppg)/2`), while the win probability came from a completely
different path — team strength + home field + injuries + rest + caps. The two
were free to disagree, and did. Measured across live cards:

    NFL   mean 6.2 pts off, worst 27.7  (Sep 10 SF@LAR projected "LAR by 24"
                                         while the model believed SF by 3.7 —
                                         the SIGN was flipped)
    WNBA  mean 4.9 pts off, worst  5.7

Fix: the TOTAL still comes from scoring data; the MARGIN is derived from the
model's own probability, so they agree by construction.
"""
import re

import pytest
from scipy.stats import norm

from src.models.edge_finder import _score_from_prob, _value_dog_note


class TestScoreFromProb:
    @pytest.mark.parametrize("p,std", [
        (0.602, 14.0), (0.398, 14.0), (0.50, 14.0),
        (0.75, 12.0), (0.25, 12.0), (0.55, 2.4),
    ])
    def test_margin_round_trips_to_the_probability(self, p, std):
        """The whole point: reading the margin back gives the same probability."""
        h, a = _score_from_prob(50.0, p, std)
        assert norm.cdf((h - a) / std) == pytest.approx(p, abs=1e-6)

    def test_total_is_preserved(self):
        h, a = _score_from_prob(47.0, 0.61, 14.0)
        assert h + a == pytest.approx(47.0)

    def test_favourite_is_shown_ahead(self):
        h, a = _score_from_prob(50.0, 0.65, 14.0)
        assert h > a

    def test_underdog_is_shown_behind(self):
        h, a = _score_from_prob(50.0, 0.35, 14.0)
        assert h < a

    def test_pickem_is_level(self):
        h, a = _score_from_prob(50.0, 0.50, 14.0)
        assert h == pytest.approx(a)

    def test_extremes_do_not_blow_up(self):
        for p in (0.0, 1.0, -5.0, 99.0):
            h, a = _score_from_prob(50.0, p, 14.0)
            assert all(x == x for x in (h, a))      # not NaN
            assert abs(h) < 1e4 and abs(a) < 1e4

    def test_todays_real_game(self):
        """Detroit @ Buffalo: model P(Buffalo)=60.2%, ppg total 51."""
        h, a = _score_from_prob(51.0, 0.602, 14.0)
        assert round(h) == 27 and round(a) == 24          # was 26-25
        assert norm.cdf((h - a) / 14.0) == pytest.approx(0.602, abs=1e-6)

    def test_the_sign_flip_case(self):
        """Sep 10 SF@LAR: card said LAR by 24, model believed SF by 3.7."""
        p_lar = float(norm.cdf(-3.7 / 14.0))
        h, a = _score_from_prob(45.0, p_lar, 14.0)
        assert h < a                                       # LAR now behind
        assert (h - a) == pytest.approx(-3.7, abs=0.05)


class TestValueDogNote:
    def test_fires_for_an_underdog_pick(self):
        note = _value_dog_note("Detroit Lions", 0.398, 0.321)
        assert note is not None
        assert "not a predicted win" in note
        assert "39.8%" in note and "32.1%" in note

    def test_silent_for_a_favourite(self):
        assert _value_dog_note("X", 0.62, 0.55) is None

    def test_silent_at_exactly_even(self):
        assert _value_dog_note("X", 0.50, 0.45) is None


class TestAnalyzersUseIt:
    """Guard the wiring: a pre-computed score string is what caused this."""

    def _src(self):
        with open("src/models/edge_finder.py") as f:
            return f.read()

    @pytest.mark.parametrize("sport", ["NFL", "NBA", "NHL", "WNBA"])
    def test_every_scoring_sport_derives_the_margin(self, sport):
        std = {"NFL": "NFL_SPREAD_STD", "NBA": "NBA_SPREAD_STD",
               "NHL": "NHL_SPREAD_STD", "WNBA": "WNBA_SPREAD_STD"}[sport]
        src = self._src()
        assert f"_score_from_prob(" in src
        assert std in src

    def test_nfl_no_longer_prebakes_the_signal_from_ppg(self):
        """The exact old line that caused the contradiction."""
        src = self._src()
        assert '_nfl_proj_signal = f"Model projected score: {home} {_nfl_ph:.0f}' not in src

    def test_nhl_no_longer_prebakes_the_signal(self):
        src = self._src()
        assert '_nhl_proj_signal = f"Model projected score: {home} {_nhl_ph:.1f}' not in src

    def test_cfb_already_derived_from_margin(self):
        """CFB was the one sport that always did this correctly — keep it."""
        src = self._src()
        cfb = src.split("def analyze_cfb_game")[1]
        assert "abs(margin) / 2.0" in cfb


class TestBlockLivesInTheRightFunction:
    """The Sep 17 change put the NFL block inside analyze_nba_game.

    `str.replace(old, new, 1)` was used on a marker line
    (`home_edge = adjusted_home_prob - market_home_prob`) that appears in FOUR
    analyzers — and NBA's comes first in the file. Consequences:
      * NFL lost its projected score entirely from the Sep 20 slate onward
        (user-reported: "i noticed they were removed at some point")
      * analyze_nba_game referenced `_nfl_proj_total` / `NFL_SPREAD_STD`, which
        do not exist there -> NameError the moment the NBA season starts. Silent
        only because NBA is out of season, which is exactly the stale-seasonal
        trap that hid the NFL shadow-settlement bug.
    """

    def _owner_of_each_call(self):
        import re
        src = open("src/models/edge_finder.py").read().split("\n")
        fns = [(i + 1, l.split("(")[0][4:]) for i, l in enumerate(src)
               if l.startswith("def ")]

        def owner(ln):
            cur = None
            for start, name in fns:
                if start <= ln:
                    cur = name
                else:
                    break
            return cur

        out = []
        for i, line in enumerate(src, 1):
            if "_score_from_prob(" in line and not line.strip().startswith("def "):
                ctx = "\n".join(src[i - 2:i + 1])
                toks = set(re.findall(
                    r"_(?:nfl|nba|nhl|wnba)_\w+|(?:NFL|NBA|NHL|WNBA)_SPREAD_STD", ctx))
                out.append((owner(i), toks))
        return out

    @pytest.mark.parametrize("sport", ["nfl", "nba", "nhl", "wnba"])
    def test_each_analyzer_has_its_own_block(self, sport):
        owners = [o for o, _ in self._owner_of_each_call()]
        assert f"analyze_{sport}_game" in owners

    def test_no_analyzer_uses_another_sports_variables(self):
        for fn, toks in self._owner_of_each_call():
            mine = fn.replace("analyze_", "").replace("_game", "")
            for t in toks:
                other = t.lower().lstrip("_").split("_")[0]
                assert other == mine, f"{fn} references {t}"

    def test_no_undefined_sport_locals_anywhere(self):
        """Static catch for the NameError class, across every analyzer."""
        import symtable
        src = open("src/models/edge_finder.py").read()
        bad = []
        for fn in symtable.symtable(src, "edge_finder.py", "exec").get_children():
            if not fn.get_name().startswith("analyze_"):
                continue
            for sym in fn.get_symbols():
                n = sym.get_name()
                if n.startswith(("_nfl_", "_nba_", "_nhl_", "_wnba_")):
                    if sym.is_referenced() and not sym.is_assigned() and not sym.is_global():
                        bad.append((fn.get_name(), n))
        assert not bad, f"undefined locals: {bad}"


class TestNflAnalyzerEmitsIt:
    """End-to-end: the user-visible symptom was an NFL card with no score."""

    def _run(self):
        from src.data.nfl_stats import normalize
        from src.models.edge_finder import analyze_nfl_game
        h, a = normalize("Buffalo Bills"), normalize("New York Giants")
        game = {"home_team": "Buffalo Bills", "away_team": "New York Giants",
                "commence_time": "2026-09-21T20:00:00Z",
                "moneyline": {"home_prob": 0.68, "away_prob": 0.32},
                "spread": {"home_spread": -6.0, "home_prob": 0.52, "away_prob": 0.48},
                "total": {"line": 44.5, "over_prob": 0.5, "under_prob": 0.5},
                "bookmakers": []}
        ctx = {"season_stats": {
                   h: {"ppg": 27.5, "oppg": 19.0, "wins": 2, "losses": 0, "net_rtg": 8.5},
                   a: {"ppg": 18.0, "oppg": 24.5, "wins": 0, "losses": 2, "net_rtg": -6.5}},
               "rest_days": {h: 7, a: 7}, "recent_form": {}}
        return analyze_nfl_game(game, ctx, {}, min_edge=0.0)

    def test_every_nfl_rec_carries_a_projected_score(self):
        recs = self._run()
        assert recs
        for r in recs:
            assert any("Model projected score" in s for s in r.signals), r.pick

    def test_all_markets_show_the_SAME_score(self):
        """ML and Spread cards for one game must not disagree."""
        scores = set()
        for r in self._run():
            for s in r.signals:
                if "Model projected score" in s:
                    scores.add(s)
        assert len(scores) == 1, scores

    def test_score_matches_the_probability(self):
        import re
        from src.models.edge_finder import NFL_SPREAD_STD
        recs = [r for r in self._run() if r.bet_type == "Moneyline"]
        r = recs[0]
        line = next(s for s in r.signals if "Model projected score" in s)
        nums = [float(x) for x in re.findall(r"(\d+)(?:\s|$)", line)]
        assert len(nums) == 2
        margin = nums[0] - nums[1]                      # home - away
        p_home = r.model_prob if r.pick == r.home_team else 1 - r.model_prob
        assert norm.cdf(margin / NFL_SPREAD_STD) == pytest.approx(p_home, abs=0.02)
