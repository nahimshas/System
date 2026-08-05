"""
First-5-innings (F5) suite — model + settlement.

F5 is watchlist-only probation (Aug 2026): it is a bet on the STARTING PITCHERS
only, in a thinner market than the full game. These tests lock the two things
most likely to break silently:
  1. Calibration — F5 run expectations must stay near league reality (~2.4
     runs/team through 5). A first draft applied the full 9-inning pitching
     coefficient to a 5-inning baseline, making the starter ~2.7x too strong and
     producing 15pp disagreements with the market (phantom edges).
  2. Settlement — F5 grades on the score AFTER 5 INNINGS, never the final.
"""
import pytest

from src.data.outcome_checker import _determine_f5_outcome
from src.models.edge_finder import analyze_mlb_f5_game
from src.config import BUDGET_EXCLUDED_MARKETS

_ACE  = {"xfip": 3.10, "era": 2.90, "bb_per_9": 2.0, "k_per_9": 10.5,
         "innings_pitched": 120, "avg_ip_per_start": 6.0}
_WEAK = {"xfip": 5.20, "era": 5.40, "bb_per_9": 4.2, "k_per_9": 6.5,
         "innings_pitched": 100, "avg_ip_per_start": 5.0}
_AVG  = {"xfip": 4.10, "era": 4.05, "bb_per_9": 3.0, "k_per_9": 8.5,
         "innings_pitched": 110, "avg_ip_per_start": 5.5}
_CTX  = {"season_stats": {"Home Team": {"ops": 0.735}, "Away Team": {"ops": 0.735}}}


def _game(**mkts):
    g = {"home_team": "Home Team", "away_team": "Away Team",
         "commence_time": "2026-08-05T23:10:00Z"}
    g.update(mkts)
    return g


def _features(hp, ap, **mkts):
    g = _game(**(mkts or {"f5_moneyline": {"home_prob": 0.45, "draw_prob": 0.20,
                                           "away_prob": 0.35}}))
    analyze_mlb_f5_game(g, hp, ap, _CTX, min_edge=0.0)
    return g["_decision"]["features"]


class TestF5Calibration:
    def test_even_matchup_lands_near_league_average(self):
        """Two average starters → ~2.4 runs each through 5 (league reality)."""
        f = _features(_AVG, _AVG)
        assert 2.0 <= f["f5_lam_home"] <= 2.9
        assert 2.0 <= f["f5_lam_away"] <= 2.9
        total = f["f5_lam_home"] + f["f5_lam_away"]
        assert 4.0 <= total <= 5.6, f"F5 total {total:.2f} outside realistic range"

    def test_pitcher_effect_is_scaled_to_five_innings(self):
        """Regression guard: an ace must not swing F5 by an implausible amount.

        With the 9-inning coefficient mis-applied, ace-vs-weak produced a ~15pp
        disagreement with the market. Correctly scaled (x5/9) it is single digits.
        """
        f = _features(_ACE, _WEAK)
        assert f["f5_p_home"] < 0.65, (
            f"ace-vs-weak home prob {f['f5_p_home']:.3f} too extreme — "
            "pitcher weight likely unscaled"
        )
        # still directionally right
        assert f["f5_p_home"] > f["f5_p_away"]

    def test_three_way_probabilities_sum_to_one(self):
        f = _features(_AVG, _AVG)
        total = f["f5_p_home"] + f["f5_p_tie"] + f["f5_p_away"]
        assert abs(total - 1.0) < 1e-6

    def test_tie_probability_is_material(self):
        """A 5-inning game ties often — the tie leg must not be ~0."""
        f = _features(_AVG, _AVG)
        assert 0.10 < f["f5_p_tie"] < 0.35


class TestF5IsWatchlistOnly:
    def test_all_f5_markets_excluded_from_budget(self):
        for mkt in ("F5 Moneyline", "F5 Tie", "F5 Spread", "F5 Total"):
            assert mkt in BUDGET_EXCLUDED_MARKETS

    def test_recs_carry_zero_sizing(self):
        g = _game(f5_moneyline={"home_prob": 0.30, "draw_prob": 0.20, "away_prob": 0.50})
        recs = analyze_mlb_f5_game(g, _ACE, _WEAK, _CTX, min_edge=0.0)
        assert recs, "expected at least one F5 rec"
        for r in recs:
            assert r.sizing.num_contracts == 0
            assert r.sizing.total_cost == 0

    def test_no_recs_when_book_lists_no_f5(self):
        assert analyze_mlb_f5_game(_game(), _ACE, _WEAK, _CTX, min_edge=0.0) == []


class TestF5Settlement:
    """Real game: WSH led 3-0 after 5 innings, then LOST 6-3."""
    H, A = "Philadelphia Phillies", "Washington Nationals"
    H5, A5 = 0.0, 3.0

    def _g(self, pick, bt):
        return _determine_f5_outcome(pick, bt, self.H, self.A, self.H5, self.A5)

    def test_moneyline_uses_five_inning_score_not_final(self):
        assert self._g(self.A, "F5 Moneyline") == "WON"    # led after 5
        assert self._g(self.H, "F5 Moneyline") == "LOST"   # despite winning the game

    def test_tie_loses_a_team_pick(self):
        assert _determine_f5_outcome(self.A, "F5 Moneyline", self.H, self.A, 2.0, 2.0) == "LOST"
        assert _determine_f5_outcome("Tie", "F5 Tie", self.H, self.A, 2.0, 2.0) == "WON"
        assert self._g("Tie", "F5 Tie") == "LOST"

    def test_spread(self):
        assert self._g(f"{self.A} +1.5", "F5 Spread") == "WON"
        assert self._g(f"{self.H} -1.5", "F5 Spread") == "LOST"

    def test_total_and_push(self):
        assert self._g("Under 4.5", "F5 Total") == "WON"    # 3 total runs
        assert self._g("Over 4.5", "F5 Total") == "LOST"
        assert _determine_f5_outcome("Over 4", "F5 Total", self.H, self.A, 2.0, 2.0) == "PUSH"

    def test_missing_five_inning_data_is_unknown_not_guessed(self):
        assert _determine_f5_outcome(self.A, "F5 Moneyline", self.H, self.A, None, None) == "UNKNOWN"
