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


# ---------------------------------------------------------------------------
# F5 CLV extraction — proves our side of the pipeline works given a snapshot
# that contains the *_1st_5_innings bookmaker markets. The only remaining
# unknown is whether the Odds API historical endpoint serves them.
# ---------------------------------------------------------------------------

def _f5_event(three_way_ml=True):
    ml_outcomes = [
        {"name": "New York Yankees", "price": -140},
        {"name": "Atlanta Braves", "price": 120},
    ]
    if three_way_ml:
        ml_outcomes = [
            {"name": "New York Yankees", "price": -110},
            {"name": "Atlanta Braves", "price": 160},
            {"name": "Draw", "price": 260},
        ]
    return {
        "home_team": "New York Yankees",
        "away_team": "Atlanta Braves",
        "commence_time": "2026-08-08T19:05:00Z",
        "bookmakers": [{
            "key": "book1",
            "markets": [
                {"key": "h2h_1st_5_innings", "outcomes": ml_outcomes},
                {"key": "spreads_1st_5_innings", "outcomes": [
                    {"name": "New York Yankees", "price": -120, "point": 0.5},
                    {"name": "Atlanta Braves", "price": 100, "point": -0.5},
                ]},
                {"key": "totals_1st_5_innings", "outcomes": [
                    {"name": "Over", "price": -105, "point": 4.5},
                    {"name": "Under", "price": -115, "point": 4.5},
                ]},
            ],
        }],
    }


def test_f5_clv_extracts_all_three_markets():
    from src.data.closing_lines import _closing_prob_for_entry
    cases = [
        ("F5 Moneyline", "New York Yankees", "New York Yankees"),
        ("F5 Spread", "New York Yankees +0.5", "New York Yankees +0.5"),
        ("F5 Total", "Over 4.5", "Over 4.5"),
    ]
    ev = _f5_event()
    for market_type, pick_side, pick in cases:
        entry = {"market_type": market_type, "pick_side": pick_side, "pick": pick}
        res = _closing_prob_for_entry(ev, entry)
        assert res is not None, f"{market_type} produced no closing prob"
        assert 0.0 < res["prob"] < 1.0, f"{market_type} prob out of range: {res}"


def test_f5_moneyline_handles_two_way_books():
    """Some books price F5 h2h without the tie — must still yield a prob."""
    from src.data.closing_lines import _closing_prob_for_entry
    ev = _f5_event(three_way_ml=False)
    res = _closing_prob_for_entry(
        ev, {"market_type": "F5 Moneyline", "pick_side": "New York Yankees", "pick": ""}
    )
    assert res is not None and 0.0 < res["prob"] < 1.0


def test_f5_three_way_ml_prob_is_below_two_way():
    """With the tie priced, a team's F5 win prob must be lower than the
    two-way version of the same price — the draw takes probability mass."""
    from src.data.closing_lines import _closing_ml_prob
    p3 = _closing_ml_prob(_f5_event(True), "New York Yankees", "F5",
                          market_key="h2h_1st_5_innings")
    p2 = _closing_ml_prob(_f5_event(False), "New York Yankees", "F5",
                          market_key="h2h_1st_5_innings")
    assert p3 < p2


def test_f5_market_types_are_all_mapped():
    """The mapping stays complete even while F5 CLV is switched off, so
    flipping F5_CLV_ENABLED is the only change needed to re-enable it."""
    from src.data.closing_lines import _MARKET_KEYS
    for mt in ("F5 Moneyline", "F5 Tie", "F5 Spread", "F5 Total"):
        assert mt in _MARKET_KEYS, f"{mt} missing from _MARKET_KEYS"
        assert _MARKET_KEYS[mt].endswith("_1st_5_innings")


def test_f5_clv_is_disabled_and_gates_every_f5_market():
    """Aug 10 2026: the historical endpoint does not serve *_1st_5_innings, so
    F5 CLV is off by decision. The gate must exclude F5 without touching any
    other market, or we burn credits on requests that always 422."""
    from src.data import closing_lines as cl
    assert cl.F5_CLV_ENABLED is False
    for mt in ("F5 Moneyline", "F5 Tie", "F5 Spread", "F5 Total"):
        assert cl._clv_market_enabled(mt) is False
    for mt in ("Moneyline", "Spread", "Total", "Draw"):
        assert cl._clv_market_enabled(mt) is True
    assert cl._clv_market_enabled("Player Props") is False   # unmapped stays out


def test_f5_clv_switch_re_enables_cleanly():
    """Flipping the switch must restore F5 collection with no other edits."""
    from src.data import closing_lines as cl
    orig = cl.F5_CLV_ENABLED
    try:
        cl.F5_CLV_ENABLED = True
        for mt in ("F5 Moneyline", "F5 Tie", "F5 Spread", "F5 Total"):
            assert cl._clv_market_enabled(mt) is True
    finally:
        cl.F5_CLV_ENABLED = orig


# ---------------------------------------------------------------------------
# F5 grading in the nightly results snapshot (the debrief's only source).
# Aug 10 2026: every F5 pick showed ⏳ in the debrief because _resolve fell
# through to full-game grading, which has no F5 branch.
# ---------------------------------------------------------------------------

def _snapshot_event(home_f5, away_f5, completed=True):
    return {
        "home_name": "New York Yankees", "away_name": "Atlanta Braves",
        "home_score": 7.0, "away_score": 2.0,       # deliberately unlike the F5 score
        "home_f5": home_f5, "away_f5": away_f5,
        "completed": completed, "postponed": False,
        "event_date": "2026-08-09T19:05Z",
    }


def _resolve_f5(pick, bet_type, event):
    from src.data.results_snapshot import _resolve
    return _resolve("MLB", pick, bet_type, "New York Yankees", "Atlanta Braves",
                    "2026-08-09T19:05:00Z", {"MLB": [event]})


def test_snapshot_grades_f5_off_the_five_inning_score():
    """Braves lead 3-1 after 5 but lose 7-2 — the F5 pick must grade on the 5."""
    ev = _snapshot_event(home_f5=1.0, away_f5=3.0)
    assert _resolve_f5("Atlanta Braves", "F5 Moneyline", ev)["result"] == "WON"
    assert _resolve_f5("New York Yankees", "F5 Moneyline", ev)["result"] == "LOST"
    assert _resolve_f5("Over 3.5", "F5 Total", ev)["result"] == "WON"
    assert _resolve_f5("Under 3.5", "F5 Total", ev)["result"] == "LOST"


def test_snapshot_f5_settles_before_the_game_ends():
    """F5 is decided at the 5th — an unfinished game must not force PENDING."""
    ev = _snapshot_event(home_f5=1.0, away_f5=3.0, completed=False)
    assert _resolve_f5("Atlanta Braves", "F5 Moneyline", ev)["result"] == "WON"


def test_snapshot_f5_pending_without_five_innings():
    ev = _snapshot_event(home_f5=None, away_f5=None, completed=True)
    out = _resolve_f5("Atlanta Braves", "F5 Moneyline", ev)
    assert out["result"] == "PENDING"


def test_snapshot_f5_score_string_is_labelled_through_5():
    ev = _snapshot_event(home_f5=1.0, away_f5=3.0)
    assert "through 5" in _resolve_f5("Atlanta Braves", "F5 Moneyline", ev)["score"]


def test_snapshot_full_game_pick_unaffected_by_f5_branch():
    ev = _snapshot_event(home_f5=1.0, away_f5=3.0)
    out = _resolve_f5("New York Yankees", "Moneyline", ev)
    assert out["result"] == "WON" and "through 5" not in out["score"]
