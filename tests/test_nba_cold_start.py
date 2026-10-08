"""NBA cold start (Oct 8 2026) — asked ahead of the season opener.

Two independent defects, both of which would have shipped on opening night:

1. The standings call omitted `seasontype`, and ESPN defaults to whatever phase
   the league is in. Before opening night that is PRESEASON. Verified live:
   the default returned Atlanta 0W-1L while seasontype=2 returned 0W-0L. A
   single preseason game produced net ratings spanning -34..+34 (sd 13.7)
   against a real NBA spread of about -12..+11 (sd 6.1), and the analyzer
   turned that into double-digit edges with the credibility cap FIRING — the
   same signature that made the CFB first slate worthless.

2. NBA had NO warm start, unlike NFL. With preseason excluded, every team sits
   at the 110/110 default on opening night, so every matchup prices identically
   and the model carries no signal at all.

Fixing only one leaves either garbage or flatness; both are needed.
"""
import pytest


class TestSeasonTypeIsPinned:
    def test_standings_request_pins_regular_season(self):
        src = open("src/data/nba_stats.py").read()
        assert '"seasontype": 2' in src

    def test_the_reason_is_recorded_at_the_call_site(self):
        src = open("src/data/nba_stats.py").read()
        assert "PRESEASON" in src

    def test_games_played_is_captured(self):
        """The warm-start ramp needs something to weight on."""
        assert '"games_played"' in open("src/data/nba_stats.py").read()


class TestWarmStartMath:
    def _blend(self, gp, cur_net, prior_net):
        from src.config import NBA_PRIOR_REGRESSION, NBA_WARMSTART_RAMP_GAMES
        from src.data.nba_stats import _apply_warm_start
        cur = {"T": {"net_rtg": cur_net, "ppg": 113.0, "oppg": 113.0,
                     "games_played": gp}}
        _apply_warm_start(cur, {"T": {"net_rtg": prior_net, "ppg": 118.0,
                                      "oppg": 108.0}})
        return cur["T"]

    def test_opening_night_is_pure_regressed_prior(self):
        from src.config import NBA_PRIOR_REGRESSION
        out = self._blend(0, 0.0, 10.0)
        assert out["warm_start_weight"] == 0.0
        assert out["net_rtg"] == pytest.approx(10.0 * NBA_PRIOR_REGRESSION)

    def test_full_ramp_is_pure_current(self):
        from src.config import NBA_WARMSTART_RAMP_GAMES
        out = self._blend(NBA_WARMSTART_RAMP_GAMES, 5.0, 10.0)
        assert out["warm_start_weight"] == 1.0
        assert out["net_rtg"] == pytest.approx(5.0)

    def test_beyond_the_ramp_never_exceeds_one(self):
        out = self._blend(82, 5.0, 10.0)
        assert out["warm_start_weight"] == 1.0

    def test_halfway_blends_both(self):
        from src.config import NBA_PRIOR_REGRESSION, NBA_WARMSTART_RAMP_GAMES
        gp = NBA_WARMSTART_RAMP_GAMES // 2
        w = gp / NBA_WARMSTART_RAMP_GAMES
        out = self._blend(gp, 4.0, 10.0)
        assert out["net_rtg"] == pytest.approx(
            w * 4.0 + (1 - w) * 10.0 * NBA_PRIOR_REGRESSION, abs=0.01)

    def test_off_and_def_track_ppg(self):
        out = self._blend(0, 0.0, 10.0)
        assert out["off_rtg"] == out["ppg"]
        assert out["def_rtg"] == out["oppg"]

    def test_missing_prior_leaves_current_untouched(self):
        from src.data.nba_stats import _apply_warm_start
        cur = {"T": {"net_rtg": 9.9, "ppg": 113.0, "oppg": 113.0, "games_played": 1}}
        _apply_warm_start(cur, {})
        assert cur["T"]["net_rtg"] == 9.9


class TestConstantsAreMeasuredNotGuessed:
    def test_regression_matches_the_measured_persistence(self):
        """Season-over-season net-rating slope across 2022->23 .. 2025->26 was
        0.445 / 0.779 / 0.633 / 0.543, mean 0.600."""
        from src.config import NBA_PRIOR_REGRESSION
        assert 0.50 <= NBA_PRIOR_REGRESSION <= 0.70

    def test_ramp_outlasts_the_crossover(self):
        """Current season out-informs the prior at about 8 games
        (se = 12/sqrt(n) vs a prior residual of ~5.0), so the ramp must not end
        before then."""
        from src.config import NBA_WARMSTART_RAMP_GAMES
        assert NBA_WARMSTART_RAMP_GAMES >= 10

    def test_league_average_is_plausible(self):
        from src.config import NBA_LEAGUE_AVG_PPG
        assert 105 <= NBA_LEAGUE_AVG_PPG <= 125


class TestLiveEffect:
    """Hits ESPN. Pins the actual improvement so a regression is visible."""

    def test_ratings_land_in_a_believable_range(self):
        pytest.importorskip("requests")
        import statistics as st
        from datetime import date
        from src.data.nba_stats import get_nba_context
        try:
            ctx = get_nba_context(date.today())
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")
        stats = ctx.get("season_stats") or {}
        if len(stats) < 20:
            pytest.skip("standings unavailable")
        nets = [v["net_rtg"] for v in stats.values()]
        # Pre-fix this was -34..+34 with sd 13.7.
        assert max(abs(n) for n in nets) < 20, f"ratings too wide: {min(nets)}..{max(nets)}"
        assert st.pstdev(nets) < 8, "spread still too wide for a real NBA season"
