"""Two silent data bugs found on the first Kalshi slate (Oct 28 2026).

Both were one-sided: the broken half was invisible because the working half
kept the output looking plausible.
"""
import pytest


class TestEspnStartTimeCorrection:
    """⚠️ Kalshi has NO start-time field. `occurrence_datetime` is the expected
    SETTLEMENT moment — for an NBA game ~3h after tip (LAL@LAC on 2026-10-28
    read 05:30Z for a game at 02:30Z). That is not merely a cosmetic display
    bug: commence_time drives the shadow log's GAME LOCK and the Kalshi CLV
    closing sample, so a late timestamp leaves picks editable deep into a game
    and samples the "close" after the market has already resolved."""

    def test_espn_paths_are_derived_not_hardcoded(self):
        """A literal list here silently excluded ligamx, wc and cfb —
        stamp_game_types returns 0 for any slug it cannot find, so those
        sports got no season type AND no start-time correction."""
        from src.data.game_types import _ESPN_PATHS
        for slug in ("nba", "mlb", "nfl", "nhl", "wnba", "mls",
                     "ligamx", "wc", "cfb", "ipl"):
            assert slug in _ESPN_PATHS, slug

    def test_espn_paths_match_the_settlement_maps(self):
        from src.data.game_types import _ESPN_PATHS
        from src.data.outcome_checker import (ESPN_SPORT_PATHS,
                                              ESPN_WATCHLIST_PATHS)
        for k, v in {**ESPN_WATCHLIST_PATHS, **ESPN_SPORT_PATHS}.items():
            assert _ESPN_PATHS[k.lower()] == v

    def test_la_clippers_matches_los_angeles_clippers(self):
        """ESPN's displayName is "LA Clippers"; our maps say "Los Angeles
        Clippers". Token overlap alone does NOT bridge it — "clippers" is one
        shared token against a two-token city difference. The cost: LAL@LAC was
        the ONE game of twelve that kept Kalshi's settlement timestamp, so it
        displayed 10:30 PM for a 7:30 PM tip, and the one game that never got
        a season type."""
        from src.data.game_types import _team_match
        assert _team_match("Los Angeles Clippers", "LA Clippers")
        assert _team_match("Los Angeles Lakers", "LA Lakers")

    def test_unrelated_teams_still_do_not_match(self):
        from src.data.game_types import _team_match
        assert not _team_match("Los Angeles Clippers", "Boston Celtics")

    def test_commence_time_is_taken_from_espn(self, monkeypatch):
        import src.data.game_types as GT
        monkeypatch.setattr(GT, "_fetch_events", lambda slug, d: [{
            "date": "2026-10-29T02:30Z",
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": "LA Clippers"}},
                {"homeAway": "away", "team": {"displayName": "Los Angeles Lakers"}},
            ], "notes": []}],
            "season": {"type": 2},
        }])
        games = [{"home_team": "Los Angeles Clippers",
                  "away_team": "Los Angeles Lakers",
                  # Kalshi's settlement stamp: 3h late
                  "commence_time": "2026-10-29T05:30:00Z"}]
        GT.stamp_game_types(games, "nba", "2026-10-28")
        assert games[0]["commence_time"] == "2026-10-29T02:30Z"

    def test_a_correct_time_is_left_alone(self, monkeypatch):
        """No-op on the Odds API path, which carries a real start time."""
        import src.data.game_types as GT
        monkeypatch.setattr(GT, "_fetch_events", lambda slug, d: [{
            "date": "2026-10-29T02:30Z",
            "competitions": [{"competitors": [
                {"homeAway": "home", "team": {"displayName": "LA Clippers"}},
                {"homeAway": "away", "team": {"displayName": "Los Angeles Lakers"}},
            ], "notes": []}],
            "season": {"type": 2},
        }])
        games = [{"home_team": "Los Angeles Clippers",
                  "away_team": "Los Angeles Lakers",
                  "commence_time": "2026-10-29T02:33:00Z"}]   # 3 min drift
        GT.stamp_game_types(games, "nba", "2026-10-28")
        assert games[0]["commence_time"] == "2026-10-29T02:33:00Z"


class TestMlsExpectedGoalsForKey:
    """⚠️ The ASA field is `xgoals_for`. None of the three names tried before it
    existed on the response, so EVERY team's attacking xG read 0.0 — while
    `xgoals_against` WAS in the defensive fallback list and worked.

    One-sided breakage is why it survived a month: the model still RANKED teams
    correctly off xGA, so MLS moneylines stayed well calibrated (2.2pp off the
    market) and the card looked fine. But expected total goals collapsed to
    ~1.3 against a market implying ~2.9, so every MLS total became "Under" at
    the capped +10% edge and the draw probability was inflated — two teams
    projected to score nothing draw 0-0 very often. Raw model-vs-market gap:
    47pp on totals, 39.8pp median across MLS, falling to 3.9pp once fixed."""

    def test_parses_the_real_asa_field_name(self):
        import src.data.mls_stats as M
        from src.data.mls_stats import _get_asa_season_stats
        rec = {"team_id": "T1", "count_games": 31,
               "xgoals_for": 56.2845, "xgoals_against": 49.185}
        orig = M._get_list
        M._get_list = lambda url, params=None: [rec]
        try:
            out = _get_asa_season_stats(2026, {"T1": "Austin FC"})
        finally:
            M._get_list = orig
        s = out["Austin FC"]
        assert s["xgf_per_game"] == pytest.approx(1.816, abs=1e-3)
        assert s["xga_per_game"] == pytest.approx(1.587, abs=1e-3)

    def test_attack_and_defence_are_both_populated(self):
        """The invariant that would have caught this: a team cannot have a
        real xGA and a zero xGF."""
        import src.data.mls_stats as M
        from src.data.mls_stats import _get_asa_season_stats
        rec = {"team_id": "T1", "count_games": 10,
               "xgoals_for": 16.0, "xgoals_against": 15.0}
        orig = M._get_list
        M._get_list = lambda url, params=None: [rec]
        try:
            out = _get_asa_season_stats(2026, {"T1": "Austin FC"})
        finally:
            M._get_list = orig
        s = out["Austin FC"]
        assert s["xgf_per_game"] > 0 and s["xga_per_game"] > 0
