"""Kalshi player props — ladder parsing, rung selection, and the small-sample
shrink that keeps one-game averages off the card."""
import pytest


class TestSubtitleParsing:
    def test_parses_name_and_threshold(self):
        from src.data.kalshi_props import _parse_subtitle
        assert _parse_subtitle("Jayson Tatum: 25+") == ("Jayson Tatum", 25.0)

    def test_handles_suffixed_and_accented_names(self):
        """⚠️ Parsed from yes_sub_title, never reversed out of the ticker:
        'BOSJTATUM0' and 'ATLNALEXANDERWALKER7' cannot be turned back into a
        name that matches ESPN."""
        from src.data.kalshi_props import _parse_subtitle
        assert _parse_subtitle("Michael Porter Jr.: 20+")[0] == "Michael Porter Jr."
        assert _parse_subtitle("Egor Dëmin: 10+")[0] == "Egor Dëmin"

    def test_rejects_junk(self):
        from src.data.kalshi_props import _parse_subtitle
        for bad in ("", "Yes", "No threshold here", "Tatum:", ": 25+"):
            assert _parse_subtitle(bad) is None


class TestLadderIsNativeNotTranslated:
    def test_rung_becomes_line_minus_half(self):
        """A rung at N is P(stat >= N); the analyzer wants P(stat > line), so
        line = N - 0.5. That is the continuity correction, not a fudge."""
        from src.data.kalshi_props import _parse_subtitle
        _, rung = _parse_subtitle("Jayson Tatum: 25+")
        assert rung - 0.5 == 24.5

    def test_poisson_rung_is_exact(self):
        """For the Poisson markets int(N-0.5) == N-1, and sf(N-1) == P(X>=N),
        so the conversion is exact rather than approximate."""
        from src.models.props_analyzer import _cover_prob
        from scipy.stats import poisson
        lam = 1.7
        assert _cover_prob("Home Runs Over", lam, 2 - 0.5) == pytest.approx(
            float(1 - poisson.cdf(1, lam)))


class TestMlbTickerTimeSegment:
    def test_mlb_key_drops_the_time(self):
        """MLB prop tickers carry a time the NBA ones do not. Without this
        every MLB prop fails to join its game and props look simply absent."""
        from src.data.kalshi_props import _mlb_game_key
        assert _mlb_game_key("26OCT272000MILTB") == "26OCT27MILTB"

    def test_nba_style_code_is_untouched(self):
        from src.data.kalshi_props import _mlb_game_key
        assert _mlb_game_key("26OCT27BKNBOS") == "26OCT27BKNBOS"


class TestUntradedQuotesAreRejected:
    def test_wide_quote_is_not_a_price(self):
        """An untraded contract sits at 2c/98c and a naive midpoint reads as a
        confident 50%."""
        from src.data.kalshi_props import _mid
        assert _mid({"yes_bid_dollars": 0.02, "yes_ask_dollars": 0.98}) is None

    def test_tight_quote_prices(self):
        from src.data.kalshi_props import _mid
        assert _mid({"yes_bid_dollars": 0.61, "yes_ask_dollars": 0.64}) == 0.625


class TestSmallSampleShrink:
    """⚠️ Wiring props to Kalshi produced 50 picks at a MEDIAN +14.0% edge with
    48/54 projections above the market. Median games played was ONE, and the
    stats source is ESPN's top-25 leaderboard, so a thin sample was also a
    SELECTED one. These tests pin the fix."""

    def test_thin_sample_refuses_to_project(self):
        from src.models.props_analyzer import shrink_projection, MIN_PROP_GAMES
        assert shrink_projection("Points Over", 18.4, 9.5, 0.59,
                                 MIN_PROP_GAMES - 1) is None

    def test_unknown_sample_refuses_to_project(self):
        """ESPN returns NO games count for prop players — verified, all 43 came
        back 0. Unknown must behave as untrusted, not as a full season."""
        from src.models.props_analyzer import shrink_projection
        assert shrink_projection("Points Over", 18.4, 9.5, 0.59, 0) is None

    def test_shrink_pulls_toward_the_market(self):
        from src.models.props_analyzer import shrink_projection
        got = shrink_projection("Points Over", 18.4, 9.5, 0.59, 5)
        assert got is not None and got < 18.4

    def test_bigger_sample_keeps_more_of_the_model(self):
        from src.models.props_analyzer import shrink_projection
        thin = shrink_projection("Points Over", 18.4, 9.5, 0.59, 5)
        thick = shrink_projection("Points Over", 18.4, 9.5, 0.59, 60)
        assert thin < thick <= 18.4

    def test_blank_stats_no_longer_claims_forty_games(self):
        """The sentinel was 40, which the shrink read as forty games of
        evidence — the opposite of what a missing stat line should buy."""
        from src.data.nba_stats import _BLANK_STATS
        assert _BLANK_STATS()["games"] == 0


class TestKalshiRoutingKeepsSportSideEffects:
    def test_mlb_exposes_an_enrichment_hook(self):
        """Kalshi bypasses fetch_games, and MLB's is the only one with side
        effects the sport depends on (probable pitchers + _last_schedule).
        Without the hook MLB props silently produce nothing."""
        from src.main import SPORT_MODULES
        assert callable(getattr(SPORT_MODULES["mlb"], "enrich_kalshi_games", None))

    def test_routing_calls_the_hook_when_present(self):
        src = open("src/data/kalshi_pricing.py").read()
        assert 'getattr(module, "enrich_kalshi_games", None)' in src
