"""Kalshi game matching by ticker segment, not prose (Sep 11 2026).

NFL moneyline CLV resolved, but spread and total never did. Kalshi's rules
prose is inconsistent BETWEEN SERIES for the same game:

    KXNFLSPREAD : "San Francisco vs Los Angeles R Pro Football game ..."
    KXNFLTOTAL  : "San Francisco and Los Angeles collectively score ..."

Our token is "Los Angeles R", so the total's blob never matched and the pick
looked like an unlisted market. The ticker is exact — every book for one game
carries the same segment (26SEP10SFLAR) — so that is the join key now.
"""
from src.data.kalshi import _anchor_game_code, _game_code, resolve_pick

SERIES_MAP = {"Moneyline": "KXNFLGAME", "Spread": "KXNFLSPREAD",
              "Total": "KXNFLTOTAL"}


def _m(ticker, event, sub, rules="", date="2026-09-10"):
    return {"ticker": ticker, "event_ticker": event, "yes_sub_title": sub,
            "rules_primary": rules, "expiration_time": f"{date}T23:59:00Z",
            "close_time": f"{date}T23:59:00Z"}


def _markets():
    return {
        "KXNFLGAME": [
            _m("KXNFLGAME-26SEP10SFLAR-SF", "KXNFLGAME-26SEP10SFLAR",
               "San Francisco"),
            _m("KXNFLGAME-26SEP10SFLAR-LAR", "KXNFLGAME-26SEP10SFLAR",
               "Los Angeles R"),
        ],
        "KXNFLTOTAL": [
            # NOTE the prose: plain "Los Angeles", no "R".
            _m("KXNFLTOTAL-26SEP10SFLAR-49", "KXNFLTOTAL-26SEP10SFLAR",
               "Over 48.5 points scored",
               "If San Francisco and Los Angeles collectively score more than "
               "48.5 points ..."),
            _m("KXNFLTOTAL-26SEP10SFLAR-51", "KXNFLTOTAL-26SEP10SFLAR",
               "Over 50.5 points scored",
               "If San Francisco and Los Angeles collectively score more than "
               "50.5 points ..."),
        ],
        "KXNFLSPREAD": [
            _m("KXNFLSPREAD-26SEP10SFLAR-SF4", "KXNFLSPREAD-26SEP10SFLAR",
               "San Francisco wins by over 3.5 points",
               "If San Francisco wins by more than 3.5 points in the San "
               "Francisco vs Los Angeles R Pro Football game ..."),
        ],
    }


class TestGameCode:
    def test_extracts_segment_from_event_ticker(self):
        assert _game_code({"event_ticker": "KXNFLTOTAL-26SEP10SFLAR"}) == "26SEP10SFLAR"

    def test_falls_back_to_market_ticker(self):
        assert _game_code({"ticker": "KXNFLTOTAL-26SEP10SFLAR-49"}) == "26SEP10SFLAR"

    def test_same_segment_across_different_series(self):
        mk = _markets()
        codes = {_game_code(m) for ms in mk.values() for m in ms}
        assert codes == {"26SEP10SFLAR"}

    def test_malformed_ticker_is_not_fatal(self):
        assert _game_code({}) == ""
        assert _game_code({"ticker": "NODASHES"}) == ""


class TestAnchor:
    def test_finds_code_from_moneyline_book(self):
        code = _anchor_game_code(_markets(), "KXNFLGAME", "Los Angeles R",
                                 "San Francisco", "2026-09-10")
        assert code == "26SEP10SFLAR"

    def test_requires_both_teams(self):
        code = _anchor_game_code(_markets(), "KXNFLGAME", "Chicago",
                                 "San Francisco", "2026-09-10")
        assert code is None

    def test_no_anchor_series_is_not_fatal(self):
        assert _anchor_game_code(_markets(), None, "a", "b", "2026-09-10") is None


class TestResolution:
    """The actual regression: total and spread must resolve for NFL."""

    def test_total_resolves_despite_prose_mismatch(self):
        r = resolve_pick(
            {"bet_type": "Total", "pick": "Under 48.5",
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)
        assert r is not None
        assert r["ticker"] == "KXNFLTOTAL-26SEP10SFLAR-49"
        assert r["side"] == "no"          # Under = NO on an Over market

    def test_over_takes_the_yes_side(self):
        r = resolve_pick(
            {"bet_type": "Total", "pick": "Over 48.5",
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)
        assert r["side"] == "yes"

    def test_does_not_match_a_different_line(self):
        """`over 4.5` must never resolve onto `over 48.5`."""
        r = resolve_pick(
            {"bet_type": "Total", "pick": "Over 4.5",
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)
        assert r is None

    def test_spread_resolves(self):
        r = resolve_pick(
            {"bet_type": "Spread", "pick": "San Francisco 49ers -3.5",
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)
        assert r is not None
        assert r["ticker"] == "KXNFLSPREAD-26SEP10SFLAR-SF4"

    def test_whole_number_spread_has_no_market(self):
        """Kalshi is binary — -3.0 does not exist. This is why we snap to halves."""
        r = resolve_pick(
            {"bet_type": "Spread", "pick": "San Francisco 49ers -3.0",
             "home_team": "Los Angeles Rams", "away_team": "San Francisco 49ers"},
            _markets(), "2026-09-10", require_quote=False, series_map=SERIES_MAP)
        assert r is None


class TestAnchorSeriesIndexed:
    """The moneyline book must be fetched even with no moneyline pick that day."""

    def test_anchor_series_included(self):
        from src.data.kalshi_clv import _anchor_series
        assert "KXNFLGAME" in _anchor_series({"NFL"})
        assert "KXNCAAFGAME" in _anchor_series({"CFB"})

    def test_unknown_sport_is_not_fatal(self):
        from src.data.kalshi_clv import _anchor_series
        assert _anchor_series({"QUIDDITCH"}) == set()
