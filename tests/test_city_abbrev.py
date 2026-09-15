"""Feed city shorthand broke CLV and settlement at once (Sep 15 2026).

The odds feed emits "NY Jets" / "NY Giants" while both the Kalshi team map and
ESPN say "New York Jets" / "New York Giants". One inconsistency, two unrelated-
looking silent failures:

  * _team_token("NY Jets") -> None, so the game was never found and the pick got
    no CLV -- and no "not buyable" warning either, because an unmapped team
    empties the pool before any ladder is read.
  * _determine_outcome("NY Jets", ..., away="New York Jets") -> UNKNOWN, because
    the match is bidirectional substring and neither contains the other. Those
    picks stayed permanently unsettled.
"""
import pytest

from src.data.team_names import CITY_ABBREV, expand_city


class TestExpandCity:
    @pytest.mark.parametrize("raw,want", [
        ("NY Jets", "New York Jets"),
        ("NY Giants", "New York Giants"),
        ("LA Rams", "Los Angeles Rams"),
        ("SF 49ers", "San Francisco 49ers"),
        ("KC Chiefs", "Kansas City Chiefs"),
    ])
    def test_expands_leading_token(self, raw, want):
        assert expand_city(raw) == want

    @pytest.mark.parametrize("raw", [
        "Dallas Cowboys", "New York Jets", "Tennessee Titans", "Kansas Jayhawks",
    ])
    def test_leaves_full_names_alone(self, raw):
        assert expand_city(raw) == raw

    def test_only_the_leading_token(self):
        """A name containing an abbrev later must be untouched."""
        assert expand_city("Sporting KC") == "Sporting KC"

    def test_bare_token_is_not_expanded(self):
        assert expand_city("NY") == "NY"

    def test_empty_and_none_are_safe(self):
        assert expand_city("") == ""
        assert expand_city(None) == ""

    def test_idempotent(self):
        assert expand_city(expand_city("NY Jets")) == "New York Jets"


class TestKalshiTokens:
    @pytest.mark.parametrize("raw", ["NY Jets", "NY Giants"])
    def test_ny_nfl_teams_now_map(self, raw):
        from src.data.kalshi import _team_token
        assert _team_token(raw) is not None

    def test_expanded_matches_canonical(self):
        from src.data.kalshi import _team_token
        assert _team_token("NY Jets") == _team_token("New York Jets")
        assert _team_token("NY Giants") == _team_token("New York Giants")

    def test_unknown_team_still_none(self):
        from src.data.kalshi import _team_token
        assert _team_token("NY Fictional") is None

    def test_single_source_of_truth(self):
        """kalshi.py must not keep its own copy of the map."""
        with open("src/data/kalshi.py") as f:
            src = f.read()
        assert "_CITY_ABBREV" not in src
        assert "from src.data.team_names import expand_city" in src


class TestOutcomeMatching:
    """The real Sep 13 games: Jets won 23-10, Giants won 28-20."""

    @pytest.mark.parametrize("pick,bt,home,away,hs,aws,want", [
        ("NY Jets",       "Moneyline", "Tennessee Titans", "New York Jets",  10, 23, "WON"),
        ("NY Jets +1.5",  "Spread",    "Tennessee Titans", "New York Jets",  10, 23, "WON"),
        ("NY Giants",     "Moneyline", "New York Giants",  "Dallas Cowboys", 28, 20, "WON"),
        ("NY Giants +2.5","Spread",    "New York Giants",  "Dallas Cowboys", 28, 20, "WON"),
        # the opposing side must still grade correctly
        ("Dallas Cowboys","Moneyline", "New York Giants",  "Dallas Cowboys", 28, 20, "LOST"),
        ("Tennessee Titans","Moneyline","Tennessee Titans","New York Jets",  10, 23, "LOST"),
    ])
    def test_grades_shorthand_picks(self, pick, bt, home, away, hs, aws, want):
        from src.data.outcome_checker import _determine_outcome
        assert _determine_outcome(pick, bt, home, away, float(hs), float(aws)) == want

    def test_no_unknowns_for_ny(self):
        from src.data.outcome_checker import _determine_outcome
        got = _determine_outcome("NY Jets", "Moneyline",
                                 "Tennessee Titans", "New York Jets", 10.0, 23.0)
        assert got != "UNKNOWN"


class TestShadowSettlerCoverage:
    """MAIN_SPORTS was {"NBA","MLB"} with a docstring saying NFL was out of season."""

    def test_supported_sports_are_derived(self):
        with open("src/state/shadow_log.py") as f:
            src = f.read()
        assert 'MAIN_SPORTS      = {"NBA", "MLB"}' not in src
        assert "ESPN_SPORT_PATHS" in src
        # The docstring must no longer CLAIM NFL is skipped. Match that claim
        # specifically: "skipped" alone also covers the idempotency note, and
        # the phrase survives in the comment explaining the old bug.
        doc = src.split("def settle_shadow_from_espn")[1].split('"""')[1]
        assert not [l for l in doc.splitlines()
                    if "NFL" in l and "skip" in l.lower()]

    def test_nfl_is_covered(self):
        from src.data.outcome_checker import ESPN_SPORT_PATHS
        main = set(ESPN_SPORT_PATHS) - {"IPL"}
        assert "NFL" in main
        assert "NHL" in main          # budget sport, belongs with the main map

    def test_watchlist_set_excludes_main_sports(self):
        from src.data.outcome_checker import ESPN_SPORT_PATHS, ESPN_WATCHLIST_PATHS
        main = set(ESPN_SPORT_PATHS) - {"IPL"}
        watch = (set(ESPN_WATCHLIST_PATHS) - main) - {"IPL"}
        assert not (main & watch)
        assert "CFB" in watch
