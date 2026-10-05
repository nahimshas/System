"""Kalshi team maps for WNBA / MLS / LigaMX (Sep 19 2026).

These sports produced picks for months with ZERO CLV coverage -- WNBA alone had
243 shadow rows and not one measurable. `_teams_mappable()` skips a sport whose
names cannot be resolved, so the gap was silent BY DESIGN: it read as "not wired
yet" rather than "broken", which is accurate and therefore easy to leave
forever. Without CLV a market needs 500+ settled picks for a verdict instead of
~50, so this was the single biggest blind spot in the system.
"""
from collections import Counter

import pytest

from src.data.kalshi import (LIGAMX_TEAM_TO_KALSHI, MLS_TEAM_TO_KALSHI,
                             NBA_TEAM_TO_KALSHI, NHL_TEAM_TO_KALSHI,
                             WNBA_TEAM_TO_KALSHI, _team_token)

MAPS = {"WNBA": WNBA_TEAM_TO_KALSHI, "MLS": MLS_TEAM_TO_KALSHI,
        "LIGAMX": LIGAMX_TEAM_TO_KALSHI, "NBA": NBA_TEAM_TO_KALSHI,
        "NHL": NHL_TEAM_TO_KALSHI}

# NHL deliberately carries two ALIAS keys for name variants that appear in our
# own logs, so its token count is lower than its entry count.
ALIASED = {"NHL": {"Montreal", "St. Louis"}}


class TestNoCrossResolution:
    """A duplicate token resolves one team's CLV against ANOTHER team's book —
    strictly worse than having no CLV, because it looks like data."""

    @pytest.mark.parametrize("league", list(MAPS))
    def test_tokens_are_unique_within_a_league(self, league):
        allowed = ALIASED.get(league, set())
        dupes = [t for t, c in Counter(MAPS[league].values()).items()
                 if c > 1 and t not in allowed]
        assert not dupes, f"{league} would cross-resolve: {dupes}"

    def test_the_two_los_angeles_mls_clubs_are_distinct(self):
        assert (MLS_TEAM_TO_KALSHI["LA Galaxy"]
                != MLS_TEAM_TO_KALSHI["Los Angeles FC"])

    def test_the_two_new_york_mls_clubs_are_distinct(self):
        assert (MLS_TEAM_TO_KALSHI["New York City FC"]
                != MLS_TEAM_TO_KALSHI["New York Red Bulls"])

    def test_no_la_club_uses_the_bare_city(self):
        """"Los Angeles" alone is ambiguous in MLS and must never be a token."""
        assert "Los Angeles" not in MLS_TEAM_TO_KALSHI.values()


class TestCoverage:
    """Every name the odds feed has actually produced must resolve."""

    def _seen(self, sport):
        import glob
        import json
        out = set()
        for f in glob.glob("state/shadow_log/*.json") + glob.glob("state/decision_log/*.json"):
            d = json.load(open(f))
            rows = d.get("entries", d)
            rows = list(rows.values()) if isinstance(rows, dict) else rows
            for r in rows:
                if str(r.get("sport", "")).upper() != sport:
                    continue
                g = r.get("game") or ""
                if " @ " in g:
                    a, h = g.split(" @ ", 1)
                    out.update({a.strip(), h.strip()})
        return out

    @pytest.mark.parametrize("league", list(MAPS))
    def test_every_observed_name_resolves(self, league):
        unmapped = sorted(n for n in self._seen(league) if not _team_token(n))
        assert not unmapped, f"{league} unmapped: {unmapped}"

    @pytest.mark.parametrize("league,n", [("WNBA", 15), ("MLS", 30), ("LIGAMX", 18),
                                          ("NBA", 30), ("NHL", 34)])
    def test_map_is_the_expected_size(self, league, n):
        assert len(MAPS[league]) == n

    def test_accented_names_resolve(self):
        for name in ("América", "León", "Querétaro", "Atlético San Luis", "FC Juárez"):
            assert _team_token(name), name

    def test_unknown_name_is_still_none(self):
        assert _team_token("Nonexistent United") is None


class TestDrawMarket:
    """MLS Draw was the single biggest untracked bucket: 187 rows, no CLV."""

    def test_draw_maps_to_the_moneyline_book(self):
        from src.data.kalshi_clv import SPORT_SERIES
        for league in ("MLS", "LIGAMX"):
            assert SPORT_SERIES[league]["Draw"] == SPORT_SERIES[league]["Moneyline"]

    def test_resolver_recognises_draw(self):
        with open("src/data/kalshi.py") as f:
            src = f.read()
        assert 'bet_type in ("F5 Tie", "Draw")' in src

    def test_wc_is_absent_not_guessed(self):
        """WC is out of season; its series ticker could not be verified, and a
        guessed one would resolve nothing while LOOKING wired."""
        from src.data.kalshi_clv import SPORT_SERIES
        assert "WC" not in SPORT_SERIES


class TestMappableGate:
    """_teams_mappable is what skipped these sports entirely."""

    @pytest.mark.parametrize("sport,game", [
        ("WNBA", "Indiana Fever @ Las Vegas Aces"),
        ("MLS", "Los Angeles FC @ LA Galaxy"),
        ("LIGAMX", "Cruz Azul @ Monterrey"),
    ])
    def test_now_mappable(self, sport, game):
        from src.data.kalshi_clv import _teams_mappable
        assert _teams_mappable(game, sport) is True


class TestNbaNhlAreBudgetSports:
    """Both place REAL money, so an unmapped team is a real-money blind spot.

    history.json at the time these maps were added: NBA 57 bets (Apr-Jun,
    +$121.71 on $541 staked) and NHL 22 (Jun-Oct, 17 of them in the first days
    of October). Neither had a Kalshi map, so neither had ever produced CLV.
    """

    def test_both_are_flagged_as_budget(self):
        from src.sports.registry import REGISTRY
        for slug in ("nba", "nhl"):
            assert REGISTRY[slug].caps.enters_budget is True

    def test_full_leagues_are_mapped_not_just_teams_seen(self):
        """Our logs hold only NBA playoff teams; all 30 appear in the regular
        season, so mapping what we had seen would break in October."""
        assert len(set(NBA_TEAM_TO_KALSHI.values())) == 30
        assert len(set(NHL_TEAM_TO_KALSHI.values())) == 32


class TestAmbiguousPairsNbaNhl:
    def test_two_los_angeles_nba_clubs_are_distinct(self):
        assert (NBA_TEAM_TO_KALSHI["Los Angeles Lakers"]
                != NBA_TEAM_TO_KALSHI["Los Angeles Clippers"])

    def test_no_nba_club_uses_the_bare_city(self):
        assert "Los Angeles" not in NBA_TEAM_TO_KALSHI.values()

    def test_two_new_york_nhl_clubs_are_distinct(self):
        assert (NHL_TEAM_TO_KALSHI["New York Rangers"]
                != NHL_TEAM_TO_KALSHI["New York Islanders"])

    def test_no_nhl_club_uses_the_bare_new_york(self):
        assert "New York" not in NHL_TEAM_TO_KALSHI.values()

    def test_la_shorthand_resolves_via_expand_city(self):
        assert _team_token("LA Clippers") == NBA_TEAM_TO_KALSHI["Los Angeles Clippers"]


class TestNameVariants:
    """_team_token's fallback normalises whitespace and case only — it does not
    strip accents or punctuation, so both forms need explicit keys."""

    def test_accented_montreal(self):
        assert _team_token("Montréal Canadiens") == _team_token("Montreal Canadiens")

    def test_unpunctuated_st_louis(self):
        assert _team_token("St Louis Blues") == _team_token("St. Louis Blues")

    def test_utah_is_shared_across_leagues_safely(self):
        """Utah Jazz (NBA) and Utah Mammoth (NHL) both map to "Utah". That is
        fine because the series map scopes each lookup to one league's book."""
        assert _team_token("Utah Jazz") == "Utah"
        assert _team_token("Utah Mammoth") == "Utah"
