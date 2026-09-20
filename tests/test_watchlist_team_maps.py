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
                             WNBA_TEAM_TO_KALSHI, _team_token)

MAPS = {"WNBA": WNBA_TEAM_TO_KALSHI, "MLS": MLS_TEAM_TO_KALSHI,
        "LIGAMX": LIGAMX_TEAM_TO_KALSHI}


class TestNoCrossResolution:
    """A duplicate token resolves one team's CLV against ANOTHER team's book —
    strictly worse than having no CLV, because it looks like data."""

    @pytest.mark.parametrize("league", list(MAPS))
    def test_tokens_are_unique_within_a_league(self, league):
        dupes = [t for t, c in Counter(MAPS[league].values()).items() if c > 1]
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

    @pytest.mark.parametrize("league,n", [("WNBA", 15), ("MLS", 30), ("LIGAMX", 18)])
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
