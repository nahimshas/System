"""The Kalshi pricing path + its coverage trial (Oct 5 2026).

WHY A TRIAL AT ALL, GIVEN 9,698 ROWS OF PRICE DATA
Those rows prove Kalshi can PRICE A GAME WE ALREADY KNOW ABOUT — every one
began as a game the Odds API handed us. Nothing has ever exercised DISCOVERY:

    games = module.fetch_games(today_str) if ODDS_API_KEY else []

and no sport has another source. Pricing also gets no second chance: CLV
self-heals over a 7-day window, whereas a pricing failure at 9am is that day's
card, permanently. So the trial measures reliability, not price agreement.
"""
import pytest


class TestDiscoveryShape:
    """Must emit the same game dict the analyzers already read."""

    @pytest.fixture(scope="class")
    def games(self):
        pytest.importorskip("requests")
        from src.data.kalshi_pricing import discover
        try:
            g = discover("NHL", "2026-10-05")
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")
        if not g:
            pytest.skip("no NHL games listed for that date")
        return g

    def test_carries_the_odds_api_keys(self, games):
        for key in ("game_id", "sport", "home_team", "away_team",
                    "commence_time", "moneyline", "spread", "total"):
            assert key in games[0], key

    def test_team_names_are_ours_not_kalshi_tokens(self, games):
        """Analyzers key on our full names; a raw token would break every
        downstream lookup (stats, injuries, settlement)."""
        from src.data.kalshi import _team_token
        for g in games:
            assert _team_token(g["home_team"]), g["home_team"]
            assert _team_token(g["away_team"]), g["away_team"]

    def test_moneyline_sides_sum_to_about_one(self, games):
        """Both sides are one binary market, so no de-vigging is needed — and
        unlike the Odds API totals path, none is silently skipped either."""
        for g in games:
            ml = g.get("moneyline")
            if ml:
                assert abs(ml["home_prob"] + ml["away_prob"] - 1.0) < 0.03

    def test_commence_time_is_populated(self, games):
        assert all(g["commence_time"] for g in games)


class TestDiscoveryIsConservative:
    def test_unmappable_sports_return_nothing(self):
        """CFB has no static map by design; guessing would mis-assign teams."""
        from src.data.kalshi_pricing import discover
        assert discover("CFB", "2026-10-05") == []

    def test_unknown_sport_is_safe(self):
        from src.data.kalshi_pricing import discover
        assert discover("QUIDDITCH", "2026-10-05") == []


class TestCoverageLog:
    def test_report_never_raises(self):
        from src.data.kalshi_pricing import coverage_report
        rep = coverage_report("QUIDDITCH", "2026-10-05")
        assert rep["sport"] == "QUIDDITCH"

    def test_record_is_idempotent(self, tmp_path, monkeypatch):
        import json
        from datetime import date

        import src.data.kalshi_pricing as KP
        monkeypatch.setattr(KP, "COVERAGE_DIR", str(tmp_path))
        monkeypatch.setattr(KP, "coverage_report",
                            lambda s, d, o=None: {"sport": s, "date": d, "ok": True})
        assert KP.record_coverage(date(2026, 10, 5), "NFL")
        assert KP.record_coverage(date(2026, 10, 5), "NFL")
        shard = json.load(open(tmp_path / "2026-10.json"))
        assert len(shard["entries"]) == 1          # updated, not duplicated

    def test_wired_behind_the_scenes_only(self):
        """It must log and nothing else — the card stays on the Odds API."""
        src = open("src/main.py").read()
        assert "record_coverage" in src
        block = src.split("Kalshi pricing trial")[1][:700]
        assert "try:" in block and "except Exception" in block
        # must not feed the analyzers
        assert "analyze_games" not in block


class TestMapHealth:
    """The one check that matters before trusting Kalshi to price: a renamed
    MONEYLINE book takes the whole sport down, since every other book joins
    to it by ticker."""

    def test_all_mapped_sports_are_in_step(self):
        pytest.importorskip("requests")
        from src.data.kalshi_pricing import map_health
        stale = {}
        for sport in ("MLB", "NFL", "NBA", "NHL", "WNBA", "MLS", "LIGAMX"):
            try:
                h = map_health(sport)
            except Exception as e:
                pytest.skip(f"network unavailable: {e}")
            if h.get("error"):
                pytest.skip(h["error"])
            if not h.get("ok"):
                stale[sport] = h.get("unknown")
        assert not stale, f"team maps drifted from Kalshi: {stale}"

    def test_sports_without_a_static_map_are_not_flagged(self):
        from src.data.kalshi_pricing import map_health
        assert map_health("CFB")["ok"] is True

    def test_health_report_surfaces_it(self):
        src = open("tools/analysis/health_report.py").read()
        assert "check_kalshi_maps" in src
        assert '"kalshi_maps": check_kalshi_maps()' in src


class TestTickerOrderIsPerLeague:
    """⚠️ Kalshi's ticker segment orders the teams DIFFERENTLY by league.

        US sports : <date><AWAY><HOME>   26OCT04 LAR PHI  (LAR @ PHI)
        SOCCER    : <date><HOME><AWAY>   26OCT06 CHI VAN  (VAN @ CHI)

    Both verified against ESPN's own homeAway flags. Assuming one order
    everywhere SILENTLY SWAPS home and away for soccer — worse than no price,
    because every edge would be computed against the wrong team while the
    pipeline looks healthy. The coverage trial caught it: MLS and LigaMX
    discovered games normally but `matched` sat at 0 across 15 slates.
    """

    def test_soccer_leagues_are_marked_home_first(self):
        from src.data.kalshi_pricing import _CODE_HOME_FIRST
        assert _CODE_HOME_FIRST == {"MLS", "LIGAMX"}

    def test_us_sports_are_not_marked(self):
        from src.data.kalshi_pricing import _CODE_HOME_FIRST
        for sport in ("NFL", "NBA", "NHL", "MLB", "WNBA"):
            assert sport not in _CODE_HOME_FIRST

    def test_the_reason_is_recorded(self):
        src = open("src/data/kalshi_pricing.py").read()
        assert "SWAPS home and away" in src or "SWAPS" in src

    @pytest.mark.parametrize("sport,path,date", [
        ("NHL", "hockey/nhl", "2026-10-22"),
        ("MLS", "soccer/usa.1", "2026-10-25"),
        ("LIGAMX", "soccer/mex.1", "2026-10-25"),
    ])
    def test_home_away_matches_espn(self, sport, path, date):
        """The only check that actually proves it — ESPN is authoritative."""
        import json
        import urllib.request
        pytest.importorskip("requests")
        from src.data.kalshi_pricing import discover
        url = (f"https://site.api.espn.com/apis/site/v2/sports/{path}"
               f"/scoreboard?dates={date.replace('-', '')}")
        try:
            events = json.load(urllib.request.urlopen(url, timeout=20)).get("events", [])
            games = discover(sport, date)
        except Exception as e:
            pytest.skip(f"network unavailable: {e}")
        if not events or not games:
            pytest.skip("no games listed for that date")

        def last(s):
            return s.split()[-1].lower()

        espn = set()
        for e in events:
            c = e["competitions"][0]
            h = next(x for x in c["competitors"] if x["homeAway"] == "home")
            a = next(x for x in c["competitors"] if x["homeAway"] == "away")
            espn.add((last(a["team"]["displayName"]), last(h["team"]["displayName"])))
        ours = {(last(g["away_team"]), last(g["home_team"])) for g in games}
        flipped = {(h, a) for a, h in espn} & ours
        assert not flipped, f"{sport}: home/away inverted for {flipped}"
