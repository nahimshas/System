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


class TestCardComparison:
    """The decisive half of the trial: would the same BETS come out?

    Coverage counts only prove the games were found. This re-runs the sport's
    own analyzer over Kalshi-priced games with the SAME context, so the only
    variable is the price source.
    """

    def test_pick_key_is_price_source_independent(self):
        from src.data.kalshi_pricing import _pick_key

        class R:
            game, bet_type, pick = "A @ B", "Moneyline", "A"
        assert _pick_key(R()) == ("A @ B", "Moneyline", "A")

    def test_no_kalshi_games_is_not_an_error(self):
        """Out-of-season or unmapped sports must report ok, not fail."""
        from src.data.kalshi_pricing import compare_picks
        rep = compare_picks(None, None, [], "CFB", "2026-10-22")
        assert rep["ok"] is True
        assert rep.get("skipped")

    def test_analyzer_errors_are_contained(self, monkeypatch):
        """Must not depend on a live slate — Kalshi only serves OPEN markets,
        so any hard-coded date stops discovering once its games settle."""
        import src.data.kalshi_pricing as KP
        monkeypatch.setattr(KP, "discover",
                            lambda s, d: [{"game_id": "x", "home_team": "A",
                                           "away_team": "B"}])

        class Boom:
            def analyze_games(self, g, c):
                raise RuntimeError("boom")
        rep = KP.compare_picks(Boom(), {}, [], "NHL", "2026-10-22")
        assert rep["ok"] is False and rep.get("error")

    def test_it_states_what_it_does_not_replay(self):
        """Budget routing runs ACROSS sports and cannot be reproduced here —
        the limitation must be recorded, not implied."""
        src = open("src/data/kalshi_pricing.py").read()
        body = src.split("def compare_picks")[1][:1200]
        assert "does not replay budget routing" in body.lower() or \
               "DOES NOT DO" in body

    def test_wired_behind_the_scenes(self):
        src = open("src/main.py").read()
        assert "record_card_comparison" in src
        block = src.split("Card comparison")[1][:1200]
        assert "try:" in block and "except Exception" in block
        assert "record_card_comparison" in block


class TestDiscoverMatchesTheValidatedResolver:
    """discover() must assemble the same prices the CLV resolver reads — that
    resolver is validated by 9,698 rows carrying both feeds."""

    def test_moneylines_agree_to_three_decimals(self):
        pytest.importorskip("requests")
        from datetime import date, timedelta
        from src.data.kalshi import resolve_pick
        from src.data.kalshi_clv import SPORT_SERIES, build_index
        from src.data.kalshi_pricing import discover

        for offset in range(0, 4):           # find a date with a live slate
            d = (date.today() + timedelta(days=offset)).isoformat()
            try:
                games = discover("NHL", d)
            except Exception as e:
                pytest.skip(f"network unavailable: {e}")
            if games:
                break
        else:
            pytest.skip("no NHL slate in the next few days")

        sm = SPORT_SERIES["NHL"]
        idx = build_index(sorted(set(sm.values())), {d})
        checked = 0
        for g in games:
            ml = g.get("moneyline")
            if not ml:
                continue
            r = resolve_pick({"bet_type": "Moneyline", "pick": g["home_team"],
                              "home_team": g["home_team"], "away_team": g["away_team"]},
                             idx, d, require_quote=True, series_map=sm)
            if not r or r.get("mid") is None:
                continue
            assert abs(ml["home_prob"] - r["mid"]) < 0.005, g["game_id"]
            checked += 1
        if not checked:
            pytest.skip("no two-sided quotes available")


class TestLiquidityGuard:
    """An untraded placeholder quote must never become a price.

    MLS totals quote bid 2c / ask 98c with zero open interest on every rung.
    The midpoint is 0.50, which looks exactly like a real coin-flip price — so
    `_best_total` happily selected "Over 0.5 goals at 50%" and the model found
    a 10% edge against a line that does not exist. Measured medians: NFL
    spread/total 2.0c, NBA 2.0/3.0c, NHL spread 3.0c, NHL total 19.0c, MLS
    spread/total ~40c.
    """

    def test_threshold_is_set(self):
        from src.data.kalshi_pricing import MAX_SPREAD_CENTS
        assert 1.0 <= MAX_SPREAD_CENTS <= 20.0

    def test_placeholder_quote_is_rejected(self):
        from src.data.kalshi_pricing import _mid
        assert _mid({"yes_bid_dollars": "0.02", "yes_ask_dollars": "0.98"}) is None

    def test_wide_but_plausible_quote_is_rejected(self):
        from src.data.kalshi_pricing import _mid
        assert _mid({"yes_bid_dollars": "0.40", "yes_ask_dollars": "0.60"}) is None

    def test_tight_quote_is_accepted(self):
        from src.data.kalshi_pricing import _mid
        assert _mid({"yes_bid_dollars": "0.48", "yes_ask_dollars": "0.52"}) == 0.50

    def test_one_sided_book_is_rejected(self):
        from src.data.kalshi_pricing import _mid
        assert _mid({"yes_bid_dollars": "0.00", "yes_ask_dollars": "0.05"}) is None


class TestThreeWaySoccer:
    """analyze_mls_game bails on `"draw_prob" not in ml`, so a 2-way moneyline
    produced ZERO soccer recs across the entire trial while discovery looked
    healthy — 13 MLS games found on Oct 24, 0 recommendations."""

    def test_soccer_leagues_are_marked_three_way(self):
        from src.data.kalshi_pricing import _THREE_WAY
        assert {"MLS", "LIGAMX"} <= _THREE_WAY

    def test_us_sports_are_not(self):
        from src.data.kalshi_pricing import _THREE_WAY
        for s in ("NFL", "NBA", "NHL", "MLB"):
            assert s not in _THREE_WAY

    def test_draw_is_included_and_normalised(self):
        pytest.importorskip("requests")
        from datetime import date, timedelta
        from src.data.kalshi_pricing import discover
        for off in range(0, 8):
            d = (date.today() + timedelta(days=off)).isoformat()
            try:
                games = [g for g in discover("MLS", d) if g.get("moneyline")]
            except Exception as e:
                pytest.skip(f"network unavailable: {e}")
            if games:
                break
        else:
            pytest.skip("no MLS slate in the next week")
        for g in games:
            ml = g["moneyline"]
            assert "draw_prob" in ml, g["game_id"]
            total = ml["home_prob"] + ml["draw_prob"] + ml["away_prob"]
            assert abs(total - 1.0) < 0.01, total


class TestPriceSourceRouting:
    """Kalshi is PRIMARY from Oct 27 2026; the Odds API subscription is being
    cancelled. CFB/IPL/WC stay on the Odds API because Kalshi cannot DISCOVER
    them — CFB has no static team map (matched structurally for CLV, which is
    not enough to build a slate), IPL has no markets at all, and WC's series
    ticker was never verified against a live book."""

    def test_budget_sports_are_all_kalshi_priceable(self):
        from src.data.kalshi_pricing import KALSHI_PRICEABLE
        from src.sports.registry import REGISTRY
        for slug, e in REGISTRY.items():
            if e.caps.enters_budget:
                assert slug.upper() in KALSHI_PRICEABLE, slug

    def test_unpriceable_sports_keep_the_odds_api(self):
        from src.data.kalshi_pricing import price_source_for
        for s in ("CFB", "IPL", "WC"):
            assert price_source_for(s) == "oddsapi"

    def test_priceable_sports_use_kalshi(self):
        from src.data.kalshi_pricing import price_source_for
        for s in ("MLB", "NBA", "NFL", "NHL", "WNBA", "MLS", "LIGAMX"):
            assert price_source_for(s) == "kalshi"

    def test_env_override_forces_odds_api(self, monkeypatch):
        """Reversible without a deploy if Kalshi goes wrong."""
        from src.data.kalshi_pricing import price_source_for
        monkeypatch.setenv("PRICE_SOURCE", "oddsapi")
        assert price_source_for("NHL") == "oddsapi"

    def test_env_override_forces_kalshi(self, monkeypatch):
        from src.data.kalshi_pricing import price_source_for
        monkeypatch.setenv("PRICE_SOURCE", "kalshi")
        assert price_source_for("CFB") == "kalshi"


class TestEmptySlateIsNotAnAlert:
    """⚠️ An alert that fires on every quiet day is one the reader learns to
    ignore — which is how the earlier silent failures survived for days. NFL on
    a Tuesday, WNBA out of season and a soccer league with no fixtures must all
    be silent; only a slate ESPN can see and Kalshi cannot is a real gap."""

    def test_quiet_day_does_not_report_a_fallback(self, monkeypatch):
        import src.data.kalshi_pricing as KP
        monkeypatch.setattr(KP, "discover", lambda s, d: [])
        monkeypatch.setattr(KP, "_espn_game_count", lambda s, d: 0)
        games, src, fell = KP.fetch_games_routed("NFL", None, "2026-10-27",
                                                 odds_key_available=False)
        assert games == [] and fell is False

    def test_real_gap_does_report_a_fallback(self, monkeypatch):
        import src.data.kalshi_pricing as KP
        monkeypatch.setattr(KP, "discover", lambda s, d: [])
        monkeypatch.setattr(KP, "_espn_game_count", lambda s, d: 8)

        class M:
            def fetch_games(self, d):
                return [{"game_id": "x"}]
        games, src, fell = KP.fetch_games_routed("NHL", M(), "2026-10-27",
                                                 odds_key_available=True)
        assert fell is True and src == "oddsapi" and len(games) == 1

    def test_unreachable_espn_does_not_silence_a_real_gap(self, monkeypatch):
        """ESPN down must not be read as 'no games'."""
        import src.data.kalshi_pricing as KP
        monkeypatch.setattr(KP, "discover", lambda s, d: [])
        monkeypatch.setattr(KP, "_espn_game_count", lambda s, d: None)
        games, src, fell = KP.fetch_games_routed("NHL", None, "2026-10-27",
                                                 odds_key_available=False)
        assert fell is True

    def test_discover_raising_falls_back(self, monkeypatch):
        import src.data.kalshi_pricing as KP

        def boom(s, d):
            raise RuntimeError("kalshi down")
        monkeypatch.setattr(KP, "discover", boom)
        monkeypatch.setattr(KP, "_espn_game_count", lambda s, d: 5)

        class M:
            def fetch_games(self, d):
                return [{"game_id": "x"}]
        games, src, fell = KP.fetch_games_routed("NHL", M(), "2026-10-27",
                                                 odds_key_available=True)
        assert src == "oddsapi" and len(games) == 1


class TestPropsAreOddsApiOnly:
    def test_prop_error_is_scoped_to_odds_api_slates(self):
        """Kalshi game dicts carry no player_props key, so without the guard
        every Kalshi slate would report a phantom props error daily."""
        src = open("src/main.py").read()
        assert '_price_src == "oddsapi"' in src
