"""
College football — Elo-margin model, watchlist only.

CFB is modelled as a POINT MARGIN, unlike MLB (pitcher probabilities) or soccer
(Poisson scorelines): ~134 FBS teams play ~12 games each with enormous talent
dispersion, so a power rating that converts to a spread is the right shape and
per-game stat aggregation never gets the sample it needs.
"""
import math

import pytest

from src.config import (CFB_ELO_TO_POINTS, CFB_HOME_ADV_POINTS, CFB_MARGIN_STD,
                        CFB_CRED_CAP, CFB_PRIOR_REGRESSION)
from src.data.cfb_stats import canon, _expected, _margin_multiplier, _season_of
from src.models.edge_finder import analyze_cfb_game


# ── ratings plumbing ──────────────────────────────────────────────────────
def test_season_runs_august_to_january():
    """A January bowl belongs to the PREVIOUS season — getting this wrong
    regresses ratings mid-playoff."""
    from datetime import date
    assert _season_of(date(2026, 8, 30)) == 2026
    assert _season_of(date(2026, 12, 20)) == 2026
    assert _season_of(date(2027, 1, 8)) == 2026
    assert _season_of(date(2027, 7, 1)) == 2026
    assert _season_of(date(2027, 8, 1)) == 2027


def test_margin_multiplier_dampens_blowouts_and_corrects_favourites():
    """A 50-point win must not move a rating 2.5x as much as a 20-point win,
    and a heavy favourite blowing out a cupcake must move less than an even
    matchup doing the same (the 538 autocorrelation correction)."""
    small, big = _margin_multiplier(20, 0), _margin_multiplier(50, 0)
    assert big > small
    # Dampening means the multiplier grows SLOWER than the margin: a 2.5x
    # bigger win must move the rating by less than 2.5x.
    assert big / small < (50 / 20), "blowouts are not being dampened"
    assert _margin_multiplier(30, 400) < _margin_multiplier(30, 0)


def test_canon_normalises_punctuation():
    assert canon("Texas A&M Aggies") == canon("Texas A&M  Aggies")
    assert "and" in canon("Texas A&M")


# ── the margin model ──────────────────────────────────────────────────────
def _ctx(home_pts, away_pts, games=10):
    """Ratings are SRS margins IN POINTS (0 = average team), not Elo."""
    return {"srs": {canon("Home U"): home_pts, canon("Away U"): away_pts},
            "games": {canon("Home U"): games, canon("Away U"): games}}


def _game(home_spread=None, mk_home=0.5, mk_away=0.5, neutral=False):
    g = {"home_team": "Home U", "away_team": "Away U",
         "commence_time": "2026-09-05T23:00:00Z", "game_time": "7:00 PM",
         "moneyline": {"home_prob": mk_home, "away_prob": mk_away},
         "neutral_site": neutral}
    if home_spread is not None:
        g["spread"] = {"home_spread": home_spread, "home_prob": 0.5, "away_prob": 0.5}
    return g


def test_equal_ratings_give_the_home_team_exactly_the_home_edge():
    recs = analyze_cfb_game(_game(mk_home=0.40, mk_away=0.60), _ctx(0.0, 0.0),
                            min_edge=0.0)
    home = next(r for r in recs if r.pick == "Home U")
    expected = 0.5 + 0.0  # margin == HOME_ADV_POINTS
    from scipy.stats import norm
    assert home.model_prob_raw == pytest.approx(
        float(norm.cdf(CFB_HOME_ADV_POINTS / CFB_MARGIN_STD)), abs=1e-6)
    assert home.model_prob_raw > 0.5


def test_neutral_site_removes_home_advantage():
    normal = analyze_cfb_game(_game(mk_home=0.40), _ctx(0.0, 0.0), min_edge=0.0)
    neutral = analyze_cfb_game(_game(mk_home=0.40, neutral=True), _ctx(0.0, 0.0),
                               min_edge=0.0)
    h1 = next(r for r in normal if r.pick == "Home U").model_prob_raw
    h2 = next(r for r in neutral if r.pick == "Home U").model_prob_raw
    assert h1 > h2
    assert h2 == pytest.approx(0.5, abs=1e-6)


def test_rating_gap_is_already_in_points():
    """SRS ratings ARE margins, so a 10-point gap projects 10 points (plus home
    field) with NO conversion constant. That conversion is what was
    miscalibrated ~8x and produced cap-pinned picks on Sep 3 2026."""
    g = _game(home_spread=-10.5, mk_home=0.30)
    analyze_cfb_game(g, _ctx(10.0, 0.0, games=9), min_edge=0.0)
    margin = g["_decision"]["features"]["cfb_projected_margin"]
    assert margin == pytest.approx(10.0 + CFB_HOME_ADV_POINTS, abs=1e-6)


def test_unrated_opponent_is_skipped_entirely():
    """FCS teams and unmappable names have no rating. Defaulting them to average
    would manufacture a huge phantom edge against a real opponent — the single
    most likely way this model produces nonsense."""
    ctx = {"srs": {canon("Home U"): 4.0}, "games": {canon("Home U"): 10}}
    assert analyze_cfb_game(_game(mk_home=0.30), ctx, min_edge=0.0) == []


def test_early_season_shrinks_the_rating_gap():
    """Week 1 ratings are a regressed prior, not evidence. The gap must be
    damped until real games accumulate."""
    from src.config import CFB_MIN_RATED_GAMES, CFB_WARMSTART_RAMP_GAMES
    assert CFB_WARMSTART_RAMP_GAMES > CFB_MIN_RATED_GAMES, \
        "ramp must outlast the min-games gate or it never damps a live pick"
    # Both fixtures clear the min-games gate; only the ramp differs.
    early = _game(home_spread=-6.5, mk_home=0.30)
    late_g = _game(home_spread=-6.5, mk_home=0.30)
    analyze_cfb_game(early, _ctx(4.8, 0.0, games=CFB_MIN_RATED_GAMES), min_edge=0.0)
    analyze_cfb_game(late_g, _ctx(4.8, 0.0, games=20), min_edge=0.0)
    assert early["_decision"]["features"]["cfb_ramp"] < 1.0
    assert late_g["_decision"]["features"]["cfb_ramp"] == 1.0
    assert (early["_decision"]["features"]["cfb_projected_margin"]
            < late_g["_decision"]["features"]["cfb_projected_margin"])


def test_spread_probability_uses_the_line():
    """P(home covers L) = Phi((margin + L)/sigma): a bigger cushion must help.

    Read the probabilities off the DECISION LOG, which records both sides
    regardless of edge — an unfavourable side is correctly never emitted as a
    pick, so asserting on recs would only test the edge filter."""
    # Lines kept inside CFB_MAX_MARGIN_DISAGREE so the sanity gate lets them through.
    gp = _game(home_spread=5.5, mk_home=0.5)
    gm = _game(home_spread=-5.5, mk_home=0.5)
    analyze_cfb_game(gp, _ctx(0.0, 0.0), min_edge=0.0)
    analyze_cfb_game(gm, _ctx(0.0, 0.0), min_edge=0.0)
    def _home_cover(g):
        return next(c["model_prob_raw"] for c in g["_decision"]["candidates"]
                    if c["market_type"] == "Spread" and c["side"] == "Home U")
    assert _home_cover(gp) > 0.5 > _home_cover(gm)


def test_totals_are_not_modelled():
    """Deliberate: totals need a pace model and are the worst-priced CFB market
    on Kalshi (median open interest 0, 18-cent spreads)."""
    g = _game(home_spread=-3.5, mk_home=0.45)
    g["total"] = {"line": 55.5, "over_prob": 0.5, "under_prob": 0.5}
    recs = analyze_cfb_game(g, _ctx(1550, 1500), min_edge=0.0)
    assert not [r for r in recs if r.bet_type == "Total"]


def test_all_picks_are_zero_sized():
    """Watchlist only — CFB must never allocate money."""
    recs = analyze_cfb_game(_game(home_spread=-3.5, mk_home=0.40), _ctx(4.0, 0.0),
                            min_edge=0.0)
    assert recs
    for r in recs:
        assert r.sizing.num_contracts == 0 and r.sizing.total_cost == 0
        assert r.sport == "CFB"


def test_registry_keeps_cfb_out_of_the_budget():
    from src.sports.registry import REGISTRY
    caps = REGISTRY["cfb"].caps
    assert caps.enters_budget is False
    assert caps.enters_parlays is False
    assert caps.in_main_display_pool is False


def test_decision_log_stamps_the_model_inputs():
    g = _game(home_spread=-3.5, mk_home=0.40)
    analyze_cfb_game(g, _ctx(4.0, 0.0), min_edge=0.0)
    feats = g["_decision"]["features"]
    for k in ("cfb_srs_home", "cfb_srs_away", "cfb_projected_margin", "cfb_ramp",
              "cfb_neutral_site", "cfb_games_home"):
        assert k in feats, f"{k} missing from decision-log features"
    kinds = {c["market_type"] for c in g["_decision"]["candidates"]}
    assert kinds == {"Moneyline", "Spread"}


# ── Kalshi team matching (no static map for 134 schools) ──────────────────
class TestCfbTeamMatching:
    def test_state_abbreviation_matches(self):
        from src.data.kalshi import cfb_team_matches
        assert cfb_team_matches("Boise State Broncos", "Boise St.")
        assert cfb_team_matches("Appalachian State Mountaineers", "Appalachian St.")

    def test_a_shorter_school_never_swallows_a_longer_one(self):
        """'Ohio' and 'Ohio State' are different FBS programmes. A prefix match
        alone would resolve Ohio State's market to Ohio's book."""
        from src.data.kalshi import cfb_team_matches
        assert not cfb_team_matches("Ohio State Buckeyes", "Ohio")
        assert cfb_team_matches("Ohio State Buckeyes", "Ohio St.")
        assert cfb_team_matches("Ohio Bobcats", "Ohio")
        assert not cfb_team_matches("Georgia Bulldogs", "Georgia Tech")

    def test_ambiguous_token_refuses_rather_than_guesses(self):
        from src.data.kalshi import _cfb_token_from_markets
        markets = [{"yes_sub_title": "Miami"}, {"yes_sub_title": "Miami"}]
        assert _cfb_token_from_markets("Miami Hurricanes", markets) == "Miami"
        assert _cfb_token_from_markets("Nowhere State", markets) is None

    def test_series_map_must_be_passed_for_non_mlb(self):
        """resolve_pick defaults to the MLB series map; without an override every
        CFB lookup lands in the baseball books and resolves nothing."""
        from src.data.kalshi import resolve_pick
        from src.data.kalshi_clv import SPORT_SERIES
        rules = "wins the Boise St. vs Fresno St. college football game"
        m = {"ticker": "KXNCAAFGAME-26SEP05AAABBB-AAA",
             "event_ticker": "KXNCAAFGAME-26SEP05AAABBB",
             "yes_sub_title": "Boise St.", "rules_primary": "If Boise St. " + rules,
             "yes_bid_dollars": "0.55", "yes_ask_dollars": "0.57",
             "open_interest_fp": "100"}
        m2 = dict(m, ticker="KXNCAAFGAME-26SEP05AAABBB-BBB",
                  yes_sub_title="Fresno St.",
                  rules_primary="If Fresno St. " + rules)
        pick = {"bet_type": "Moneyline", "pick": "Boise State Broncos",
                "home_team": "Fresno State Bulldogs", "away_team": "Boise State Broncos"}
        mk = {"KXNCAAFGAME": [m, m2]}
        assert resolve_pick(pick, mk, "2026-09-05") is None          # MLB map
        assert resolve_pick(pick, mk, "2026-09-05",
                            series_map=SPORT_SERIES["CFB"]) is not None


def test_new_watchlist_sport_cannot_break_the_spa_render():
    """A sport with no settled picks yields an EMPTY dict in the tile loop, and
    `{}.won` raises in strict Jinja — which aborts the ENTIRE SPA render, not
    just that tile. Adding CFB hit exactly this and silently killed the PWA
    (the render is exception-guarded, so it only logged a warning)."""
    tpl = open("src/report/templates/report_spa.html").read()
    assert "wl.get(sport_key, {'won': 0, 'lost': 0, 'total': 0, 'win_rate_pct': none})" in tpl, \
        "tile loop must default to a zeroed record, not {}"


def test_cfb_is_present_on_every_display_surface():
    tpl = open("src/report/templates/report_spa.html").read()
    for marker in ('tab-cfb', 'badge-cfb', "wl-tile-{{ sport_key }}",
                   "('CFB', 'badge-cfb', '🏈 CFB')", 'key:"cfb"'):
        assert marker in tpl, f"missing {marker}"
    gen = open("src/report/generator.py").read()
    assert '"cfb_watchlist":' in gen and '"has_cfb":' in gen


class TestCfbSanityGate:
    """Sep 3 2026: the model shipped and produced GARBAGE on its first slate.

    All five picks came out at EXACTLY +8.0% edge — the credibility cap value —
    on +26.5 to +42.5 underdogs the model believed were near-even. Root cause:
    week-1 Elo spanned only 372 points (max expressible margin 14.9) while the
    lines reached 42.5, so the model literally could not represent the game and
    the cap clamped every pick to its ceiling.

    A cap firing on 100% of picks is a symptom, not a safety net. These tests
    encode the gate that catches it.
    """

    def test_wild_disagreement_with_the_market_produces_nothing(self):
        ctx = _ctx(0.0, -0.4, games=8)
        g = _game(home_spread=-31.5, mk_home=0.95)
        assert analyze_cfb_game(g, ctx, min_edge=0.0) == []

    def test_reasonable_disagreement_still_produces_picks(self):
        ctx = _ctx(0.0, -0.4, games=8)
        g = _game(home_spread=-3.5, mk_home=0.5)
        assert analyze_cfb_game(g, ctx, min_edge=0.0)

    def test_gate_is_symmetric(self):
        """Being wildly wrong in the other direction is equally disqualifying."""
        ctx = _ctx(20.0, -2.0, games=8)   # model: home by 22
        g = _game(home_spread=2.5, mk_home=0.5)  # market: home is a 2.5 dog
        assert analyze_cfb_game(g, ctx, min_edge=0.0) == []

    def test_too_few_rated_games_produces_nothing(self):
        """Early-season ratings are a regressed prior, not evidence."""
        from src.config import CFB_MIN_RATED_GAMES
        ctx = _ctx(4.0, 0.0, games=CFB_MIN_RATED_GAMES - 1)
        assert analyze_cfb_game(_game(home_spread=-4.5), ctx, min_edge=0.0) == []

    def test_the_cap_must_not_be_the_thing_holding_picks_in_range(self):
        """If every emitted pick sits at exactly the cap, the model is
        overconfident and the gate has failed."""
        from src.config import CFB_CRED_CAP
        ctx = _ctx(2.4, 0.0, games=8)
        recs = analyze_cfb_game(_game(home_spread=-2.5, mk_home=0.48), ctx, min_edge=0.0)
        at_cap = [r for r in recs if abs(r.edge - CFB_CRED_CAP) < 1e-6]
        assert len(at_cap) < len(recs) or not recs, \
            "every pick pinned to the cap — the model is not calibrated"


class TestSrsRatings:
    """Margin ratings, which replaced win/loss Elo on Sep 3 2026.

    Elo threw away the only signal CFB gives you — how much teams win by — and
    one replayed season spanned just 372 Elo (14.9 expressible points) against
    lines reaching 45. SRS fits actual scoring margins, so ratings come out IN
    POINTS and there is no conversion constant left to miscalibrate.
    """

    def test_srs_recovers_a_known_margin(self):
        from src.data.cfb_stats import compute_srs
        # A beats everyone by 21 on neutral fields; B and C are even.
        games = [{"home": "A", "away": "B", "home_score": 35, "away_score": 14, "neutral": True},
                 {"home": "A", "away": "C", "home_score": 35, "away_score": 14, "neutral": True},
                 {"home": "B", "away": "C", "home_score": 21, "away_score": 21, "neutral": True}]
        r = compute_srs(games)
        assert r["a"] > r["b"] and r["a"] > r["c"]
        assert abs(r["b"] - r["c"]) < 0.5, "even teams should rate together"

    def test_ratings_are_centred_on_zero(self):
        """0 means an average team, which is what makes the gap a margin."""
        from src.data.cfb_stats import compute_srs
        games = [{"home": "A", "away": "B", "home_score": 30, "away_score": 10, "neutral": True},
                 {"home": "B", "away": "C", "home_score": 20, "away_score": 17, "neutral": True},
                 {"home": "C", "away": "A", "home_score": 14, "away_score": 28, "neutral": True}]
        r = compute_srs(games)
        assert abs(sum(r.values()) / len(r)) < 1e-6

    def test_blowouts_are_capped(self):
        """A 63-0 win must not count more than roughly a 35-0 one."""
        from src.data.cfb_stats import compute_srs, SRS_MARGIN_CAP
        base = [{"home": "A", "away": "B", "home_score": 28 + int(SRS_MARGIN_CAP),
                 "away_score": 0, "neutral": True}]
        huge = [{"home": "A", "away": "B", "home_score": 84, "away_score": 0, "neutral": True}]
        assert compute_srs(base)["a"] == pytest.approx(compute_srs(huge)["a"], abs=1e-6)

    def test_home_field_is_removed_before_rating(self):
        """Otherwise every home-heavy schedule inflates a team."""
        from src.data.cfb_stats import compute_srs
        from src.config import CFB_HOME_ADV_POINTS
        at_home = [{"home": "A", "away": "B", "home_score": 21, "away_score": 0, "neutral": False}]
        neutral = [{"home": "A", "away": "B", "home_score": 21, "away_score": 0, "neutral": True}]
        assert compute_srs(at_home)["a"] < compute_srs(neutral)["a"]

    def test_the_model_can_now_express_a_real_line(self):
        """The Sep 3 failure: a 31.5 line the model could not represent at all,
        so the cap pinned every pick to its ceiling."""
        ctx = {"srs": {canon("Home U"): 18.0, canon("Away U"): -14.0},
               "games": {canon("Home U"): 9, canon("Away U"): 9}}
        g = _game(home_spread=-31.5, mk_home=0.97)
        analyze_cfb_game(g, ctx, min_edge=0.0)
        margin = g["_decision"]["features"]["cfb_projected_margin"]
        assert margin > 30, f"still cannot express a 31.5 line (got {margin:.1f})"


def test_cfb_card_matches_the_other_sports():
    """The card template renders a teal, expandable 'Model projected score'
    line. CFB emitted 'Projected score margin', which the template ignored, so
    CFB cards looked different from every other sport."""
    from src.report.card_context import build_card_context
    ctx = {"srs": {canon("Home U"): 6.0, canon("Away U"): 0.0},
           "games": {canon("Home U"): 9, canon("Away U"): 9}}
    recs = analyze_cfb_game(_game(home_spread=-6.5, mk_home=0.40), ctx, min_edge=0.0)
    assert recs
    narrative, context = build_card_context(
        "CFB", recs[0].pick, recs[0].bet_type, recs[0].signals, recs[0].research,
        recs[0].model_prob, recs[0].market_prob, recs[0].edge)
    assert any(c.startswith("Model projected score") for c in context), context
    assert context[0].startswith("Model projected score"), "must sort first, like other sports"
    assert narrative, "CFB cards need a narrative to make the details expandable"
    assert "WATCHLIST ONLY" in narrative


def test_tab_autoshow_list_includes_every_sport_with_a_tab():
    """CFB was added to _tabDots (the dot) but NOT to _srvTabsWithPicks (auto
    show/hide), so its tab stayed hidden all day despite having picks — it
    rendered and was unreachable."""
    tpl = open("src/report/templates/report_spa.html").read()
    import re
    srv = re.search(r"_srvTabsWithPicks = \[\{% for k,v in \[(.*?)\] if v", tpl, re.S)
    dots = re.search(r"window\._tabDots = \{(.*?)\};", tpl, re.S)
    assert srv and dots
    for key in ("nba", "mlb", "ipl", "nhl", "wnba", "nfl", "mls", "wc", "ligamx", "cfb"):
        assert f"'{key}'" in srv.group(1), f"{key} missing from the auto-show list"
        assert f"{key}:" in dots.group(1), f"{key} missing from the dot list"
