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
def _ctx(home_elo, away_elo, games=10):
    return {"elo": {canon("Home U"): home_elo, canon("Away U"): away_elo},
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
    recs = analyze_cfb_game(_game(mk_home=0.40, mk_away=0.60), _ctx(1500, 1500),
                            min_edge=0.0)
    home = next(r for r in recs if r.pick == "Home U")
    expected = 0.5 + 0.0  # margin == HOME_ADV_POINTS
    from scipy.stats import norm
    assert home.model_prob_raw == pytest.approx(
        float(norm.cdf(CFB_HOME_ADV_POINTS / CFB_MARGIN_STD)), abs=1e-6)
    assert home.model_prob_raw > 0.5


def test_neutral_site_removes_home_advantage():
    normal = analyze_cfb_game(_game(mk_home=0.40), _ctx(1500, 1500), min_edge=0.0)
    neutral = analyze_cfb_game(_game(mk_home=0.40, neutral=True), _ctx(1500, 1500),
                               min_edge=0.0)
    h1 = next(r for r in normal if r.pick == "Home U").model_prob_raw
    h2 = next(r for r in neutral if r.pick == "Home U").model_prob_raw
    assert h1 > h2
    assert h2 == pytest.approx(0.5, abs=1e-6)


def test_elo_gap_converts_at_the_documented_rate():
    """250 Elo should be 10 points before home field."""
    recs = analyze_cfb_game(_game(mk_home=0.30), _ctx(1750, 1500), min_edge=0.0)
    home = next(r for r in recs if r.pick == "Home U")
    from scipy.stats import norm
    margin = 250 / CFB_ELO_TO_POINTS + CFB_HOME_ADV_POINTS
    assert margin == pytest.approx(12.6, abs=0.01)
    assert home.model_prob_raw == pytest.approx(
        float(norm.cdf(margin / CFB_MARGIN_STD)), abs=1e-6)


def test_unrated_opponent_is_skipped_entirely():
    """FCS teams and unmappable names have no rating. Defaulting them to average
    would manufacture a huge phantom edge against a real opponent — the single
    most likely way this model produces nonsense."""
    ctx = {"elo": {canon("Home U"): 1600}, "games": {canon("Home U"): 10}}
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
    analyze_cfb_game(early, _ctx(1620, 1500, games=CFB_MIN_RATED_GAMES), min_edge=0.0)
    analyze_cfb_game(late_g, _ctx(1620, 1500, games=20), min_edge=0.0)
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
    analyze_cfb_game(gp, _ctx(1500, 1500), min_edge=0.0)
    analyze_cfb_game(gm, _ctx(1500, 1500), min_edge=0.0)
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
    recs = analyze_cfb_game(_game(home_spread=-3.5, mk_home=0.40), _ctx(1600, 1500),
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
    analyze_cfb_game(g, _ctx(1600, 1500), min_edge=0.0)
    feats = g["_decision"]["features"]
    for k in ("cfb_elo_home", "cfb_elo_away", "cfb_projected_margin", "cfb_ramp",
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
        ctx = _ctx(1500, 1490, games=8)
        g = _game(home_spread=-31.5, mk_home=0.95)
        assert analyze_cfb_game(g, ctx, min_edge=0.0) == []

    def test_reasonable_disagreement_still_produces_picks(self):
        ctx = _ctx(1500, 1490, games=8)
        g = _game(home_spread=-3.5, mk_home=0.5)
        assert analyze_cfb_game(g, ctx, min_edge=0.0)

    def test_gate_is_symmetric(self):
        """Being wildly wrong in the other direction is equally disqualifying."""
        ctx = _ctx(1900, 1400, games=8)   # model: home by 22
        g = _game(home_spread=2.5, mk_home=0.5)  # market: home is a 2.5 dog
        assert analyze_cfb_game(g, ctx, min_edge=0.0) == []

    def test_too_few_rated_games_produces_nothing(self):
        """Early-season ratings are a regressed prior, not evidence."""
        from src.config import CFB_MIN_RATED_GAMES
        ctx = _ctx(1600, 1500, games=CFB_MIN_RATED_GAMES - 1)
        assert analyze_cfb_game(_game(home_spread=-4.5), ctx, min_edge=0.0) == []

    def test_the_cap_must_not_be_the_thing_holding_picks_in_range(self):
        """If every emitted pick sits at exactly the cap, the model is
        overconfident and the gate has failed."""
        from src.config import CFB_CRED_CAP
        ctx = _ctx(1560, 1500, games=8)
        recs = analyze_cfb_game(_game(home_spread=-2.5, mk_home=0.48), ctx, min_edge=0.0)
        at_cap = [r for r in recs if abs(r.edge - CFB_CRED_CAP) < 1e-6]
        assert len(at_cap) < len(recs) or not recs, \
            "every pick pinned to the cap — the model is not calibrated"
