"""
Kalshi CLV: apples-to-apples, additive, and never overwriting the existing
sportsbook CLV that calibration and the CLV governor depend on.

Built Aug 25 2026 to (a) recover F5 CLV, which the Odds API cannot provide at
any price, and (b) cut ~54% of daily Odds API credits. Validated on live data:
correlation 0.786 against sportsbook CLV where both exist.
"""
import pytest

from src.data import kalshi_clv as kc
from src.data.kalshi import resolve_pick


# ── the field-isolation contract ──────────────────────────────────────────
def test_kalshi_fields_are_namespaced_away_from_the_existing_clv():
    """Every field this module writes must be kalshi_*. Overwriting `clv` or
    `market_prob_at_close` would silently re-point the CLV governor at a
    different venue mid-history."""
    import inspect
    import re
    src = inspect.getsource(kc.update_shadow_log_kalshi_clv)
    # single '=' only — '==' is a comparison, not a write
    written = set(re.findall(r'e\["(\w+)"\]\s*=(?!=)', src))
    protected = {"clv", "market_prob_at_close", "market_prob_at_first_pick",
                 "outcome", "close_point", "clv_fetch_attempts"}
    assert not (written & protected), f"writes protected fields: {written & protected}"
    assert written, "no writes detected — did the field names change?"
    assert all(w.startswith("kalshi_") for w in written), written


def test_ipl_is_deliberately_unsupported():
    """Kalshi lists no per-match IPL markets — only generic cricket T20I and an
    IPL-winner future. Silently mapping it would produce wrong closes."""
    assert "IPL" not in kc.SPORT_SERIES


def test_every_supported_sport_maps_the_three_core_markets():
    for sport, mkts in kc.SPORT_SERIES.items():
        for core in ("Moneyline", "Spread", "Total"):
            assert core in mkts, f"{sport} missing {core}"


# ── mid extraction, incl. the NO-side mirror ──────────────────────────────
def _candle(bid, ask, price=None, ts=1000):
    c = {"end_period_ts": ts,
         "yes_bid": {"close_dollars": str(bid)},
         "yes_ask": {"close_dollars": str(ask)}}
    if price is not None:
        c["price"] = {"close_dollars": str(price)}
    return c


def test_mid_is_the_midpoint_not_the_ask():
    """Mid overround is ~0%; the ask carries ~1%. Using the ask as fair value
    would bake execution cost into the CLV signal."""
    assert kc._mid_from_candle(_candle(0.54, 0.56), "yes") == 0.55


def test_no_side_mid_is_mirrored():
    assert kc._mid_from_candle(_candle(0.54, 0.56), "no") == 0.45


def test_falls_back_to_traded_price_when_unquoted():
    assert kc._mid_from_candle(_candle(0, 0, price=0.61), "yes") == 0.61


def test_rejects_degenerate_prices():
    assert kc._mid_from_candle(_candle(0, 0), "yes") is None
    assert kc._mid_from_candle(_candle(1.0, 1.0), "yes") is None


# ── the shadow-entry -> pick conversion that broke the first attempt ──────
def test_pick_text_keeps_the_line():
    """pick_side strips the line ('over', bare team name); pick keeps it. Using
    pick_side left the resolver unable to choose a strike off Kalshi's ladder,
    and 10 of 25 entries silently failed to match."""
    e = {"market_type": "Total", "pick": "Over 6.5", "pick_side": "over",
         "game": "Toronto Blue Jays @ Tampa Bay Rays"}
    p = kc._entry_to_pick(e)
    assert p["pick"] == "Over 6.5"
    assert p["away_team"] == "Toronto Blue Jays"
    assert p["home_team"] == "Tampa Bay Rays"


def test_spread_entry_keeps_its_sign():
    e = {"market_type": "Spread", "pick": "Chicago White Sox +1.5",
         "pick_side": "Chicago White Sox",
         "game": "Atlanta Braves @ Chicago White Sox"}
    assert kc._entry_to_pick(e)["pick"].endswith("+1.5")


def test_malformed_game_string_does_not_raise():
    p = kc._entry_to_pick({"market_type": "Moneyline", "pick": "X", "game": "nonsense"})
    assert p["home_team"] == "" and p["away_team"] == ""


# ── settled markets have no book; resolution must still work ──────────────
def test_settled_market_resolves_without_a_quote():
    """Past games carry no bid/ask. Requiring a quote made every backfill
    entry unresolvable — the failure that produced 0 stamped on the first run."""
    m = {"ticker": "KXMLBGAME-26AUG201240STLCIN-STL",
         "event_ticker": "KXMLBGAME-26AUG201240STLCIN",
         "yes_sub_title": "St. Louis",
         "rules_primary": "If St. Louis wins the St. Louis vs Cincinnati professional baseball game",
         "yes_bid_dollars": "0", "yes_ask_dollars": "0", "open_interest_fp": "0"}
    pick = {"bet_type": "Moneyline", "pick": "St. Louis Cardinals",
            "home_team": "Cincinnati Reds", "away_team": "St. Louis Cardinals"}
    assert resolve_pick(pick, {"KXMLBGAME": [m]}, "2026-08-20") is None
    r = resolve_pick(pick, {"KXMLBGAME": [m]}, "2026-08-20", require_quote=False)
    assert r is not None and r["ticker"].endswith("-STL") and r["side"] == "yes"


def test_backfill_entry_point_exists_and_is_idempotent_by_design():
    import inspect
    src = inspect.getsource(kc.update_shadow_log_kalshi_clv)
    assert 'e.get("kalshi_clv") is not None' in src, "must skip already-stamped rows"
    assert "MAX_ATTEMPTS" in src, "must cap retries on unmatchable entries"


# ── the switch to Kalshi as the primary CLV feed (Aug 25 2026) ────────────
class TestClvSourceSwitch:
    """Kalshi becomes the primary CLV feed WITHOUT erasing sportsbook history.

    Kalshi retains settled markets only back to 2026-06-18 and that window
    moves forward, so pre-June history is permanently unreachable. Rather than
    truncate the governor's window, entries fall back to the `clv` they already
    carry — every historical value stays exactly where it was.
    """

    def test_kalshi_is_preferred_when_present(self):
        from src.state.clv_governor import effective_clv, clv_source
        e = {"clv": -0.02, "kalshi_clv": -0.005}
        assert effective_clv(e) == -0.005
        assert clv_source(e) == "kalshi"

    def test_falls_back_to_sportsbook_history_never_dropping_it(self):
        """Pre-2026-06-18 rows have no Kalshi data. They must still count, or
        the governor silently loses four months of evidence."""
        from src.state.clv_governor import effective_clv, clv_source
        e = {"clv": -0.02, "kalshi_clv": None}
        assert effective_clv(e) == -0.02
        assert clv_source(e) == "odds_api"

    def test_entry_with_no_clv_at_all_is_none(self):
        from src.state.clv_governor import effective_clv, clv_source
        assert effective_clv({}) is None
        assert clv_source({}) == "none"

    def test_a_gate_can_never_be_loosened_by_the_feed_change(self):
        """THE SAFETY PROPERTY. Preferring Kalshi would have un-gated MLB Total
        (gated since Jun 11 on -2.03% over 593 picks; Kalshi reads -0.88%).
        Either source may turn a gate ON; both must agree to turn one OFF."""
        from src.state.clv_governor import _phase_and_gate_multi
        from datetime import date
        today = date.today().isoformat()
        st = {"n": 594, "avg_clv": -0.0088,          # Kalshi: would not gate
              "alt_n": 354, "alt_avg_clv": -0.0209,  # Odds API: would gate
              "alt_latest": today}
        _, gated = _phase_and_gate_multi(st)
        assert gated is True, "switching feeds silently removed a live gate"

    def test_either_source_can_turn_a_gate_on(self):
        from src.state.clv_governor import _phase_and_gate_multi
        from datetime import date
        st = {"n": 200, "avg_clv": -0.03, "alt_n": 200, "alt_avg_clv": 0.01,
              "alt_latest": date.today().isoformat()}
        assert _phase_and_gate_multi(st)[1] is True

    def test_a_frozen_secondary_series_loses_its_vote(self):
        """Once the paid feed is off, its series stops updating. Without a
        sunset it would hold a market gated forever, breaking the governor's
        promise that gated markets recover automatically."""
        from src.state.clv_governor import _phase_and_gate_multi, ALT_FRESH_DAYS
        from datetime import date, timedelta
        stale = (date.today() - timedelta(days=ALT_FRESH_DAYS + 5)).isoformat()
        st = {"n": 594, "avg_clv": -0.0088,
              "alt_n": 354, "alt_avg_clv": -0.0209, "alt_latest": stale}
        _, gated = _phase_and_gate_multi(st)
        assert gated is False, "a stale secondary series still gated the market"

    def test_missing_alt_series_does_not_crash_or_gate(self):
        from src.state.clv_governor import _phase_and_gate_multi
        st = {"n": 183, "avg_clv": -0.0007}          # F5: Kalshi-only
        assert _phase_and_gate_multi(st)[1] is False

    def test_paid_feed_is_off_by_default_but_re_enablable(self):
        import re
        for path in ("src/main.py", "src/data/results_snapshot.py"):
            src = open(path).read()
            assert 'ENABLE_ODDS_API_CLV' in src, f"{path} missing the switch"
            m = re.search(r'ENABLE_ODDS_API_CLV",\s*"(\w+)"', src)
            assert m and m.group(1) == "false", f"{path} must default OFF"
