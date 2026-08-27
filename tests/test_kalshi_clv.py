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


class TestDebriefClvWiring:
    """The debrief went blank on the first night after the feed switch.

    Two coupled faults: (1) clv_lookup_for_date was only CALLED inside the
    paid-feed branch, though reading the shadow log costs nothing; and (2) the
    lookup keyed on market_prob_at_close, which only the Odds API writes, so
    kalshi_clv rows were invisible to it. The nightly snapshot also never
    captured Kalshi CLV, so tonight's picks would always lag a day.
    """

    def test_lookup_is_not_gated_behind_the_paid_feed(self):
        src = open("src/data/results_snapshot.py").read()
        i_else = src.index("Odds API CLV capture disabled")
        i_lookup = src.index("clv_lookup_for_date(picks_date)")
        assert i_lookup > i_else, "lookup must sit outside the paid-feed branch"
        block = src[src.index("clv_lookup = clv_lookup_for_date"):]
        assert "_clv_disabled" not in block.split("\n")[0]

    def test_snapshot_captures_kalshi_clv_itself(self):
        """Debrief builds at ~10:46pm, before the next morning's run."""
        src = open("src/data/results_snapshot.py").read()
        assert "update_shadow_log_kalshi_clv" in src

    def test_lookup_accepts_a_kalshi_only_entry(self, tmp_path, monkeypatch):
        from datetime import date
        import json
        import src.data.closing_lines as cl
        shard = {"schema_version": 1, "month": "2026-08", "entries": {
            "k1": {"date": "2026-08-25", "game": "A @ B", "bet_type": "Moneyline",
                   "pick": "A", "sport": "MLB", "market_type": "Moneyline",
                   "kalshi_clv": -0.02, "kalshi_prob_at_pick": 0.55,
                   "kalshi_prob_at_close": 0.53,
                   "clv": None, "market_prob_at_close": None},
        }}
        d = tmp_path / "shadow"
        d.mkdir()
        (d / "2026-08.json").write_text(json.dumps(shard))
        monkeypatch.setattr(cl, "SHADOW_LOG_DIR", d)
        out = cl.clv_lookup_for_date(date(2026, 8, 25))
        assert ("A @ B", "Moneyline", "A") in out
        row = out[("A @ B", "Moneyline", "A")]
        assert row["clv"] == -0.02 and row["clv_source"] == "kalshi"
        assert row["market_prob_at_close"] == 0.53, "must use the Kalshi close"

    def test_lookup_still_serves_legacy_sportsbook_rows(self):
        """Pre-June rows have only the Odds API values and must keep working."""
        from datetime import date
        import json, tempfile, pathlib
        import src.data.closing_lines as cl
        with tempfile.TemporaryDirectory() as t:
            d = pathlib.Path(t)
            (d / "2026-05.json").write_text(json.dumps({"entries": {
                "k": {"date": "2026-05-10", "game": "C @ D", "bet_type": "Total",
                      "pick": "Over 8.5", "market_type": "Total",
                      "clv": 0.01, "market_prob_at_first_pick": 0.50,
                      "market_prob_at_close": 0.51}}}))
            orig = cl.SHADOW_LOG_DIR
            cl.SHADOW_LOG_DIR = d
            try:
                out = cl.clv_lookup_for_date(date(2026, 5, 10))
            finally:
                cl.SHADOW_LOG_DIR = orig
        row = out[("C @ D", "Total", "Over 8.5")]
        assert row["clv"] == 0.01 and row["clv_source"] == "odds_api"


class TestDecisionLogCandidateClv:
    """CLV must reach the CANDIDATE archive, not just the picks we made.

    Candidate CLV used to ride along with the Odds API pass. That feed went off
    Aug 25, so the decision log silently stopped accruing CLV — 183 of 719 rows
    in the week the switch landed. Measuring CLV on picks the model REJECTED is
    the whole reason the archive exists.
    """

    def test_moneyline_pick_text_is_the_side(self):
        e = {"game": "A Team @ B Team", "market_type": "Moneyline",
             "side": "A Team", "line": None}
        assert kc._decision_entry_to_pick(e)["pick"] == "A Team"

    def test_spread_reassembles_the_signed_line(self):
        """The decision log stores side and line separately; the resolver needs
        them recombined or it cannot pick a rung off Kalshi's ladder."""
        e = {"game": "A @ B", "market_type": "Spread", "side": "A", "line": 1.5}
        assert kc._decision_entry_to_pick(e)["pick"] == "A +1.5"
        e2 = {"game": "A @ B", "market_type": "F5 Spread", "side": "A", "line": -0.5}
        assert kc._decision_entry_to_pick(e2)["pick"] == "A -0.5"

    def test_total_maps_over_under_to_kalshi_wording(self):
        over = {"game": "A @ B", "market_type": "Total", "side": "over", "line": 8.5}
        under = {"game": "A @ B", "market_type": "F5 Total", "side": "under", "line": 4.5}
        assert kc._decision_entry_to_pick(over)["pick"] == "Over 8.5"
        assert kc._decision_entry_to_pick(under)["pick"] == "Under 4.5"

    def test_f5_tie_is_skipped(self):
        """No reliable draw price. Historical Tie rows carry
        market_prob_at_first_pick == 0.0 against model_prob ~0.20 — a fake 20pp
        edge — so they must not be revived here."""
        e = {"game": "A @ B", "market_type": "F5 Tie", "side": "tie", "line": None}
        assert kc._decision_entry_to_pick(e) is None

    def test_missing_line_returns_none_rather_than_guessing(self):
        e = {"game": "A @ B", "market_type": "Total", "side": "over", "line": None}
        assert kc._decision_entry_to_pick(e) is None

    def test_malformed_game_returns_none(self):
        assert kc._decision_entry_to_pick(
            {"game": "nonsense", "market_type": "Moneyline", "side": "X"}) is None

    def test_both_run_paths_capture_candidate_clv(self):
        for path in ("src/main.py", "src/data/results_snapshot.py"):
            assert "update_decision_log_kalshi_clv" in open(path).read(), path


class TestClvResolution:
    """Kalshi trades in 1-cent ticks, so a single mid can only land on a
    half-cent — quantising CLV to 0.5pp, LARGER than the ~0.4pp effect we are
    measuring. Measured Aug 27: 2022 of 2022 values were exact tick multiples
    versus 78% continuous from the sportsbook feed."""

    def test_window_is_configured(self):
        assert kc.WINDOW_SECONDS >= 300, "window too short to damp tick noise"

    def test_prices_are_averaged_not_sampled(self):
        import inspect
        src = inspect.getsource(kc.prices_at)
        assert "_window_mean" in src, "must average a window, not sample one candle"
        assert "sum(vals) / len(vals)" in src
