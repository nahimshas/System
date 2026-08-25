"""
Execution-measurement layer: Kalshi resolver + execution log + settlement math.

Aug 24 2026: measured that we pay ask + $0.02 on every bet — ~2pp over fair on
a coin-flip contract — while our measured edge is ~0. Robinhood's app price is
Kalshi's raw ask (verified to the cent), so this layer records the real book to
test whether buying at the BID instead closes the gap.
"""
import pytest

from src.data.kalshi import (TEAM_TO_KALSHI, SERIES, _quote, _pick_point,
                             _strip_point, event_date, resolve_pick)
from src.state.execution_log import FEES, make_key


# ── team + parsing helpers ────────────────────────────────────────────────
def test_every_mlb_team_maps_to_a_kalshi_token():
    assert len({v for v in TEAM_TO_KALSHI.values()}) == 30
    for tricky in ("Athletics", "Chicago White Sox", "Chicago Cubs",
                   "Los Angeles Angels", "Los Angeles Dodgers",
                   "New York Mets", "New York Yankees"):
        assert tricky in TEAM_TO_KALSHI


def test_event_date_parses_eastern_game_date():
    # A 6:40pm ET game on Aug 24 must not roll to Aug 25 (it would in UTC).
    assert event_date({"event_ticker": "KXMLBGAME-26AUG241840TBDET"}) == "2026-08-24"
    assert event_date({"event_ticker": "KXMLBF5-26SEP012105PITSD"}) == "2026-09-01"
    assert event_date({"event_ticker": "garbage"}) is None


def test_pick_point_and_strip():
    assert _pick_point("Athletics +1.5") == 1.5
    assert _pick_point("Pittsburgh Pirates -0.5") == -0.5
    assert _pick_point("Tampa Bay Rays") is None
    assert _strip_point("Athletics +1.5") == "Athletics"


# ── the NO-side mirror, which is where sign errors hide ───────────────────
def test_buying_no_mirrors_the_book():
    m = {"yes_bid_dollars": "0.45", "yes_ask_dollars": "0.46"}
    yes = _quote(m, "yes")
    no = _quote(m, "no")
    assert (yes["bid"], yes["ask"]) == (0.45, 0.46)
    # NO bid = 1 - yes ask, NO ask = 1 - yes bid
    assert (no["bid"], no["ask"]) == (0.54, 0.55)
    assert no["ask"] > no["bid"], "ask must never be below bid after mirroring"


def test_quote_rejects_crossed_or_missing_books():
    assert _quote({"yes_bid_dollars": "0", "yes_ask_dollars": "0"}, "yes") is None
    assert _quote({}, "yes") is None


# ── resolver: our pick -> the market we would actually buy ────────────────
def _mkt(series, sub, rules, yb, ya, ev="KXX-26AUG241840PITSD", oi=1000):
    return {"ticker": f"{series}-T", "event_ticker": ev, "yes_sub_title": sub,
            "rules_primary": rules, "yes_bid_dollars": str(yb),
            "yes_ask_dollars": str(ya), "open_interest_fp": str(oi)}


RULES = "in the Pittsburgh vs San Diego professional baseball game"


def test_plus_spread_resolves_to_the_no_side_of_the_opponent():
    """'Pirates +1.5' is NOT 'Pirates win by over 1.5' — it is the NO side of
    'San Diego wins by over 1.5'. Getting this backwards would silently record
    the wrong price for every underdog spread we take."""
    mkts = {"KXMLBSPREAD": [_mkt("KXMLBSPREAD", "San Diego wins by over 1.5 runs",
                                 RULES, 0.45, 0.46)]}
    r = resolve_pick({"bet_type": "Spread", "pick": "Pittsburgh Pirates +1.5",
                      "home_team": "San Diego Padres",
                      "away_team": "Pittsburgh Pirates"}, mkts, "2026-08-24")
    assert r is not None and r["side"] == "no"
    assert (r["bid"], r["ask"]) == (0.54, 0.55)


def test_minus_spread_resolves_to_the_yes_side():
    mkts = {"KXMLBSPREAD": [_mkt("KXMLBSPREAD", "Pittsburgh wins by over 1.5 runs",
                                 RULES, 0.30, 0.31)]}
    r = resolve_pick({"bet_type": "Spread", "pick": "Pittsburgh Pirates -1.5",
                      "home_team": "San Diego Padres",
                      "away_team": "Pittsburgh Pirates"}, mkts, "2026-08-24")
    assert r is not None and r["side"] == "yes" and r["bid"] == 0.30


def test_f5_half_point_spread_uses_the_f5_winner_book():
    """±0.5 over five innings just means 'wins the first 5' — Kalshi lists no
    0.5 F5 spread, so it must fall through to KXMLBF5."""
    mkts = {"KXMLBF5SPREAD": [],
            "KXMLBF5": [_mkt("KXMLBF5", "Pittsburgh wins first 5 innings",
                             "in the first 5 innings of the " + RULES, 0.44, 0.45)]}
    r = resolve_pick({"bet_type": "F5 Spread", "pick": "Pittsburgh Pirates -0.5",
                      "home_team": "San Diego Padres",
                      "away_team": "Pittsburgh Pirates"}, mkts, "2026-08-24")
    assert r is not None and r["side"] == "yes" and r["ask"] == 0.45


def test_under_resolves_to_the_no_side_of_the_over():
    mkts = {"KXMLBF5TOTAL": [_mkt("KXMLBF5TOTAL", "Over 3.5 runs in the first 5",
                                  "in the first 5 innings of the " + RULES, 0.54, 0.55)]}
    r = resolve_pick({"bet_type": "F5 Total", "pick": "Under 3.5",
                      "home_team": "San Diego Padres",
                      "away_team": "Pittsburgh Pirates"}, mkts, "2026-08-24")
    assert r is not None and r["side"] == "no" and (r["bid"], r["ask"]) == (0.45, 0.46)


def test_wrong_date_never_matches():
    """A market for another day's game must not be used — doubleheaders and
    series against the same opponent would otherwise cross-match."""
    mkts = {"KXMLBGAME": [_mkt("KXMLBGAME", "Pittsburgh", RULES, 0.47, 0.48,
                               ev="KXMLBGAME-26AUG251840PITSD")]}
    assert resolve_pick({"bet_type": "Moneyline", "pick": "Pittsburgh Pirates",
                         "home_team": "San Diego Padres",
                         "away_team": "Pittsburgh Pirates"},
                        mkts, "2026-08-24") is None


def test_unmatchable_pick_returns_none_not_an_exception():
    assert resolve_pick({"bet_type": "Prop", "pick": "whatever"}, {}, "2026-08-24") is None


# ── economics the whole exercise rests on ─────────────────────────────────
def test_fees_are_the_two_cent_all_in_cost():
    """Robinhood commission ($0.01) + Kalshi exchange fee ($0.01). Confirmed by
    the user in-app. Understating this would flatter every EV number here."""
    assert FEES == 0.02


def test_all_in_cost_beats_a_sportsbook_but_still_needs_real_edge():
    bid, ask, fair = 0.47, 0.48, 0.50
    assert round(fair - (ask + FEES), 4) == 0.0      # at ask: breakeven at best
    assert round(fair - (bid + FEES), 4) == 0.01     # at bid: +1pp


def test_execution_log_key_is_stable():
    a = make_key("2026-08-24", "A @ B", "Moneyline", "A")
    b = make_key("2026-08-24", "A @ B", "Moneyline", "A")
    assert a == b and a.count("|") == 3


def test_event_date_handles_tickers_with_and_without_a_start_time():
    """MLB tickers embed the start time (26AUG242145CINSF); WNBA/MLS/NBA do not
    (26AUG24ATLLA). Requiring the time group made event_date() return None for
    every non-MLB market, silently excluding all watchlist sports from Kalshi
    CLV — the index came back '0 markets' and nothing errored."""
    from src.data.kalshi import event_date
    cases = {
        "KXMLBGAME-26AUG242145CINSF-SF": "2026-08-24",
        "KXWNBAGAME-26AUG24ATLLA-LA":    "2026-08-24",
        "KXMLSGAME-26AUG23ATLSKC-TIE":   "2026-08-23",
        "KXMLBF5-26AUG012040SFSD-TIE":   "2026-08-01",
        "KXNBAGAME-26OCT30BOSNYK-BOS":   "2026-10-30",
    }
    for ticker, want in cases.items():
        assert event_date({"ticker": ticker, "event_ticker": ticker}) == want, ticker
    assert event_date({"ticker": "garbage"}) is None
