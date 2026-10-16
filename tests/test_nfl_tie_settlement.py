"""An NFL tie is neither a push nor a plain loss (Oct 2026).

The Oct 11 NY Giants @ Washington game ended 27-27 on a moneyline pick and was
booked as a PUSH with $0.00 P&L. Two things were wrong with that, in opposite
directions:

  * Kalshi has NO separate Tie market in NFL. Every KXNFLGAME market says
    "If the game ends in a tie, the market will resolve to $0.50 for each team"
    (verified across all 56 live markets). So a tie pays out — buying under 50c
    PROFITS on a tie, buying over 50c loses. The $0.00 was wrong.
  * The pick was "this team wins" and the team did not win, so for judging the
    MODEL it is a loss, not a neutral event. (User decision, Oct 2026.)

Hence: recorded as LOST, with P&L from the real $0.50 settlement. Soccer is
different and already correct — it HAS a "Tie" market, so a draw sends a team
contract to $0 via _determine_mls_outcome.
"""
import pytest

from src.data.outcome_checker import _determine_outcome, _tie_pnl


class TestGrading:
    def test_tied_moneyline_returns_the_tie_marker(self):
        assert _determine_outcome("NY Giants", "Moneyline",
                                  "Washington Commanders", "NY Giants",
                                  27.0, 27.0) == "LOST_TIE"

    def test_it_is_no_longer_a_push(self):
        assert _determine_outcome("NY Giants", "Moneyline",
                                  "Washington Commanders", "NY Giants",
                                  27.0, 27.0) != "PUSH"

    def test_normal_results_are_untouched(self):
        assert _determine_outcome("NY Giants", "Moneyline", "Washington Commanders",
                                  "NY Giants", 20.0, 27.0) == "WON"
        assert _determine_outcome("NY Giants", "Moneyline", "Washington Commanders",
                                  "NY Giants", 27.0, 20.0) == "LOST"

    def test_soccer_draw_still_loses_a_team_bet(self):
        """Soccer HAS a Tie market, so a team contract goes to zero."""
        from src.data.outcome_checker import _determine_mls_outcome
        assert _determine_mls_outcome("Inter Miami CF", "Moneyline",
                                      "Inter Miami CF", "Orlando City SC",
                                      1.0, 1.0) == "LOST"


class TestTiePnl:
    def test_cheap_buy_profits_on_a_tie(self):
        """The real Oct 11 bet: 2 contracts at $0.3727, $0.79 total."""
        assert _tie_pnl({"num_contracts": 2, "contract_price": 0.3727}, 0.79) == 0.21

    def test_expensive_buy_loses_on_a_tie(self):
        assert _tie_pnl({"num_contracts": 2, "contract_price": 0.60}, 1.20) == -0.20

    def test_exactly_fifty_cents_is_flat(self):
        assert _tie_pnl({"num_contracts": 4, "contract_price": 0.50}, 2.00) == 0.0

    def test_contract_count_derived_when_absent(self):
        assert _tie_pnl({"contract_price": 0.40}, 0.80) == pytest.approx(0.20)

    def test_unknown_size_falls_back_to_neutral(self):
        """Safer to book nothing than to invent a payout."""
        assert _tie_pnl({}, 0.79) == 0.0


class TestNoLeaks:
    """LOST_TIE must never escape as an unrecognised result."""

    def test_shadow_log_maps_it_to_a_loss(self):
        src = open("src/state/shadow_log.py").read()
        assert '"LOST_TIE": "loss"' in src

    def test_results_snapshot_normalises_it(self):
        assert 'result == "LOST_TIE"' in open("src/data/results_snapshot.py").read()

    def test_parlay_leg_counts_as_lost(self):
        src = open("src/data/outcome_checker.py").read()
        assert '"LOST" if r == "LOST_TIE" else r' in src

    def test_watchlist_settler_normalises_it(self):
        src = open("src/data/outcome_checker.py").read()
        assert src.count('result = "LOST"      # NFL tie') >= 1

    def test_stored_result_is_LOST_not_the_marker(self):
        """History rows must carry LOST so every existing consumer works."""
        src = open("src/data/outcome_checker.py").read()
        block = src.split('if result == "LOST_TIE":')[1][:400]
        assert 'result      = "LOST"' in block


class TestHistoryWasCorrected:
    def test_no_nfl_moneyline_pushes_remain(self):
        import json
        h = json.load(open("state/history.json"))
        rows = [r for r in (h if isinstance(h, list) else []) if isinstance(r, dict)]
        bad = [r for r in rows
               if r.get("sport") == "NFL" and r.get("bet_type") == "Moneyline"
               and str(r.get("result", "")).upper() == "PUSH"]
        assert not bad, f"{len(bad)} NFL moneyline pushes still booked"

    def test_the_giants_row_reads_correctly(self):
        import json
        h = json.load(open("state/history.json"))
        rows = [r for r in (h if isinstance(h, list) else []) if isinstance(r, dict)]
        row = next((r for r in rows if r.get("date") == "2026-10-11"
                    and "Giants" in str(r.get("pick"))
                    and r.get("bet_type") == "Moneyline"), None)
        assert row, "the corrected row is missing"
        assert row["result"] == "LOST"
        assert row["actual_pnl"] == 0.21
        assert row.get("tie_settled") is True
