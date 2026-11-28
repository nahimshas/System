"""
Portfolio-level exposure cap + quarter Kelly (Nov 27 2026).

The bug these pin: robinhood_kelly() is a PER-BET rule called once per pick
with the FULL bankroll as its budget, and the card sums ~5 singles + 2 parlays.
Nothing bounded the DAY — measured median 36% of bankroll staked per day, p90
163%, max 204%, 39 of 207 days over 100%.
"""
import pytest

from src.config import KELLY_FRACTION, MAX_DAILY_EXPOSURE_PCT
from src.models.kelly import (BetSizing, cap_total_exposure, scale_sizing,
                              robinhood_kelly)


def _row(n: int, cents: float = 60.0, ev: float = 0.05) -> dict:
    cost_per = cents / 100.0 + 0.02
    return dict(
        num_contracts=n,
        total_cost=round(n * cost_per, 2),
        profit_if_win=round(n * (1 - cost_per), 2),
        loss_if_lose=round(n * cost_per, 2),
        expected_value=round(n * ev, 2),
        dollar_allocation=round(n * cost_per, 2),
        contract_price_cents=float(cents),
    )


def _spent(rows) -> float:
    return round(sum(r["total_cost"] for r in rows), 2)


class TestConstants:
    def test_quarter_kelly_shipped(self):
        # Raising this requires a market that passed market_test.py + CLV >= 0
        # + positive ROI in both halves of a time split. None has.
        assert KELLY_FRACTION == 0.25

    def test_exposure_cap_is_a_real_bound(self):
        assert 0.0 < MAX_DAILY_EXPOSURE_PCT <= 0.25


class TestExposureCap:
    def test_card_over_cap_is_trimmed_to_the_limit(self):
        rows = [_row(20) for _ in range(5)]
        rep = cap_total_exposure(rows, 100.0, 0.10)
        assert rep["capped"] is True
        assert rep["before"] == 62.0
        assert _spent(rows) <= rep["limit"] + 1e-9
        assert rep["after"] == _spent(rows)

    def test_card_under_cap_is_untouched(self):
        rows = [_row(2)]
        before = [dict(r) for r in rows]
        rep = cap_total_exposure(rows, 100.0, 0.10)
        assert rep["capped"] is False
        assert rows == before

    def test_idempotent(self):
        """Applied at three call sites (fresh / final / code_only) — a second
        pass over an already-bounded card must be a no-op."""
        rows = [_row(20) for _ in range(5)]
        cap_total_exposure(rows, 100.0, 0.10)
        after_first = [dict(r) for r in rows]
        rep2 = cap_total_exposure(rows, 100.0, 0.10)
        assert rep2["capped"] is False
        assert rows == after_first

    def test_priority_order_is_respected(self):
        """Singles come first and in card order, so the lowest-ranked bet is
        the one that loses contracts."""
        rows = [_row(20) for _ in range(5)]
        cap_total_exposure(rows, 100.0, 0.10)
        counts = [r["num_contracts"] for r in rows]
        assert counts == sorted(counts, reverse=True)

    def test_never_sizes_a_bet_above_its_own_kelly_count(self):
        """Capacity freed by zeroing one pick must not inflate another past
        what Kelly asked for."""
        rows = [_row(1), _row(1), _row(40)]
        cap_total_exposure(rows, 100.0, 0.10)
        assert rows[0]["num_contracts"] <= 1
        assert rows[1]["num_contracts"] <= 1
        assert rows[2]["num_contracts"] <= 40

    def test_money_fields_stay_consistent_with_contract_count(self):
        rows = [_row(20) for _ in range(5)]
        cap_total_exposure(rows, 100.0, 0.10)
        for r in rows:
            n, cost_per = r["num_contracts"], 0.62
            assert r["total_cost"] == pytest.approx(round(n * cost_per, 2), abs=0.01)
            assert r["loss_if_lose"] == r["total_cost"]
            assert r["profit_if_win"] == pytest.approx(round(n * 0.38, 2), abs=0.01)
            if n == 0:
                assert r["expected_value"] == 0

    def test_bankroll_at_floor_can_zero_the_whole_card(self):
        """At the $10 bankroll floor a 10% cap is $1.00, which does not buy one
        60c contract plus commission. Zeroing is the cap working."""
        rows = [_row(20) for _ in range(5)]
        rep = cap_total_exposure(rows, 10.0, 0.10)
        assert _spent(rows) <= 1.0
        assert rep["zeroed"] >= 4

    def test_reclaims_slack_left_by_flooring(self):
        """Proportional flooring alone can spend nothing at all; the top-up
        pass must put the unused capacity back to work."""
        rows = [_row(3) for _ in range(5)]
        rep = cap_total_exposure(rows, 30.0, 0.10)
        assert rep["capped"] is True
        assert _spent(rows) > 0, "flooring swallowed the entire allowance"
        assert _spent(rows) <= rep["limit"] + 1e-9

    def test_parlay_rows_are_included(self):
        """Parlays were never even sized off the rolling bankroll
        (parlay_kelly defaults to DAILY_BUDGET), so they must be inside the
        same bound as the singles."""
        singles = [_row(10) for _ in range(5)]
        parlays = [dict(num_contracts=10, total_cost=3.5, profit_if_win=6.5,
                        expected_value=1.0, contract_price_cents=33.0)]
        rep = cap_total_exposure(singles + parlays, 100.0, 0.10)
        assert rep["capped"] is True
        assert _spent(singles + parlays) <= rep["limit"] + 1e-9

    def test_never_raises_on_degenerate_input(self):
        assert cap_total_exposure([], 100.0)["capped"] is False
        assert cap_total_exposure(None, 100.0)["capped"] is False
        assert cap_total_exposure([{}], 100.0)["capped"] is False
        assert cap_total_exposure([_row(5)], 0.0)["limit"] == 0.0


class TestScaleSizing:
    def test_trims_but_never_grows(self):
        sz = robinhood_kelly(0.60, 0.50, 100.0)
        assert sz.num_contracts > 2
        smaller = scale_sizing(sz, 2)
        assert smaller.num_contracts == 2
        assert smaller.total_cost < sz.total_cost
        # asking for more than Kelly allowed returns the original untouched
        assert scale_sizing(sz, sz.num_contracts + 10) is sz

    def test_zero_contracts_zeroes_every_money_field(self):
        sz = robinhood_kelly(0.60, 0.50, 100.0)
        z = scale_sizing(sz, 0)
        assert (z.num_contracts, z.total_cost, z.profit_if_win,
                z.expected_value, z.dollar_allocation) == (0, 0.0, 0.0, 0.0, 0.0)


class TestKellyStillWorks:
    def test_quarter_kelly_halves_the_old_stake(self):
        """Same inputs, half the old allocation — the whole point of the change."""
        sz = robinhood_kelly(0.60, 0.50, 100.0)
        # full Kelly here is (b*p - q)/b with b = 0.48/0.52 -> 0.1667 of
        # bankroll. Quarter Kelly stakes 4.17%; half Kelly staked 8.33%.
        assert sz.dollar_allocation == pytest.approx(4.17, abs=0.02)
        assert sz.kelly_fraction == pytest.approx(0.1667 * KELLY_FRACTION, abs=0.001)

    def test_no_bet_when_edge_is_negative(self):
        assert robinhood_kelly(0.40, 0.55, 100.0).num_contracts == 0
