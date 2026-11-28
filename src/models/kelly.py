"""Kelly Criterion adapted for Robinhood's contract-based betting system."""
from dataclasses import dataclass
from src.config import (DAILY_BUDGET, KELLY_FRACTION, ROBINHOOD_COMMISSION,
                        MAX_DAILY_EXPOSURE_PCT)


@dataclass
class BetSizing:
    dollar_allocation: float
    num_contracts: int
    contract_price: float
    total_cost: float          # including commission
    profit_if_win: float
    loss_if_lose: float
    expected_value: float
    kelly_fraction: float


def robinhood_kelly(true_prob: float, contract_price: float, budget: float = DAILY_BUDGET) -> BetSizing:
    """
    Calculates fractional Kelly bet size for Robinhood contracts.

    contract_price: cost per contract in dollars (e.g. 0.65 for 65-cent contract)
    true_prob: our model's estimated win probability (0–1)

    Robinhood contract math:
      - Pay: contract_price + COMMISSION per contract
      - Win: receive $1.00 per contract
      - Net profit per contract if win: 1 - contract_price - COMMISSION
      - Net loss per contract if lose: contract_price + COMMISSION
    """
    cost_per = contract_price + ROBINHOOD_COMMISSION
    net_win = 1.0 - cost_per
    net_loss = cost_per

    if net_win <= 0 or true_prob <= 0:
        return BetSizing(0, 0, contract_price, 0, 0, 0, 0, 0)

    # b = net profit / net loss (odds ratio)
    b = net_win / net_loss
    q = 1.0 - true_prob

    # Full Kelly fraction of bankroll
    full_kelly = (b * true_prob - q) / b

    # Use fractional Kelly
    frac_kelly = max(0.0, full_kelly * KELLY_FRACTION)

    dollar_allocation = round(frac_kelly * budget, 2)
    num_contracts = int(dollar_allocation / cost_per)
    actual_cost = round(num_contracts * cost_per, 2)
    profit_if_win = round(num_contracts * net_win, 2)
    loss_if_lose = actual_cost

    ev = round(num_contracts * (true_prob * net_win - q * net_loss), 2)

    return BetSizing(
        dollar_allocation=dollar_allocation,
        num_contracts=num_contracts,
        contract_price=contract_price,
        total_cost=actual_cost,
        profit_if_win=profit_if_win,
        loss_if_lose=loss_if_lose,
        expected_value=ev,
        kelly_fraction=round(frac_kelly, 4),
    )


def parlay_kelly(true_prob: float, contract_price: float, budget: float = DAILY_BUDGET) -> BetSizing:
    """Same as robinhood_kelly but uses half the normal Kelly fraction (parlays have higher variance)."""
    sizing = robinhood_kelly(true_prob, contract_price, budget)
    # Halve the allocation for parlays
    reduced_alloc = sizing.dollar_allocation * 0.5
    cost_per = contract_price + ROBINHOOD_COMMISSION
    num_contracts = int(reduced_alloc / cost_per) if cost_per > 0 else 0
    net_win = 1.0 - cost_per
    net_loss = cost_per
    q = 1 - true_prob
    return BetSizing(
        dollar_allocation=round(reduced_alloc, 2),
        num_contracts=num_contracts,
        contract_price=contract_price,
        total_cost=round(num_contracts * cost_per, 2),
        profit_if_win=round(num_contracts * net_win, 2),
        loss_if_lose=round(num_contracts * cost_per, 2),
        expected_value=round(num_contracts * (true_prob * net_win - q * net_loss), 2),
        kelly_fraction=round(sizing.kelly_fraction * 0.5, 4),
    )


# ---------------------------------------------------------------------------
# Portfolio-level exposure cap  (Nov 27 2026)
#
# ⚠️ THE BUG THIS FIXES: robinhood_kelly() is a PER-BET function. Kelly's
# fraction-of-bankroll is only a fraction of the bankroll if you place ONE bet.
# main.py calls it once per pick with the FULL bankroll as `budget`, and the
# ~5 singles + 2 parlays that reach the money card are then summed — so the
# DAY was never bounded by anything. Measured over 207 betting days: median
# day staked 36% of bankroll, mean 58%, p90 163%, max 204%, and 39 days
# staked more than the entire bankroll. The bankroll ran $23 (Nov 7) -> ~$95
# (Nov 18) -> the $10 floor (Nov 22) while the win rate barely moved.
#
# These helpers work on the SERIALISED dicts (bet_to_dict / parlay_to_dict)
# rather than on BetSizing objects, on purpose: singles and parlays reach the
# card through two different serialisers but carry the same money fields, and
# a dict-level cap is therefore the ONE chokepoint that cannot be applied to
# "N-1 of N call sites". Every money field is linear in num_contracts, so
# scaling is exact.
# ---------------------------------------------------------------------------

#: Money fields on a serialised bet/parlay dict that scale linearly with
#: num_contracts. total_cost is the one the cap is measured against.
_LINEAR_MONEY_FIELDS = ("total_cost", "profit_if_win", "loss_if_lose",
                        "expected_value", "dollar_allocation")


def scale_sizing(sz: BetSizing, num_contracts: int) -> BetSizing:
    """Rebuild a BetSizing at a smaller contract count.

    Never scales UP: the Kelly result is the ceiling, the cap only trims.
    """
    if sz.num_contracts <= 0 or num_contracts >= sz.num_contracts:
        return sz
    cost_per = sz.contract_price + ROBINHOOD_COMMISSION
    if num_contracts <= 0:
        return BetSizing(0.0, 0, sz.contract_price, 0.0, 0.0, 0.0, 0.0, 0.0)
    f = num_contracts / sz.num_contracts
    return BetSizing(
        dollar_allocation=round(sz.dollar_allocation * f, 2),
        num_contracts=num_contracts,
        contract_price=sz.contract_price,
        total_cost=round(num_contracts * cost_per, 2),
        profit_if_win=round(num_contracts * (1.0 - cost_per), 2),
        loss_if_lose=round(num_contracts * cost_per, 2),
        expected_value=round(sz.expected_value * f, 2),
        kelly_fraction=round(sz.kelly_fraction * f, 4),
    )


def _row_cost_per(row: dict) -> float:
    """Per-contract cost implied by a serialised row.

    Prefers the row's own total_cost/num_contracts (exact, and correct for
    parlays whose price is a product of leg prices) and falls back to the
    quoted contract price plus commission.
    """
    n = int(row.get("num_contracts") or 0)
    tc = float(row.get("total_cost") or 0.0)
    if n > 0 and tc > 0:
        return tc / n
    cents = row.get("contract_price_cents")
    if cents:
        return float(cents) / 100.0 + ROBINHOOD_COMMISSION
    return float(row.get("contract_price") or 0.0) + ROBINHOOD_COMMISSION


def _row_rates(row: dict) -> dict:
    """Per-CONTRACT money rates for a serialised row, captured BEFORE any trim.

    ⚠️ Rates must be read once, up front. Deriving them from the row's current
    num_contracts would make a row that floors to zero unrecoverable (0/0), and
    the top-up pass below exists precisely to hand capacity back to such rows.
    expected_value per contract cannot be recomputed without the model
    probability, so it is carried as a rate rather than recalculated.
    """
    n = int(row.get("num_contracts") or 0)
    cost_per = _row_cost_per(row)
    rates = {"cost": cost_per, "win": max(0.0, 1.0 - cost_per)}
    for key in ("expected_value", "dollar_allocation"):
        v = row.get(key)
        try:
            rates[key] = (float(v) / n) if (n > 0 and v is not None) else 0.0
        except (TypeError, ValueError):
            rates[key] = 0.0
    return rates


def _set_row_contracts(row: dict, n: int, rates: dict) -> None:
    """Rewrite a row's money fields for `n` contracts (all linear in n)."""
    n = max(0, int(n))
    row["num_contracts"] = n
    row["total_cost"] = round(n * rates["cost"], 2)
    if "profit_if_win" in row:
        row["profit_if_win"] = round(n * rates["win"], 2)
    if "loss_if_lose" in row:
        row["loss_if_lose"] = row["total_cost"]
    for key in ("expected_value", "dollar_allocation"):
        if key in row:
            row[key] = round(n * rates[key], 2)


def cap_total_exposure(rows, bankroll: float,
                       cap_pct: float = MAX_DAILY_EXPOSURE_PCT) -> dict:
    """Trim `rows` in place so their summed total_cost fits the daily cap.

    `rows` are serialised bet/parlay dicts, IN PRIORITY ORDER (best first) —
    main.py passes the card's own ranking, so when capacity runs out it is the
    lowest-ranked bet that loses contracts.

    IDEMPOTENT: a card already under the cap is returned untouched, which is
    what makes it safe to apply both before state is written and again after
    merge_picks() mixes fresh picks with cached ones.

    ⚠️ The cap can legitimately zero the whole card. At the $10 bankroll floor
    a 10% cap is $1.00, which does not buy one 60c contract plus commission.
    That is the cap working, not a failure: the ledger says the money is gone.
    The report shows $0 stakes and main.py logs it loudly.

    Returns a report dict (never raises on empty input).
    """
    rows = [r for r in (rows or []) if isinstance(r, dict)]
    limit = round(max(0.0, float(bankroll)) * float(cap_pct), 2)
    before = round(sum(float(r.get("total_cost") or 0.0) for r in rows), 2)
    rep = {"limit": limit, "before": before, "after": before,
           "capped": False, "bankroll": round(float(bankroll), 2),
           "cap_pct": cap_pct, "zeroed": 0, "rows": len(rows)}
    if before <= limit or before <= 0:
        return rep

    rates = [_row_rates(r) for r in rows]
    # Kelly's own count is each pick's ceiling — the cap trims, it never sizes
    # a bet UP because other picks were zeroed. Captured before any mutation.
    ceilings = [int(r.get("num_contracts") or 0) for r in rows]

    # Proportional trim. int() floors, so this always lands at or under the
    # limit — never over it.
    f = limit / before
    for r, rt in zip(rows, rates):
        _set_row_contracts(r, int(int(r.get("num_contracts") or 0) * f), rt)

    # Flooring is lumpy and can leave real capacity unused (five picks that
    # each floor to zero spend nothing at all). Hand the slack back down the
    # ranking one contract at a time, best-ranked first, never exceeding the
    # pick's own Kelly count.
    def _spent():
        return round(sum(float(r.get("total_cost") or 0.0) for r in rows), 2)

    progressed = True
    while progressed:
        progressed = False
        for r, rt, ceil in zip(rows, rates, ceilings):
            if rt["cost"] <= 0 or int(r.get("num_contracts") or 0) >= ceil:
                continue
            if _spent() + rt["cost"] <= limit + 1e-9:
                _set_row_contracts(r, int(r.get("num_contracts") or 0) + 1, rt)
                progressed = True

    rep["after"] = _spent()
    rep["capped"] = True
    rep["zeroed"] = sum(1 for r in rows if int(r.get("num_contracts") or 0) <= 0)
    return rep


def has_positive_ev(true_prob: float, contract_price: float) -> bool:
    cost_per = contract_price + ROBINHOOD_COMMISSION
    return true_prob > cost_per
