"""Kelly Criterion adapted for Robinhood's contract-based betting system."""
from dataclasses import dataclass
from src.config import DAILY_BUDGET, KELLY_FRACTION, ROBINHOOD_COMMISSION


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


def has_positive_ev(true_prob: float, contract_price: float) -> bool:
    cost_per = contract_price + ROBINHOOD_COMMISSION
    return true_prob > cost_per
