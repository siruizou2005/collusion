"""Economic environment functions for the collusion simulation."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np


def compute_expected_quantity(price_i: float,
                              price_j: float,
                              alpha: float,
                              beta: float = 100.0,
                              a_i: float = 2.0,
                              a_j: float = 2.0,
                              a0: float = 0.0,
                              mu: float = 0.25,
                              market_factor: float = 1.0) -> Tuple[float, float]:
    """Compute expected quantity for a two-firm logit demand model."""
    if market_factor <= 0:
        raise ValueError("market_factor must be strictly positive")
    u_i = math.exp((a_i - price_i / alpha) / mu)
    u_j = math.exp((a_j - price_j / alpha) / mu)
    u0 = math.exp(a0 / mu)
    denom = u_i + u_j + u0
    q_i = market_factor * beta * u_i / denom
    q_j = market_factor * beta * u_j / denom
    return q_i, q_j


def compute_realised_quantity(expected_quantity: float,
                              noise_sigma: float = 0.0) -> float:
    """Apply multiplicative lognormal noise to the expected quantity."""
    if noise_sigma <= 0:
        return expected_quantity
    mu = -0.5 * noise_sigma * noise_sigma
    multiplier = np.random.lognormal(mean=mu, sigma=noise_sigma)
    return expected_quantity * multiplier


def compute_profit(price: float, quantity: float, alpha: float, cost: float) -> float:
    """Compute realised profit given price, quantity and cost."""
    return (price - alpha * cost) * quantity


def _profit_matrix(prices: np.ndarray,
                   alpha: float,
                   mu: float,
                   beta: float,
                   a_i: float,
                   a_j: float,
                   a0: float,
                   c: float) -> np.ndarray:
    """Return firm-i profit over all price pairs (p_i, p_j)."""
    price_i = prices[:, None]
    price_j = prices[None, :]
    u_i = np.exp((a_i - price_i / alpha) / mu)
    u_j = np.exp((a_j - price_j / alpha) / mu)
    u0 = math.exp(a0 / mu)
    q_i = beta * u_i / (u_i + u_j + u0)
    return (price_i - alpha * c) * q_i


@lru_cache(maxsize=128)
def _find_static_optima_cached(alpha: float,
                               mu: float,
                               beta: float,
                               a_i: float,
                               a0: float,
                               c: float,
                               price_min: float,
                               initial_price_max: float,
                               price_steps: int,
                               max_expansions: int) -> Tuple[float, float, float, float]:
    """Cached benchmark solver for the symmetric two-firm environment."""
    if price_steps < 2:
        raise ValueError("price_steps must be at least 2")

    search_max = initial_price_max
    best_result: Optional[Tuple[float, float, float, float]] = None

    for _ in range(max_expansions + 1):
        prices = np.linspace(price_min, search_max, price_steps)
        profit_i = _profit_matrix(prices, alpha, mu, beta, a_i, a_i, a0, c)

        # Nash benchmark: fixed point of the best-response correspondence.
        br_indices = profit_i.argmax(axis=0)
        br_prices = prices[br_indices]
        fixed_index = int(np.argmin(np.abs(br_prices - prices)))
        p_nash = float(prices[fixed_index])
        profit_nash = float(profit_i[br_indices[fixed_index], fixed_index])

        # Monopoly benchmark: maximize joint profit over both firms' prices.
        total_profit = profit_i + profit_i.T
        monopoly_index = np.unravel_index(int(np.argmax(total_profit)), total_profit.shape)
        monopoly_price_a = float(prices[monopoly_index[0]])
        monopoly_price_b = float(prices[monopoly_index[1]])
        p_monopoly = 0.5 * (monopoly_price_a + monopoly_price_b)
        profit_monopoly = 0.5 * float(total_profit[monopoly_index])

        best_result = (p_nash, profit_nash, p_monopoly, profit_monopoly)

        last_index = price_steps - 1
        hits_upper_bound = (
            fixed_index == last_index
            or br_indices[last_index] == last_index
            or monopoly_index[0] == last_index
            or monopoly_index[1] == last_index
        )
        if not hits_upper_bound:
            break
        search_max *= 2.0

    if best_result is None:
        raise RuntimeError("Failed to compute static benchmarks")
    return best_result


def find_static_optima(alpha: float,
                       mu: float = 0.25,
                       beta: float = 100.0,
                       a_i: float = 2.0,
                       a0: float = 0.0,
                       c: float = 1.0,
                       price_min: Optional[float] = None,
                       price_max: Optional[float] = None,
                       price_steps: int = 1201,
                       max_expansions: int = 6) -> Tuple[float, float, float, float]:
    """Find approximate Nash and monopoly benchmarks for a symmetric two-firm model.

    Returns
    -------
    Tuple[float, float, float, float]
        (p_nash, profit_nash, p_monopoly, profit_monopoly), where profits are
        per-firm profits at the respective benchmark prices.
    """
    if price_min is None:
        price_min = alpha * c
    if price_max is None:
        price_max = max(10.0, 3.0 * alpha * c)
    if price_max <= price_min:
        price_max = max(price_min + 1.0, price_min * 1.5)

    return _find_static_optima_cached(
        alpha,
        mu,
        beta,
        a_i,
        a0,
        c,
        float(price_min),
        float(price_max),
        price_steps,
        max_expansions,
    )
