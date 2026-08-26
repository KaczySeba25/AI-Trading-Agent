"""Reward calculation for the simulated trading environment."""

from __future__ import annotations


def calculate_reward(
    realized_pnl: float,
    unrealized_pnl_delta: float,
    fees: float,
    time_penalty: float,
) -> float:
    profit_component = max(0.0, realized_pnl + unrealized_pnl_delta) * 10.0
    loss_component = abs(min(0.0, realized_pnl + unrealized_pnl_delta)) * 2.0
    return profit_component - loss_component - fees - time_penalty
