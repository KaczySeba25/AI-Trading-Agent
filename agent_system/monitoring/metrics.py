"""Performance metrics for trading and model evaluation."""

from __future__ import annotations

import math


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    return sum(1 for pnl in trade_pnls if pnl > 0) / len(trade_pnls)


def profit_factor(trade_pnls: list[float]) -> float:
    gross_profit = sum(pnl for pnl in trade_pnls if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in trade_pnls if pnl < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak:
            worst = min(worst, (equity - peak) / peak)
    return abs(worst)


def sharpe_ratio(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(len(returns))


def summarize_performance(
    trade_pnls: list[float],
    equity_curve: list[float],
    returns: list[float],
) -> dict[str, float]:
    return {
        "win_rate": win_rate(trade_pnls),
        "profit_factor": profit_factor(trade_pnls),
        "drawdown": max_drawdown(equity_curve),
        "sharpe_ratio": sharpe_ratio(returns),
    }
