"""Backtesting with a buy-and-hold benchmark.

A strategy that returns +5% while the market returned +20% is a losing strategy.
Every backtest therefore reports the benchmark alongside the agent, so results
cannot be read out of context.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.rl.environment import ACTION_NAMES, TradingEnvironment

logger = get_logger(__name__)


def buy_and_hold_return(ticks: Sequence[MarketTick]) -> float:
    """Benchmark return in percent."""
    if len(ticks) < 2 or ticks[0].price <= 0:
        return 0.0
    return (ticks[-1].price / ticks[0].price - 1.0) * 100.0


def run_backtest(
    agent,
    ticks: Sequence[MarketTick],
    config: Settings = settings,
    save_report: bool = True,
    label: str = "backtest",
) -> dict[str, Any]:
    """Run a deterministic backtest and return metrics plus the equity curve."""
    env = TradingEnvironment(ticks, config=config)
    observation, _ = env.reset()

    action_counts = {name: 0 for name in ACTION_NAMES.values()}
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(observation, deterministic=True)
        action_counts[ACTION_NAMES.get(int(action), "hold")] += 1
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    metrics = env.metrics()
    benchmark = buy_and_hold_return(ticks)

    report = {
        "label": label,
        "timestamp": time.time(),
        "symbol": config.symbol,
        "ticks": len(ticks),
        "initial_capital": config.initial_capital,
        "metrics": metrics,
        "benchmark_return_pct": benchmark,
        "alpha_pct": metrics["return_pct"] - benchmark,
        "actions": action_counts,
        "total_reward": total_reward,
        "equity_curve": [round(value, 4) for value in env.equity_curve],
        "trades": env.trades[-100:],
    }

    logger.info(
        "%s | return %+.2f%% vs benchmark %+.2f%% (alpha %+.2f%%) | "
        "trades %d | win rate %.1f%% | PF %.2f | dd %.2f%% | Sharpe %.2f",
        label,
        metrics["return_pct"],
        benchmark,
        report["alpha_pct"],
        int(metrics["closed_trades"]),
        metrics["win_rate"] * 100,
        metrics["profit_factor"],
        metrics["drawdown"] * 100,
        metrics["sharpe"],
    )

    if save_report:
        config.ensure_directories()
        path = Path(config.data_dir) / f"{label}_{int(time.time())}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(path)

    return report


def compare_agents(
    agents: dict[str, Any],
    ticks: Sequence[MarketTick],
    config: Settings = settings,
) -> dict[str, Any]:
    """Backtest several agents on identical data for a fair comparison."""
    results = {
        name: run_backtest(agent, ticks, config, save_report=False, label=name)
        for name, agent in agents.items()
    }
    ranking = sorted(
        results.items(), key=lambda item: item[1]["metrics"]["return_pct"], reverse=True
    )
    return {
        "results": results,
        "ranking": [name for name, _ in ranking],
        "benchmark_return_pct": buy_and_hold_return(ticks),
    }
