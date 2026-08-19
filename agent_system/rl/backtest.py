"""Backtesting and benchmark comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FeatureEngine
from agent_system.rl.environment import ACTION_NAMES, TradingEnvironment

logger = get_logger(__name__)


def buy_and_hold_return(ticks: Sequence[MarketTick]) -> float:
    """Benchmark return in percent.

    Any strategy must be compared against simply holding the asset. Beating
    cash is easy in a bull market; beating buy-and-hold is the real test.
    """
    if len(ticks) < 2:
        return 0.0
    first = ticks[0].price
    last = ticks[-1].price
    if first <= 0:
        return 0.0
    return (last / first - 1.0) * 100.0


def run_backtest(
    agent: Any,
    ticks: Sequence[MarketTick],
    config: Settings = settings,
    save_report: bool = True,
    label: str = "backtest",
    feature_engine: FeatureEngine | None = None,
) -> dict[str, Any]:
    """Replay ``agent`` over ``ticks`` and produce a full report."""
    if len(ticks) < 64:
        raise ValueError(f"Need at least 64 ticks to backtest, got {len(ticks)}")

    env = TradingEnvironment(ticks, config=config, feature_engine=feature_engine)
    observation, _ = env.reset()

    total_reward = 0.0
    actions: list[int] = []

    while True:
        action = agent.act(observation, deterministic=True)
        actions.append(int(action))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    metrics = env.metrics()
    benchmark = buy_and_hold_return(ticks)

    report = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": config.symbol,
        "ticks": len(ticks),
        "metrics": metrics,
        "benchmark_return_pct": round(benchmark, 4),
        # Alpha is the only number that says whether the model added
        # anything over doing nothing clever.
        "alpha_pct": round(metrics["return_pct"] - benchmark, 4),
        "total_reward": round(total_reward, 4),
        "round_trip_cost_pct": round(config.round_trip_cost_fraction * 100, 4),
        "actions": {
            name: actions.count(value) for value, name in ACTION_NAMES.items()
        },
        "equity_curve": [round(value, 4) for value in env.equity_curve],
        "trades": [trade.as_dict() for trade in env.closed_trades[-100:]],
    }

    if save_report:
        directory = Path(config.log_dir) / "backtests"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = directory / f"{label}-{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Backtest report written to %s", path)
        report["report_path"] = str(path)

    return report


def compare_agents(
    agents: dict[str, Any],
    ticks: Sequence[MarketTick],
    config: Settings = settings,
) -> dict[str, Any]:
    """Backtest several policies over identical data."""
    feature_engine = FeatureEngine()
    results: dict[str, Any] = {}
    for name, agent in agents.items():
        try:
            report = run_backtest(
                agent, ticks, config, save_report=False, label=name, feature_engine=feature_engine
            )
            results[name] = {
                "return_pct": report["metrics"]["return_pct"],
                "alpha_pct": report["alpha_pct"],
                "trades": report["metrics"]["trades"],
                "win_rate": report["metrics"]["win_rate"],
                "profit_factor": report["metrics"]["profit_factor"],
                "drawdown": report["metrics"]["drawdown"],
                "trades_per_100_steps": report["metrics"]["trades_per_100_steps"],
            }
        except (ValueError, RuntimeError) as exc:
            results[name] = {"error": str(exc)}
    return {
        "benchmark_return_pct": round(buy_and_hold_return(ticks), 4),
        "agents": results,
    }
