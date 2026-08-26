"""Backtesting helpers for public historical market data."""

from __future__ import annotations

from agent_system.data.historical_client import HistoricalMarketDataClient
from agent_system.rl.trainer import train_ppo
from agent_system.system import TradingSystem


def run_public_history_training_and_backtest(
    interval: str = "1m",
    limit: int = 500,
    train_steps: int = 1024,
    online_cycle: bool = False,
) -> dict[str, object]:
    ticks = HistoricalMarketDataClient().fetch_klines(interval=interval, limit=limit)
    train_result = train_ppo(ticks, total_timesteps=train_steps)
    replay_result = TradingSystem().run_paper_replay(ticks)
    result: dict[str, object] = {
        "ticks": len(ticks),
        "interval": interval,
        "train": train_result,
        "paper_replay": replay_result,
    }
    if online_cycle:
        from agent_system.rl.online_loop import OnlineLearningLoop

        result["online"] = OnlineLearningLoop().run_cycle(
            ticks,
            collect_steps=min(len(ticks), 10_000),
            train_steps=train_steps,
        )
    return result
