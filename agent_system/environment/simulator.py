"""Utilities for quick environment validation runs."""

from __future__ import annotations

from agent_system.data.data_buffer import MarketTick
from agent_system.environment.trading_env import TradingAction, TradingEnvironment


def run_smoke_simulation(ticks: list[MarketTick]) -> dict[str, float]:
    env = TradingEnvironment(ticks)
    env.reset()
    actions = [TradingAction.LONG, TradingAction.HOLD, TradingAction.CLOSE]
    done = False
    total_reward = 0.0
    step_count = 0
    info: dict[str, float] = {}
    while not done:
        action = actions[step_count] if step_count < len(actions) else TradingAction.HOLD
        _, reward, done, info = env.step(int(action))
        total_reward += reward
        step_count += 1
    info["total_reward"] = total_reward
    info["steps"] = float(step_count)
    return info
