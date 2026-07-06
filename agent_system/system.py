"""End-to-end system orchestration."""

from __future__ import annotations

import json
import time
from pathlib import Path

from stable_baselines3 import PPO

from agent_system.core.logger import get_logger
from agent_system.data.binance_ws import BinanceMarketStream
from agent_system.data.data_buffer import MarketTick
from agent_system.environment.trading_env import TradingEnvironment
from agent_system.execution.policy_guard import PolicyGuard
from agent_system.rl.ppo_model import GymTradingEnv


logger = get_logger(__name__)


class TradingSystem:
    def __init__(self, model_path: str = "models/ppo_latest.zip", report_dir: str = "data") -> None:
        self.model_path = Path(model_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run_paper_replay(self, ticks: list[MarketTick]) -> dict[str, object]:
        if len(ticks) < 2:
            raise ValueError("Paper replay requires at least two ticks")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        gym_env = GymTradingEnv(ticks)
        model = PPO.load(str(self.model_path), env=gym_env)
        sim_env = TradingEnvironment(ticks)
        guard = PolicyGuard(max_hold_steps=60)
        state = sim_env.reset()
        done = False
        actions: list[int] = []
        raw_actions: list[int] = []
        rewards: list[float] = []
        info: dict[str, float] = {}

        while not done:
            action, _ = model.predict(state, deterministic=True)
            raw_action = int(action)
            current_tick = ticks[min(sim_env.index, len(ticks) - 1)]
            guarded_action = guard.filter_action(raw_action, current_tick.price, sim_env.index)
            raw_actions.append(raw_action)
            action = guarded_action
            state, reward, done, info = sim_env.step(int(action))
            guard.observe_executed_action(int(action))
            actions.append(int(action))
            rewards.append(float(reward))

        report: dict[str, object] = {
            "timestamp": int(time.time()),
            "mode": "paper_replay",
            "ticks": len(ticks),
            "actions": {
                "hold": actions.count(0),
                "long": actions.count(1),
                "short": actions.count(2),
                "close": actions.count(3),
            },
            "raw_actions": {
                "hold": raw_actions.count(0),
                "long": raw_actions.count(1),
                "short": raw_actions.count(2),
                "close": raw_actions.count(3),
            },
            "total_reward": sum(rewards),
            "final_info": info,
        }
        report_path = self.report_dir / f"paper_replay_{report['timestamp']}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Paper replay complete: %s", report)
        return report

    async def run_live_paper(
        self,
        min_ticks: int = 120,
        timeout_seconds: float = 45.0,
    ) -> dict[str, object]:
        stream = BinanceMarketStream()
        ticks = await stream.collect_ticks(min_ticks=min_ticks, timeout_seconds=timeout_seconds)
        if len(ticks) < 2:
            raise RuntimeError("Not enough live ticks collected for paper trading")

        report = self.run_paper_replay(ticks)
        report["mode"] = "live_paper"
        report["stream_stats"] = {
            "messages_received": stream.stats.messages_received,
            "ticks_emitted": stream.stats.ticks_emitted,
            "malformed_messages": stream.stats.malformed_messages,
            "reconnects": stream.stats.reconnects,
            "avg_latency_ms": stream.stats.avg_latency_ms,
        }
        report_path = self.report_dir / f"live_paper_{report['timestamp']}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Live paper run complete: %s", report)
        return report


def synthetic_ticks(count: int = 500) -> list[MarketTick]:
    now = int(time.time() * 1000)
    ticks: list[MarketTick] = []
    for index in range(count):
        trend = index * 0.006
        cycle = ((index % 40) - 20) * 0.06
        price = 100.0 + trend + cycle
        ticks.append(
            MarketTick(
                timestamp=now + index * 1000,
                price=price,
                volume=1.0 + (index % 5) * 0.1,
                bid=price - 0.01,
                ask=price + 0.01,
            )
        )
    return ticks
