"""The 24/7 learning loop.

One cycle:

1. download fresh history,
2. train a **candidate** on a walk-forward window,
3. evaluate it on unseen newer data,
4. promote it to champion only if it clears the gates,
5. trade the champion on live prices with virtual money,
6. persist everything, sleep, repeat.

The champion/challenger split is what makes unattended running safe: a bad
training cycle produces a rejected candidate, not a drained account.
"""

from __future__ import annotations

import asyncio
import json
import signal
import time
from pathlib import Path
from typing import Any

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.historical_client import HistoricalClient, load_ticks_from_csv
from agent_system.execution.paper_broker import PaperBroker
from agent_system.rl.agent import create_agent, load_agent
from agent_system.rl.environment import (
    ACTION_CLOSE,
    ACTION_LONG,
    ACTION_SHORT,
    TradingEnvironment,
)
from agent_system.rl.history_trainer import evaluate_policy, train_on_history
from agent_system.rl.memory import AgentMemory
from agent_system.rl.model_evaluator import decide_model_promotion

logger = get_logger(__name__)


class AutonomousTrader:
    """Continuous learn -> validate -> promote -> paper-trade loop."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        config.ensure_directories()
        self.memory = AgentMemory(config)
        self.broker = PaperBroker(config)
        self.client = HistoricalClient(config)
        self._stop = False

    def request_stop(self, *_: Any) -> None:
        """Ask the loop to finish the current cycle and shut down cleanly."""
        logger.info("Shutdown requested - finishing current cycle")
        self._stop = True

    def _load_history(self, interval: str, days: int, max_rows: int) -> list[MarketTick]:
        """Fetch history, falling back to the bundled CSV when offline."""
        try:
            ticks = self.client.fetch_klines(interval=interval, days=days, max_rows=max_rows)
            if len(ticks) >= 200:
                return ticks
            logger.warning("Only %d ticks from API - trying CSV fallback", len(ticks))
        except Exception as exc:  # noqa: BLE001 - offline must not kill the loop
            logger.warning("History download failed (%s) - trying CSV fallback", exc)

        csv_path = Path(self.config.data_dir) / f"{self.config.symbol}.csv"
        if csv_path.exists():
            ticks = load_ticks_from_csv(csv_path)
            logger.info("Loaded %d ticks from %s", len(ticks), csv_path.name)
            return ticks
        return []

    def _champion(self):
        if self.memory.has_champion():
            return load_agent(self.memory.champion_path, self.config)
        return None

    def run_learning_cycle(
        self,
        cycle: int,
        cycles: int,
        ticks: list[MarketTick],
        train_steps: int,
    ) -> dict[str, Any]:
        """Train a candidate and promote it if it beats the champion."""
        result = train_on_history(
            ticks, cycle=cycle, cycles=cycles, train_steps=train_steps, config=self.config
        )
        candidate = result["agent"]
        validation = result["validation_metrics"]

        candidate.save(self.memory.candidate_path)

        decision = decide_model_promotion(self.memory.champion_metrics(), validation)
        if decision.accepted:
            candidate.save(self.memory.champion_path)
            logger.info(
                "PROMOTED candidate (%s): return %.2f%% | PF %.2f | dd %.2f%%",
                decision.reason,
                validation["return_pct"],
                validation["profit_factor"],
                validation["drawdown"] * 100,
            )
        else:
            logger.info(
                "Candidate rejected (%s): return %.2f%% | trades %d",
                decision.reason, validation["return_pct"], int(validation["trades"]),
            )

        self.memory.record_cycle(
            cycle=cycle,
            mode="history_training",
            accepted=decision.accepted,
            reason=decision.reason,
            metrics=validation,
        )
        self.memory.persist()

        return {
            "cycle": cycle,
            "accepted": decision.accepted,
            "reason": decision.reason,
            "train_metrics": result["train_metrics"],
            "validation_metrics": validation,
        }

    def run_paper_session(self, ticks: list[MarketTick], max_steps: int = 500) -> dict[str, Any]:
        """Trade the champion over recent ticks with virtual money.

        Before the first promotion there is no champion, so we fall back to the
        heuristic baseline. That keeps paper mode useful on a fresh install and
        gives the loop a reference account to compare learned models against.
        """
        agent = self._champion()
        policy = "champion"
        if agent is None:
            from agent_system.rl.agent import HeuristicAgent

            agent = HeuristicAgent(self.config)
            policy = "heuristic_baseline"
            logger.info("No champion yet - paper trading the heuristic baseline")

        window = self.config.observation_window
        if len(ticks) < window + 5:
            return {"status": "insufficient_data"}

        env = TradingEnvironment(ticks[-(max_steps + window + 2):], config=self.config)
        observation, _ = env.reset()
        actions = {"hold": 0, "long": 0, "short": 0, "close": 0}
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.act(observation, deterministic=True)
            price = env.current_price

            # Mirror the decision into the persistent paper account.
            if action == ACTION_LONG:
                self.broker.open_position("LONG", price)
                actions["long"] += 1
            elif action == ACTION_SHORT:
                self.broker.open_position("SHORT", price)
                actions["short"] += 1
            elif action == ACTION_CLOSE:
                self.broker.close_position(price, "signal")
                actions["close"] += 1
            else:
                actions["hold"] += 1

            self.broker.mark_to_market(price)
            previous_observation = observation
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            # Store the real transition so learning survives restarts.
            self.memory.replay.push(previous_observation, action, reward, observation, done)

        if self.broker.positions.has_position(self.config.symbol):
            self.broker.close_position(env.current_price, "session_end")

        self.broker.save()
        self.memory.persist()

        return {
            "status": "completed",
            "policy": policy,
            "steps": steps,
            "actions": actions,
            "account": self.broker.snapshot(),
        }

    async def run(
        self,
        cycles: int = 0,
        interval: str = "1m",
        days: int = 30,
        max_rows: int = 20_000,
        train_steps: int | None = None,
        sleep_seconds: float = 300.0,
        history_cycles: int = 3,
    ) -> dict[str, Any]:
        """Main loop. ``cycles=0`` runs until stopped."""
        train_steps = train_steps or self.config.train_steps
        cycle = self.memory.next_cycle
        completed = 0

        logger.info(
            "Autonomous trader starting | symbol=%s | capital=%.2f | cycles=%s",
            self.config.symbol, self.config.initial_capital, cycles or "unlimited",
        )
        logger.info("Config: %s", json.dumps(self.config.describe(), default=str))

        while not self._stop and (cycles == 0 or completed < cycles):
            started = time.monotonic()
            try:
                ticks = self._load_history(interval, days, max_rows)
                if len(ticks) < self.config.observation_window * 4:
                    logger.error("Not enough market data (%d ticks) - retrying later", len(ticks))
                    await asyncio.sleep(min(sleep_seconds, 60))
                    continue

                learning = self.run_learning_cycle(cycle, history_cycles, ticks, train_steps)
                paper = self.run_paper_session(ticks)

                account = self.broker.snapshot()
                logger.info(
                    "Cycle %d done in %.1fs | promoted=%s | paper equity %.2f (%+.2f%%) | halted=%s",
                    cycle,
                    time.monotonic() - started,
                    learning["accepted"],
                    account["equity"],
                    account["return_pct"],
                    account["risk"]["halted"],
                )

                if account["risk"]["halted"]:
                    logger.error(
                        "Risk halt active (%s) - pausing trading, learning continues",
                        account["risk"]["halt_reason"],
                    )

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - a 24/7 loop must never die
                logger.exception("Cycle %d failed: %s", cycle, exc)

            cycle += 1
            completed += 1

            if not self._stop and (cycles == 0 or completed < cycles):
                await asyncio.sleep(sleep_seconds)

        self.memory.persist()
        self.broker.save()
        summary = {"cycles_completed": completed, "memory": self.memory.summary(),
                   "account": self.broker.snapshot()}
        logger.info("Autonomous trader stopped: %s", json.dumps(summary, default=str, indent=2))
        return summary


async def run_autonomous(config: Settings = settings, **kwargs: Any) -> dict[str, Any]:
    """Entry point with graceful SIGINT/SIGTERM handling."""
    trader = AutonomousTrader(config)

    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                loop.add_signal_handler(getattr(signal, signal_name), trader.request_stop)
            except NotImplementedError:  # Windows
                pass

    return await trader.run(**kwargs)
