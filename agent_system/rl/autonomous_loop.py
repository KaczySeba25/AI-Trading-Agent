"""The 24/7 autonomous loop.

One cycle is: pick a slice of history, train a candidate, validate it, decide
whether it beats the champion, then paper-trade with whichever model won.
Everything that matters is written to disk at the end of every cycle, so the
process can be killed at any moment and resume where it left off.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any, Sequence

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.feature_engineering import FeatureEngine
from agent_system.execution.paper_broker import PaperBroker
from agent_system.execution.risk_manager import RiskManager
from agent_system.rl.agent import HeuristicAgent, create_agent, load_agent
from agent_system.rl.environment import CLOSE, LONG, SHORT, TradingEnvironment
from agent_system.rl.history_trainer import (
    evaluate_policy,
    select_training_window,
    train_on_history,
)
from agent_system.rl.memory import AgentMemory
from agent_system.rl.model_evaluator import decide_model_promotion

logger = get_logger(__name__)


class AutonomousTrader:
    """Continuous train / validate / promote / paper-trade loop."""

    def __init__(self, config: Settings = settings) -> None:
        config.ensure_directories()
        self.config = config
        self.memory = AgentMemory(config)
        self.broker = PaperBroker(config)
        self.risk = RiskManager(self.broker.equity, config)
        self.feature_engine = FeatureEngine()
        self._stop = False

    def request_stop(self) -> None:
        logger.info("Stop requested; will finish the current cycle and persist state")
        self._stop = True

    # ------------------------------------------------------------------
    def run_learning_cycle(
        self,
        ticks: Sequence[MarketTick],
        cycle: int,
        cycles: int,
        train_steps: int | None = None,
    ) -> dict[str, Any]:
        """Train one candidate and decide whether it becomes champion."""
        window = select_training_window(ticks, cycle, cycles)
        logger.info("Cycle %d/%d: training on %d ticks", cycle, cycles, len(window))

        # Warm-start from the champion so learning accumulates across cycles
        # instead of restarting from scratch every time.
        starting_agent = None
        if self.memory.has_champion:
            starting_agent = load_agent(self.memory.champion_path, self.config)

        result = train_on_history(
            window,
            config=self.config,
            agent=starting_agent,
            steps=train_steps,
        )
        agent = result["agent"]
        validation_metrics = result["validation_metrics"]

        candidate_path = agent.save(self.memory.candidate_path)

        champion_summary = self.memory.champion_metrics
        champion_payload = (
            {
                "equity": champion_summary.get("equity", 0.0),
                "trades": champion_summary.get("trades", 0),
                "metrics": champion_summary,
            }
            if champion_summary
            else None
        )
        candidate_payload = {
            "equity": validation_metrics.get("equity", 0.0),
            "trades": validation_metrics.get("trades", 0),
            "metrics": validation_metrics,
        }

        decision = decide_model_promotion(champion_payload, candidate_payload)
        if decision.accepted:
            promoted_path = agent.save(self.memory.champion_path)
            logger.info("Candidate promoted to champion: %s (%s)", promoted_path, decision.reason)
        else:
            logger.info("Candidate rejected: %s", decision.reason)

        self.memory.record_cycle(
            train_metrics=result["train_metrics"],
            validation_metrics=validation_metrics,
            promoted=decision.accepted,
            reason=decision.reason,
        )
        self.memory.persist()

        return {
            "cycle": cycle,
            "candidate_path": str(candidate_path),
            "promoted": decision.accepted,
            "reason": decision.reason,
            "train_metrics": result["train_metrics"],
            "validation_metrics": validation_metrics,
        }

    # ------------------------------------------------------------------
    def run_paper_session(
        self,
        ticks: Sequence[MarketTick],
        max_steps: int = 500,
    ) -> dict[str, Any]:
        """Paper-trade the current champion over ``ticks``.

        Runs against the persistent paper account, so equity carries over
        between sessions and a multi-day run tells a continuous story.
        """
        agent = load_agent(self.memory.champion_path, self.config)
        policy = "champion"
        if agent is None:
            # Cold start: trade the heuristic rather than sitting idle until
            # the first model is trained.
            agent = HeuristicAgent(self.config)
            policy = "heuristic"

        env = TradingEnvironment(ticks, config=self.config, feature_engine=self.feature_engine)
        observation, _ = env.reset()

        steps = 0
        for _ in range(max_steps):
            allowed, reason = self.risk.can_trade()
            action = agent.act(observation, deterministic=False)

            # Risk veto only blocks *opening* exposure. Closing must always be
            # permitted, otherwise a halt would trap an open position.
            if not allowed and action in (LONG, SHORT):
                action = CLOSE if env.position is not None else 0

            next_observation, reward, terminated, truncated, info = env.step(action)

            # Every transition is remembered, not just the profitable ones.
            self.memory.replay_buffer.push(
                observation, action, reward, next_observation, terminated or truncated
            )

            price = info["price"]
            self._mirror_to_broker(env, price)
            self.risk.update_equity(self.broker.equity)

            observation = next_observation
            steps += 1
            if terminated or truncated:
                break

        self.broker.save()
        self.memory.persist()

        snapshot = self.broker.snapshot()
        return {
            "policy": policy,
            "steps": steps,
            "env_metrics": env.metrics(),
            "account": snapshot,
            "risk": self.risk.snapshot(),
            "halted": self.risk.state.halted,
        }

    def _mirror_to_broker(self, env: TradingEnvironment, price: float) -> None:
        """Keep the persistent paper account in step with the environment."""
        env_side = 0 if env.position is None else env.position.side
        broker_side = 0 if self.broker.position is None else self.broker.position.side

        if env_side == broker_side:
            self.broker.mark_to_market(price)
            return

        if broker_side != 0:
            self.broker.close_position(price, "sync")
        if env_side != 0 and env.position is not None:
            self.broker.open_position(
                side=env_side,
                price=price,
                quantity=env.position.quantity,
                stop_loss=env.position.stop_loss,
                take_profit=env.position.take_profit,
            )
        self.broker.mark_to_market(price)

    # ------------------------------------------------------------------
    async def run(
        self,
        ticks: Sequence[MarketTick],
        cycles: int = 0,
        train_steps: int | None = None,
        paper_steps: int = 500,
        sleep_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Run ``cycles`` cycles, or forever when ``cycles`` is 0."""
        completed: list[dict[str, Any]] = []
        cycle = self.memory.next_cycle
        planned = cycles if cycles > 0 else max(cycle + 9, 10)

        while not self._stop:
            if cycles > 0 and len(completed) >= cycles:
                break

            try:
                learning = self.run_learning_cycle(ticks, cycle, planned, train_steps)
                paper = self.run_paper_session(ticks, paper_steps)
            except Exception as exc:  # noqa: BLE001
                # A 24/7 process must not die because one cycle failed. Log,
                # persist, back off, and try the next cycle.
                logger.exception("Cycle %d failed: %s", cycle, exc)
                self.memory.persist()
                await asyncio.sleep(max(sleep_seconds, 5.0))
                cycle += 1
                continue

            completed.append({"learning": learning, "paper": paper})
            logger.info(
                "Cycle %d done | promoted=%s | equity=%.2f | trades=%d",
                cycle,
                learning["promoted"],
                paper["account"]["equity"],
                paper["env_metrics"]["trades"],
            )

            cycle += 1
            if cycles > 0 and len(completed) >= cycles:
                break
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

        return {
            "cycles_run": len(completed),
            "memory": self.memory.summary(),
            "account": self.broker.snapshot(),
            "results": completed,
        }


def run_autonomous(
    ticks: Sequence[MarketTick],
    config: Settings = settings,
    cycles: int = 0,
    train_steps: int | None = None,
    paper_steps: int = 500,
    sleep_seconds: float = 0.0,
) -> dict[str, Any]:
    """Entry point with graceful shutdown on SIGINT/SIGTERM."""
    trader = AutonomousTrader(config)

    async def _main() -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, trader.request_stop)
            except (NotImplementedError, RuntimeError):
                # Windows and non-main threads do not support this.
                pass
        return await trader.run(
            ticks,
            cycles=cycles,
            train_steps=train_steps,
            paper_steps=paper_steps,
            sleep_seconds=sleep_seconds,
        )

    return asyncio.run(_main())
