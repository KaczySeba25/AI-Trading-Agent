"""Command line entry point.

    python -m agent_system.main --mode status
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from agent_system.core.config import Settings, settings
from agent_system.core.exceptions import MarketDataError, TradingAgentError
from agent_system.core.logger import get_logger
from agent_system.data.data_buffer import MarketTick
from agent_system.data.historical_client import HistoricalClient, load_ticks_from_csv
from agent_system.execution.diagnostics import run_testnet_diagnostic
from agent_system.rl.agent import SB3_AVAILABLE, HeuristicAgent, load_agent
from agent_system.rl.autonomous_loop import AutonomousTrader, run_autonomous
from agent_system.rl.backtest import compare_agents, run_backtest
from agent_system.rl.history_trainer import train_on_history
from agent_system.rl.memory import AgentMemory

logger = get_logger(__name__)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _load_ticks(args: argparse.Namespace, config: Settings) -> list[MarketTick]:
    """Get price history: live API first, local CSV as a fallback.

    The fallback is not a nicety. Without network access -- a locked-down CI
    box, an offline laptop -- every mode that needs data would be unusable.
    """
    if not args.csv:
        try:
            client = HistoricalClient(config)
            ticks = client.fetch_klines(
                interval=args.interval, days=args.days, max_rows=args.max_rows
            )
            if len(ticks) >= 64:
                return ticks
            logger.warning("API returned only %d ticks; falling back to CSV", len(ticks))
        except (MarketDataError, TradingAgentError) as exc:
            logger.warning("Could not download klines (%s); falling back to CSV", exc)

    csv_path = Path(args.csv) if args.csv else Path(config.data_dir) / f"{config.symbol}.csv"
    return load_ticks_from_csv(csv_path, max_rows=args.max_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous Binance trading agent")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "status",
            "stream",
            "backtest",
            "train",
            "paper",
            "autonomous",
            "testnet-diagnostic",
        ],
    )
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--csv", default=None, help="Load ticks from this CSV instead of the API")
    parser.add_argument("--cycles", type=int, default=1, help="0 runs until interrupted")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--steps", type=int, default=500, help="Paper trading steps per cycle")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--reset", action="store_true", help="Reset the paper account first")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = settings
    config.ensure_directories()

    if args.mode == "status":
        memory = AgentMemory(config)
        _print(
            {
                "config": config.describe(),
                "sb3_available": SB3_AVAILABLE,
                "memory": memory.summary(),
            }
        )
        return 0

    if args.mode == "testnet-diagnostic":
        _print(run_testnet_diagnostic(config))
        return 0

    if args.mode == "stream":
        from agent_system.data.binance_ws import run_phase_one

        try:
            asyncio.run(run_phase_one())
        except KeyboardInterrupt:
            logger.info("Stream stopped by user")
        return 0

    ticks = _load_ticks(args, config)
    logger.info("Loaded %d ticks", len(ticks))

    if args.mode == "backtest":
        memory = AgentMemory(config)
        champion = load_agent(memory.champion_path, config)
        agents: dict[str, Any] = {"heuristic": HeuristicAgent(config)}
        if champion is not None:
            agents["champion"] = champion
        comparison = compare_agents(agents, ticks, config)
        report = run_backtest(
            agents.get("champion", agents["heuristic"]), ticks, config, label="cli"
        )
        _print({"comparison": comparison, "report": report})
        return 0

    if args.mode == "train":
        result = train_on_history(ticks, config=config, steps=args.train_steps)
        memory = AgentMemory(config)
        path = result["agent"].save(memory.candidate_path)
        _print(
            {
                "candidate": str(path),
                "train_metrics": result["train_metrics"],
                "validation_metrics": result["validation_metrics"],
            }
        )
        return 0

    if args.mode == "paper":
        trader = AutonomousTrader(config)
        if args.reset:
            trader.broker.reset()
        _print(trader.run_paper_session(ticks, max_steps=args.steps))
        return 0

    if args.mode == "autonomous":
        result = run_autonomous(
            ticks,
            config=config,
            cycles=args.cycles,
            train_steps=args.train_steps,
            paper_steps=args.steps,
            sleep_seconds=args.sleep_seconds,
        )
        _print(result)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
