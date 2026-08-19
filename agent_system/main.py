"""Command line entry point.

    python -m agent_system.main --mode <mode> [options]

Run ``--mode status`` first: it reports configuration and whether the agent has
a champion model, without touching the network.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from agent_system.core.config import settings
from agent_system.core.logger import get_logger
from agent_system.data.historical_client import HistoricalClient, load_ticks_from_csv
from agent_system.execution.diagnostics import run_testnet_diagnostic
from agent_system.rl.agent import SB3_AVAILABLE, create_agent, load_agent
from agent_system.rl.autonomous_loop import AutonomousTrader, run_autonomous
from agent_system.rl.backtest import run_backtest
from agent_system.rl.environment import TradingEnvironment
from agent_system.rl.memory import AgentMemory

logger = get_logger("agent_system.main")

MODES = (
    "status",
    "stream",
    "backtest",
    "train",
    "paper",
    "autonomous",
    "testnet-diagnostic",
)


def _load_ticks(args) -> list:
    """Load market data, preferring the API and falling back to CSV."""
    if args.csv:
        return load_ticks_from_csv(args.csv)

    try:
        client = HistoricalClient(settings)
        ticks = client.fetch_klines(interval=args.interval, days=args.days, max_rows=args.max_rows)
        if ticks:
            return ticks
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download history: %s", exc)

    fallback = Path(settings.data_dir) / f"{settings.symbol}.csv"
    if fallback.exists():
        logger.info("Falling back to %s", fallback)
        return load_ticks_from_csv(fallback)

    logger.error("No market data available. Pass --csv or check connectivity.")
    return []


def cmd_status() -> int:
    memory = AgentMemory(settings)
    payload = {
        "config": settings.describe(),
        "sb3_available": SB3_AVAILABLE,
        "memory": memory.summary(),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_stream(args) -> int:
    from agent_system.data.binance_ws import run_phase_one

    try:
        asyncio.run(run_phase_one())
    except KeyboardInterrupt:
        logger.info("Stream stopped")
    return 0


def cmd_backtest(args) -> int:
    ticks = _load_ticks(args)
    if len(ticks) < settings.observation_window + 5:
        return 1

    memory = AgentMemory(settings)
    if memory.has_champion():
        agent = load_agent(memory.champion_path, settings)
        logger.info("Backtesting champion model")
    else:
        agent = create_agent(config=settings)
        logger.info("No champion found - backtesting the heuristic baseline")

    run_backtest(agent, ticks, settings, label="backtest")
    return 0


def cmd_train(args) -> int:
    ticks = _load_ticks(args)
    if len(ticks) < settings.observation_window + 5:
        return 1

    trader = AutonomousTrader(settings)
    for cycle in range(1, args.cycles + 1):
        trader.run_learning_cycle(
            cycle=trader.memory.next_cycle,
            cycles=args.cycles,
            ticks=ticks,
            train_steps=args.train_steps,
        )
    print(json.dumps(trader.memory.summary(), indent=2, default=str))
    return 0


def cmd_paper(args) -> int:
    ticks = _load_ticks(args)
    if not ticks:
        return 1

    trader = AutonomousTrader(settings)
    if args.reset:
        trader.broker.reset()
        logger.info("Paper account reset to %.2f", settings.initial_capital)

    result = trader.run_paper_session(ticks, max_steps=args.steps)
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_autonomous(args) -> int:
    try:
        asyncio.run(
            run_autonomous(
                settings,
                cycles=args.cycles,
                interval=args.interval,
                days=args.days,
                max_rows=args.max_rows,
                train_steps=args.train_steps,
                sleep_seconds=args.sleep_seconds,
            )
        )
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    return 0


def cmd_diagnostic() -> int:
    result = run_testnet_diagnostic(settings)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] in {"ok", "missing_credentials"} else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_system",
        description="Autonomous RL crypto trading agent",
    )
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument("--days", type=int, default=30, help="Days of history (default: 30)")
    parser.add_argument("--max-rows", type=int, default=20_000, dest="max_rows")
    parser.add_argument("--csv", help="Load market data from a CSV instead of the API")
    parser.add_argument("--cycles", type=int, default=1, help="0 = run forever (autonomous)")
    parser.add_argument("--train-steps", type=int, default=settings.train_steps, dest="train_steps")
    parser.add_argument("--steps", type=int, default=500, help="Paper session length")
    parser.add_argument("--sleep-seconds", type=float, default=300.0, dest="sleep_seconds")
    parser.add_argument("--reset", action="store_true", help="Reset the paper account first")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings.ensure_directories()

    if args.mode == "status":
        return cmd_status()
    if args.mode == "stream":
        return cmd_stream(args)
    if args.mode == "backtest":
        return cmd_backtest(args)
    if args.mode == "train":
        return cmd_train(args)
    if args.mode == "paper":
        return cmd_paper(args)
    if args.mode == "autonomous":
        return cmd_autonomous(args)
    if args.mode == "testnet-diagnostic":
        return cmd_diagnostic()
    return 1


if __name__ == "__main__":
    sys.exit(main())
