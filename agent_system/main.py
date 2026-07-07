"""Entry point for the trading agent."""

from __future__ import annotations

import argparse
import asyncio


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous trading agent")
    parser.add_argument(
        "--mode",
        choices=[
            "stream",
            "train-smoke",
            "online-smoke",
            "paper-smoke",
            "live-paper",
            "paper-loop",
            "public-history-train",
            "history-learning-loop",
            "testnet-diagnostic",
        ],
        default="stream",
    )
    parser.add_argument("--min-ticks", type=int, default=120)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=10.0)
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--train-steps", type=int, default=1024)
    parser.add_argument("--online-cycle", action="store_true")
    args = parser.parse_args()

    if args.mode == "stream":
        from agent_system.data.binance_ws import run_phase_one

        asyncio.run(run_phase_one())
        return

    if args.mode == "train-smoke":
        from agent_system.rl.trainer import train_ppo
        from agent_system.system import synthetic_ticks

        ticks = synthetic_ticks()
        print(train_ppo(ticks, total_timesteps=256))
        return

    if args.mode == "online-smoke":
        from agent_system.rl.online_loop import OnlineLearningLoop
        from agent_system.system import synthetic_ticks

        ticks = synthetic_ticks()
        print(OnlineLearningLoop().run_cycle(ticks, collect_steps=400, train_steps=256))
        return

    if args.mode == "paper-smoke":
        from agent_system.system import TradingSystem, synthetic_ticks

        ticks = synthetic_ticks()
        print(TradingSystem().run_paper_replay(ticks))
        return

    if args.mode == "live-paper":
        from agent_system.system import TradingSystem

        print(
            asyncio.run(
                TradingSystem().run_live_paper(
                    min_ticks=args.min_ticks,
                    timeout_seconds=args.timeout_seconds,
                )
            )
        )
        return

    if args.mode == "paper-loop":
        from agent_system.runner import run_paper_loop

        cycles = args.cycles if args.cycles > 0 else None
        print(
            asyncio.run(
                run_paper_loop(
                    cycles=cycles,
                    min_ticks=args.min_ticks,
                    timeout_seconds=args.timeout_seconds,
                    sleep_seconds=args.sleep_seconds,
                )
            )
        )
        return

    if args.mode == "public-history-train":
        from agent_system.environment.backtester import (
            run_public_history_training_and_backtest,
        )

        print(
            run_public_history_training_and_backtest(
                interval=args.interval,
                limit=args.limit,
                train_steps=args.train_steps,
                online_cycle=args.online_cycle,
            )
        )
        return

    if args.mode == "history-learning-loop":
        from agent_system.rl.history_trainer import PublicHistoryTrainer

        print(
            PublicHistoryTrainer().run(
                interval=args.interval,
                limit=args.limit,
                cycles=args.cycles,
                train_steps=args.train_steps,
            )
        )
        return

    if args.mode == "testnet-diagnostic":
        from agent_system.execution.diagnostics import run_testnet_diagnostic

        print(run_testnet_diagnostic())
        return


if __name__ == "__main__":
    main()
