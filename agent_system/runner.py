"""Long-running operational loops."""

from __future__ import annotations

import asyncio

from agent_system.core.logger import get_logger
from agent_system.rl.history_trainer import PublicHistoryTrainer
from agent_system.system import TradingSystem


logger = get_logger(__name__)


async def run_paper_loop(
    cycles: int | None,
    min_ticks: int,
    timeout_seconds: float,
    sleep_seconds: float,
) -> list[dict[str, object]]:
    system = TradingSystem()
    reports: list[dict[str, object]] = []
    cycle_index = 0
    while cycles is None or cycle_index < cycles:
        cycle_index += 1
        logger.info("Starting paper loop cycle %d", cycle_index)
        try:
            report = await system.run_live_paper(
                min_ticks=min_ticks,
                timeout_seconds=timeout_seconds,
            )
            reports.append(report)
        except Exception:
            logger.exception("Paper loop cycle failed")
        if cycles is None or cycle_index < cycles:
            await asyncio.sleep(sleep_seconds)
    return reports


async def run_autonomous_learning_loop(
    cycles: int | None,
    interval: str,
    days: int,
    max_rows: int | None,
    history_cycles: int,
    train_steps: int,
    live_min_ticks: int,
    live_timeout_seconds: float,
    sleep_seconds: float,
) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    loop_index = 0
    while cycles is None or loop_index < cycles:
        loop_index += 1
        logger.info("Starting autonomous learning loop %d", loop_index)
        loop_report: dict[str, object] = {"loop": loop_index}
        try:
            loop_report["history"] = PublicHistoryTrainer().run(
                interval=interval,
                days=days,
                max_rows=max_rows,
                cycles=history_cycles,
                train_steps=train_steps,
            )
        except Exception:
            logger.exception("History learning phase failed")
            loop_report["history_error"] = "failed"

        try:
            loop_report["live_paper"] = await TradingSystem().run_live_paper(
                min_ticks=live_min_ticks,
                timeout_seconds=live_timeout_seconds,
            )
        except Exception:
            logger.exception("Live paper phase failed")
            loop_report["live_paper_error"] = "failed"

        reports.append(loop_report)
        if cycles is None or loop_index < cycles:
            await asyncio.sleep(sleep_seconds)
    return reports
