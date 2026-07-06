"""Long-running operational loops."""

from __future__ import annotations

import asyncio

from agent_system.core.logger import get_logger
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
