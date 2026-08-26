"""Logger-backed metric reporting."""

from __future__ import annotations

from agent_system.core.logger import get_logger
from agent_system.monitoring.dashboard import render_metrics


logger = get_logger(__name__)


def log_metrics(metrics: dict[str, float]) -> None:
    logger.info("metrics | %s", render_metrics(metrics))
