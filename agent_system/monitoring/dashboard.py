"""Console dashboard placeholder."""

from __future__ import annotations


def render_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.4f}" for key, value in metrics.items())
