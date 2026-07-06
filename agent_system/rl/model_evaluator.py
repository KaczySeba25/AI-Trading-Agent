"""Model evaluation gates for training promotion decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationDecision:
    accepted: bool
    reason: str


def decide_model_promotion(
    baseline: dict[str, object] | None,
    candidate: dict[str, object],
    min_trades: int = 1,
    max_drawdown: float = 0.05,
) -> EvaluationDecision:
    candidate_metrics = candidate.get("metrics", {})
    if not isinstance(candidate_metrics, dict):
        return EvaluationDecision(False, "missing_candidate_metrics")

    trades = int(candidate.get("trades", 0))
    drawdown = float(candidate_metrics.get("drawdown", 1.0))
    candidate_equity = float(candidate.get("equity", 0.0))

    if trades < min_trades:
        return EvaluationDecision(False, "not_enough_trades")
    if drawdown > max_drawdown:
        return EvaluationDecision(False, "drawdown_too_high")

    if baseline is None:
        return EvaluationDecision(True, "no_baseline")

    baseline_equity = float(baseline.get("equity", 0.0))
    if candidate_equity >= baseline_equity:
        return EvaluationDecision(True, "candidate_not_worse")
    return EvaluationDecision(False, "candidate_worse_equity")
