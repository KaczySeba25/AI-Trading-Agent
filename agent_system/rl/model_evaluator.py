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
    starting_equity: float = 500.0,
    min_trades: int = 3,
    max_drawdown: float = 0.03,
    min_profit_factor: float = 1.05,
    min_equity_return: float = 0.0,
) -> EvaluationDecision:
    candidate_metrics = candidate.get("metrics", {})
    if not isinstance(candidate_metrics, dict):
        return EvaluationDecision(False, "missing_candidate_metrics")

    trades = int(candidate.get("trades", 0))
    drawdown = float(candidate_metrics.get("drawdown", 1.0))
    profit_factor = float(candidate_metrics.get("profit_factor", 0.0))
    candidate_equity = float(candidate.get("equity", 0.0))
    equity_return = (candidate_equity - starting_equity) / starting_equity

    if trades < min_trades:
        return EvaluationDecision(False, "not_enough_trades")
    if drawdown > max_drawdown:
        return EvaluationDecision(False, "drawdown_too_high")
    if profit_factor < min_profit_factor:
        return EvaluationDecision(False, "profit_factor_too_low")
    if equity_return < min_equity_return:
        return EvaluationDecision(False, "equity_return_too_low")

    if baseline is None:
        return EvaluationDecision(True, "no_baseline")

    baseline_equity = float(baseline.get("equity", 0.0))
    if candidate_equity >= baseline_equity:
        return EvaluationDecision(True, "candidate_not_worse")
    return EvaluationDecision(False, "candidate_worse_equity")
