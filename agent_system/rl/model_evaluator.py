"""Champion/challenger gating.

A newly trained model is never trusted automatically. It must clear absolute
safety gates and then beat the incumbent champion. This is the single most
important guard against the classic failure mode: a model that looks brilliant
on its training window and then drains the account live.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Absolute gates a candidate must clear regardless of the champion.
MIN_TRADES = 3
MAX_DRAWDOWN = 0.15
MIN_PROFIT_FACTOR = 1.0

#: How much worse than the champion a candidate may be and still pass.
EQUITY_TOLERANCE = 0.995


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    details: dict[str, Any] | None = None


def _metric(report: Mapping[str, Any] | None, name: str, default: float = 0.0) -> float:
    """Read a metric from either the top level or a nested ``metrics`` dict."""
    if not report:
        return default
    if name in report:
        try:
            return float(report[name])
        except (TypeError, ValueError):
            return default
    metrics = report.get("metrics")
    if isinstance(metrics, Mapping) and name in metrics:
        try:
            return float(metrics[name])
        except (TypeError, ValueError):
            return default
    return default


def decide_model_promotion(
    champion: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> PromotionDecision:
    """Decide whether ``candidate`` should replace ``champion``."""
    if not candidate:
        return PromotionDecision(False, "no_candidate")

    trades = _metric(candidate, "trades")
    drawdown = _metric(candidate, "drawdown")
    profit_factor = _metric(candidate, "profit_factor")
    equity = _metric(candidate, "equity")

    # --- Absolute safety gates -----------------------------------------
    if trades < MIN_TRADES:
        return PromotionDecision(
            False, "not_enough_trades", {"trades": trades, "required": MIN_TRADES}
        )
    if drawdown > MAX_DRAWDOWN:
        return PromotionDecision(
            False, "drawdown_too_high", {"drawdown": drawdown, "limit": MAX_DRAWDOWN}
        )
    if profit_factor < MIN_PROFIT_FACTOR:
        return PromotionDecision(
            False,
            "profit_factor_too_low",
            {"profit_factor": profit_factor, "required": MIN_PROFIT_FACTOR},
        )

    # --- First model with no incumbent ----------------------------------
    if not champion:
        return PromotionDecision(
            True, "no_champion_yet", {"equity": equity, "profit_factor": profit_factor}
        )

    # --- Relative comparison --------------------------------------------
    champion_equity = _metric(champion, "equity")
    if equity < champion_equity * EQUITY_TOLERANCE:
        return PromotionDecision(
            False,
            "worse_than_champion",
            {"candidate_equity": equity, "champion_equity": champion_equity},
        )

    champion_drawdown = _metric(champion, "drawdown")
    if champion_drawdown > 0 and drawdown > champion_drawdown * 2.0:
        return PromotionDecision(
            False,
            "drawdown_regression",
            {"candidate_drawdown": drawdown, "champion_drawdown": champion_drawdown},
        )

    return PromotionDecision(
        True,
        "candidate_not_worse",
        {"candidate_equity": equity, "champion_equity": champion_equity},
    )
