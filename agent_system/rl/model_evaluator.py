"""Champion/challenger promotion rules.

A model is only allowed to take over live decision-making if it clears every
gate below. The gates are deliberately conservative: promoting a bad model
costs real money, while refusing to promote a good one costs only a delay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_system.core.logger import get_logger

logger = get_logger(__name__)

#: Fewer trades than this and the metrics are noise, not evidence.
MIN_TRADES_FOR_PROMOTION = 3

#: A model that draws down more than this in validation is rejected outright,
#: however good its final equity looks.
MAX_ACCEPTABLE_DRAWDOWN = 0.15

#: Below 1.0 the model loses money gross of everything else.
MIN_PROFIT_FACTOR = 1.0

#: Allow the challenger to be marginally behind the champion; demanding strict
#: improvement every cycle just locks in the first lucky model.
EQUITY_TOLERANCE = 0.995


@dataclass
class PromotionDecision:
    accepted: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason, "details": self.details}


def decide_model_promotion(
    champion: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> PromotionDecision:
    """Decide whether ``candidate`` should replace ``champion``."""
    if not candidate:
        return PromotionDecision(False, "no_candidate")

    candidate_metrics = candidate.get("metrics", {}) or {}
    candidate_equity = float(candidate.get("equity", 0.0))
    candidate_trades = int(candidate.get("trades", 0))
    candidate_drawdown = float(candidate_metrics.get("drawdown", 1.0))
    candidate_pf = float(candidate_metrics.get("profit_factor", 0.0))

    details = {
        "candidate_equity": candidate_equity,
        "candidate_trades": candidate_trades,
        "candidate_drawdown": candidate_drawdown,
        "candidate_profit_factor": candidate_pf,
    }

    if candidate_trades < MIN_TRADES_FOR_PROMOTION:
        return PromotionDecision(False, "not_enough_trades", details)

    if candidate_drawdown > MAX_ACCEPTABLE_DRAWDOWN:
        return PromotionDecision(False, "drawdown_too_high", details)

    if candidate_pf < MIN_PROFIT_FACTOR:
        return PromotionDecision(False, "profit_factor_too_low", details)

    if not champion:
        return PromotionDecision(True, "no_champion_yet", details)

    champion_metrics = champion.get("metrics", {}) or {}
    champion_equity = float(champion.get("equity", 0.0))
    champion_drawdown = float(champion_metrics.get("drawdown", 1.0))
    details["champion_equity"] = champion_equity
    details["champion_drawdown"] = champion_drawdown

    if candidate_equity < champion_equity * EQUITY_TOLERANCE:
        return PromotionDecision(False, "worse_than_champion", details)

    # Guard against the trap of a slightly richer model that is far riskier.
    if candidate_drawdown > max(champion_drawdown * 1.5, champion_drawdown + 0.05):
        return PromotionDecision(False, "drawdown_regression", details)

    return PromotionDecision(True, "candidate_not_worse", details)
