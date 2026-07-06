from agent_system.rl.model_evaluator import decide_model_promotion


def test_rejects_candidate_without_enough_trades() -> None:
    decision = decide_model_promotion(
        None,
        {"equity": 501, "trades": 0, "metrics": {"drawdown": 0.01}},
    )

    assert not decision.accepted
    assert decision.reason == "not_enough_trades"


def test_accepts_candidate_with_better_equity_and_low_drawdown() -> None:
    decision = decide_model_promotion(
        {"equity": 500, "trades": 2, "metrics": {"drawdown": 0.01}},
        {"equity": 501, "trades": 2, "metrics": {"drawdown": 0.02}},
    )

    assert decision.accepted
    assert decision.reason == "candidate_not_worse"
