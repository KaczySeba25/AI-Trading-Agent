import time

from agent_system.data.data_buffer import MarketTick
from agent_system.rl.history_trainer import select_training_window, split_train_validation


def make_ticks(count: int) -> list[MarketTick]:
    now = int(time.time() * 1000)
    return [
        MarketTick(now + index * 60_000, 100 + index, 1.0, 100 + index, 100 + index)
        for index in range(count)
    ]


def test_split_train_validation_keeps_newest_for_validation() -> None:
    ticks = make_ticks(100)
    train, validation = split_train_validation(ticks, 0.2)

    assert len(train) == 50
    assert len(validation) == 50
    assert validation[0].timestamp > train[-1].timestamp


def test_select_training_window_moves_across_history() -> None:
    ticks = make_ticks(5000)

    first = select_training_window(ticks, cycle=1, cycles=3)
    last = select_training_window(ticks, cycle=3, cycles=3)

    assert len(first) == 2000
    assert len(last) == 2000
    assert last[0].timestamp > first[0].timestamp
