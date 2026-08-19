"""Memory tests -- learning must survive a restart."""

from __future__ import annotations

from agent_system.rl.memory import AgentMemory, ReplayBuffer


def test_replay_buffer_evicts_oldest_beyond_capacity() -> None:
    buffer = ReplayBuffer(capacity=10)
    for index in range(25):
        buffer.push([float(index)], index % 4, float(index), [float(index + 1)], False)

    assert len(buffer) == 10
    assert buffer.buffer[0].action == 15 % 4  # the first 15 were evicted


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=100)
    for index in range(50):
        buffer.push([float(index), 0.5], 1, 1.0, [float(index), 0.6], False)

    states, actions, rewards, next_states, dones = buffer.sample(16)

    assert states.shape == (16, 2)
    assert actions.shape == (16,)
    assert rewards.shape == (16,)
    assert next_states.shape == (16, 2)
    assert dones.shape == (16,)


def test_replay_buffer_round_trips_through_disk(tmp_path) -> None:
    buffer = ReplayBuffer(capacity=100)
    for index in range(20):
        buffer.push([float(index)], 1, 1.0, [float(index)], False)
    buffer.save(tmp_path / "replay.pkl")

    restored = ReplayBuffer(capacity=100)
    assert restored.load(tmp_path / "replay.pkl")
    assert len(restored) == 20


def test_loading_a_missing_buffer_is_not_an_error(tmp_path) -> None:
    assert not ReplayBuffer().load(tmp_path / "nope.pkl")


def test_journal_persists_across_instances(test_settings) -> None:
    memory = AgentMemory(test_settings)
    memory.record_cycle(1, "history_training", True, "no_champion_yet", {"equity": 1010.0})
    memory.persist()

    reloaded = AgentMemory(test_settings)
    assert len(reloaded.journal) == 1
    assert reloaded.journal[0].accepted
    assert reloaded.next_cycle == 2


def test_champion_metrics_come_from_the_last_promotion(test_settings) -> None:
    memory = AgentMemory(test_settings)
    memory.record_cycle(1, "t", True, "promoted", {"equity": 1010.0})
    memory.record_cycle(2, "t", False, "rejected", {"equity": 900.0})
    memory.record_cycle(3, "t", True, "promoted", {"equity": 1050.0})

    assert memory.champion_metrics()["equity"] == 1050.0


def test_summary_reports_counts(test_settings) -> None:
    memory = AgentMemory(test_settings)
    memory.record_cycle(1, "t", True, "promoted", {"equity": 1010.0})
    memory.record_cycle(2, "t", False, "rejected", {"equity": 990.0})

    summary = memory.summary()
    assert summary["cycles_completed"] == 2
    assert summary["promotions"] == 1
