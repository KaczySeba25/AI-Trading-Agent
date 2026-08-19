import numpy as np

from agent_system.rl.memory import AgentMemory, ReplayBuffer


def test_replay_buffer_round_trips_through_disk(tmp_path) -> None:
    buffer = ReplayBuffer(capacity=100)
    for index in range(10):
        buffer.push(np.ones(4) * index, index % 4, float(index), np.zeros(4), index == 9)

    path = buffer.save(tmp_path / "buffer.pkl")
    restored = ReplayBuffer(capacity=100).load(path)

    assert len(restored) == 10


def test_replay_buffer_respects_capacity() -> None:
    buffer = ReplayBuffer(capacity=5)
    for index in range(20):
        buffer.push(np.zeros(3), 0, 1.0, np.zeros(3), False)

    assert len(buffer) == 5


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer()
    for index in range(30):
        buffer.push(np.zeros(6), 1, 0.5, np.ones(6), False)

    observations, actions, rewards, next_observations, dones = buffer.sample(8)

    assert observations.shape == (8, 6)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_observations.shape == (8, 6)
    assert dones.shape == (8,)


def test_corrupt_buffer_does_not_crash_startup(tmp_path) -> None:
    """A truncated pickle must not stop a 24/7 agent from booting."""
    path = tmp_path / "buffer.pkl"
    path.write_bytes(b"not a pickle at all")

    buffer = ReplayBuffer().load(path)

    assert len(buffer) == 0


def test_memory_persists_cycles_across_restart(config) -> None:
    memory = AgentMemory(config)
    memory.record_cycle({"equity": 500}, {"equity": 505}, promoted=True, reason="no_champion_yet")
    memory.persist()

    reloaded = AgentMemory(config)

    assert reloaded.summary()["cycles_completed"] == 1
    assert reloaded.next_cycle == 2
    assert reloaded.champion_metrics == {"equity": 505}


def test_corrupt_journal_falls_back_to_empty(config) -> None:
    memory = AgentMemory(config)
    memory.journal_path.write_text("{ broken json", encoding="utf-8")

    reloaded = AgentMemory(config)

    assert reloaded.summary()["cycles_completed"] == 0
