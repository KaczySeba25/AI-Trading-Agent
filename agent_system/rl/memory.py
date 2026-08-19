"""Persistent memory across restarts.

For an agent meant to run 24/7 for days, "learning" is worthless if a restart
wipes it. This module persists three things to disk:

* the champion model (the only one allowed to trade),
* a JSON journal of every learning cycle and its metrics,
* an experience replay buffer that survives process restarts.
"""

from __future__ import annotations

import json
import pickle
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Deque, Sequence

import numpy as np

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    """Bounded replay buffer with O(1) eviction and disk persistence.

    The original implementation used ``list.pop(0)``, which is O(n) per insert;
    a ``deque`` with ``maxlen`` evicts in constant time.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        self.buffer.append(
            Experience(
                state=[float(value) for value in np.asarray(state).flatten()],
                action=int(action),
                reward=float(reward),
                next_state=[float(value) for value in np.asarray(next_state).flatten()],
                done=bool(done),
            )
        )

    def sample(self, batch_size: int = 64) -> tuple[np.ndarray, ...]:
        if not self.buffer:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch = random.sample(list(self.buffer), min(len(self.buffer), batch_size))
        return (
            np.array([item.state for item in batch], dtype=np.float32),
            np.array([item.action for item in batch], dtype=np.int64),
            np.array([item.reward for item in batch], dtype=np.float32),
            np.array([item.next_state for item in batch], dtype=np.float32),
            np.array([item.done for item in batch], dtype=np.float32),
        )

    def save(self, path: str | Path) -> Path:
        """Persist atomically so a crash mid-write cannot corrupt the buffer."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(list(self.buffer), handle)
        temporary.replace(path)
        return path

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        try:
            with path.open("rb") as handle:
                items = pickle.load(handle)
            self.buffer = deque(items, maxlen=self.capacity)
            logger.info("Loaded %d experiences from %s", len(self.buffer), path.name)
            return True
        except (pickle.PickleError, EOFError, AttributeError) as exc:
            logger.warning("Could not load replay buffer: %s", exc)
            return False

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class CycleRecord:
    """One learning cycle's outcome."""

    cycle: int
    timestamp: float
    mode: str
    accepted: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


class AgentMemory:
    """Champion model, cycle journal and replay buffer on disk."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        config.ensure_directories()
        self.state_dir = Path(config.state_dir)
        self.model_dir = Path(config.model_dir)
        self.journal_path = self.state_dir / "journal.json"
        self.buffer_path = self.state_dir / "replay_buffer.pkl"
        self.champion_path = self.model_dir / "champion"
        self.candidate_path = self.model_dir / "candidate"

        self.replay = ReplayBuffer()
        self.replay.load(self.buffer_path)
        self.journal: list[CycleRecord] = self._load_journal()

    def _load_journal(self) -> list[CycleRecord]:
        if not self.journal_path.exists():
            return []
        try:
            payload = json.loads(self.journal_path.read_text(encoding="utf-8"))
            return [CycleRecord(**record) for record in payload]
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Journal unreadable, starting fresh: %s", exc)
            return []

    def record_cycle(
        self,
        cycle: int,
        mode: str,
        accepted: bool,
        reason: str,
        metrics: dict[str, float],
    ) -> CycleRecord:
        record = CycleRecord(
            cycle=cycle,
            timestamp=time.time(),
            mode=mode,
            accepted=accepted,
            reason=reason,
            metrics={key: float(value) for key, value in metrics.items()},
        )
        self.journal.append(record)
        self._save_journal()
        return record

    def _save_journal(self) -> None:
        temporary = self.journal_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([asdict(record) for record in self.journal], indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.journal_path)

    def persist(self) -> None:
        """Flush everything that must survive a restart."""
        self.replay.save(self.buffer_path)
        self._save_journal()

    @property
    def next_cycle(self) -> int:
        return (self.journal[-1].cycle + 1) if self.journal else 1

    def champion_metrics(self) -> dict[str, float] | None:
        """Metrics of the most recently promoted model, if any."""
        for record in reversed(self.journal):
            if record.accepted:
                return record.metrics
        return None

    def has_champion(self) -> bool:
        return (
            self.champion_path.with_suffix(".zip").exists()
            or self.champion_path.with_suffix(".txt").exists()
        )

    def summary(self) -> dict[str, Any]:
        accepted = [record for record in self.journal if record.accepted]
        return {
            "cycles_completed": len(self.journal),
            "promotions": len(accepted),
            "experiences_stored": len(self.replay),
            "has_champion": self.has_champion(),
            "champion_metrics": self.champion_metrics(),
            "last_cycles": [asdict(record) for record in self.journal[-10:]],
        }
