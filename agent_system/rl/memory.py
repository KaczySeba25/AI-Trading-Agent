"""Persistent memory: replay buffer and cross-session journal.

The requirement is that the agent remembers what it learned between restarts
and survives days of continuous operation. That means three things must
outlive the process: the model weights, the experience that produced them,
and the record of which model was judged best and why.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import tempfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque

import numpy as np

from agent_system.core.config import Settings, settings
from agent_system.core.logger import get_logger

logger = get_logger(__name__)


def _atomic_write(path: Path, write_bytes) -> None:
    """Write via a temp file and rename.

    A 24/7 process will eventually be killed mid-write. Renaming into place
    is atomic on POSIX, so a crash leaves either the old file or the new one
    -- never a half-written one that fails to parse on the next start.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        write_bytes(handle)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(handle.name, path)
    except Exception:
        handle.close()
        Path(handle.name).unlink(missing_ok=True)
        raise


@dataclass
class Experience:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded FIFO buffer of transitions."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.capacity = capacity
        self._items: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._items.append(
            Experience(
                observation=np.asarray(observation, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_observation=np.asarray(next_observation, dtype=np.float32),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int):
        if not self._items:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch = random.sample(list(self._items), min(batch_size, len(self._items)))
        return (
            np.stack([item.observation for item in batch]),
            np.array([item.action for item in batch], dtype=np.int64),
            np.array([item.reward for item in batch], dtype=np.float32),
            np.stack([item.next_observation for item in batch]),
            np.array([item.done for item in batch], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._items)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        _atomic_write(path, lambda handle: pickle.dump(list(self._items), handle, protocol=4))
        return path

    def load(self, path: str | Path) -> "ReplayBuffer":
        path = Path(path)
        if not path.exists():
            return self
        try:
            with path.open("rb") as handle:
                items = pickle.load(handle)
            self._items = deque(items, maxlen=self.capacity)
            logger.info("Loaded %d experiences from %s", len(self._items), path)
        except (pickle.UnpicklingError, EOFError, AttributeError) as exc:
            # A corrupt buffer is an inconvenience, not a reason to refuse to
            # start; the agent can rebuild experience from the next session.
            logger.error("Replay buffer at %s is unreadable (%s); starting empty", path, exc)
        return self


@dataclass
class CycleRecord:
    cycle: int
    timestamp: str
    train_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "timestamp": self.timestamp,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "promoted": self.promoted,
            "reason": self.reason,
        }


class AgentMemory:
    """Everything the agent must remember between sessions."""

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self.state_dir = Path(config.state_dir)
        self.model_dir = Path(config.model_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.journal_path = self.state_dir / "journal.json"
        self.buffer_path = self.state_dir / "replay_buffer.pkl"
        self.champion_path = self.model_dir / "champion"
        self.candidate_path = self.model_dir / "candidate"

        self.replay_buffer = ReplayBuffer().load(self.buffer_path)
        self.journal: dict[str, Any] = self._load_journal()

    # ------------------------------------------------------------------
    def _load_journal(self) -> dict[str, Any]:
        if not self.journal_path.exists():
            return {"cycles": [], "champion_metrics": None, "created_at": _now()}
        try:
            data = json.loads(self.journal_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("cycles", [])
                data.setdefault("champion_metrics", None)
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Journal at %s unreadable (%s); starting fresh", self.journal_path, exc)
        return {"cycles": [], "champion_metrics": None, "created_at": _now()}

    def record_cycle(
        self,
        train_metrics: dict[str, Any],
        validation_metrics: dict[str, Any],
        promoted: bool,
        reason: str,
    ) -> CycleRecord:
        record = CycleRecord(
            cycle=self.next_cycle,
            timestamp=_now(),
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            promoted=promoted,
            reason=reason,
        )
        self.journal["cycles"].append(record.as_dict())
        # Keep the journal bounded: after weeks of running, an unbounded list
        # is a slow read on every startup.
        self.journal["cycles"] = self.journal["cycles"][-500:]
        if promoted:
            self.journal["champion_metrics"] = validation_metrics
        return record

    def persist(self) -> None:
        self.journal["updated_at"] = _now()
        payload = json.dumps(self.journal, indent=2, default=str).encode("utf-8")
        _atomic_write(self.journal_path, lambda handle: handle.write(payload))
        self.replay_buffer.save(self.buffer_path)

    @property
    def next_cycle(self) -> int:
        return len(self.journal.get("cycles", [])) + 1

    @property
    def champion_metrics(self) -> dict[str, Any] | None:
        return self.journal.get("champion_metrics")

    @property
    def has_champion(self) -> bool:
        return self.champion_path.with_suffix(".zip").exists() or (
            self.champion_path.with_suffix(".txt").exists()
        )

    def summary(self) -> dict[str, Any]:
        cycles = self.journal.get("cycles", [])
        return {
            "cycles_completed": len(cycles),
            "next_cycle": self.next_cycle,
            "experiences": len(self.replay_buffer),
            "has_champion": self.has_champion,
            "champion_metrics": self.champion_metrics,
            "promotions": sum(1 for cycle in cycles if cycle.get("promoted")),
            "last_cycle": cycles[-1] if cycles else None,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
