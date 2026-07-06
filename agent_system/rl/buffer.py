"""Experience buffer primitives for later online learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ExperienceBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._items: list[Experience] = []

    def add(self, experience: Experience) -> None:
        self._items.append(experience)
        if len(self._items) > self.capacity:
            self._items.pop(0)

    def snapshot(self) -> list[Experience]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
