from __future__ import annotations

from abc import ABC, abstractmethod


class RLAlgorithm(ABC):
    @abstractmethod
    def compute_advantage(self, rewards: list[float]) -> list[float]:
        ...