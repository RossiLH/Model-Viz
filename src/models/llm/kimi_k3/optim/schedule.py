from __future__ import annotations

from abc import ABC, abstractmethod


class LRSchedule(ABC):
    @abstractmethod
    def get_lr(self, step: int) -> float:
        ...