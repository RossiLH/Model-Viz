from __future__ import annotations

from abc import ABC, abstractmethod


class SFTTrainer(ABC):
    @abstractmethod
    def train_step(self, batch: dict) -> dict:
        ...