from __future__ import annotations

from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def run(self, model) -> dict:
        ...