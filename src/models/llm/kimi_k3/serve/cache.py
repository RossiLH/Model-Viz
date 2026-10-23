from __future__ import annotations

from abc import ABC, abstractmethod


class Cache(ABC):
    @abstractmethod
    def get(self, key: str):
        ...

    @abstractmethod
    def put(self, key: str, value: any) -> None:
        ...