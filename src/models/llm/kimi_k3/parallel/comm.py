from __future__ import annotations

from abc import ABC, abstractmethod


class Communicator(ABC):
    """进程间通信原语抽象（broadcast / gossip 等单机不用，多机 use backend）。"""

    @abstractmethod
    def broadcast(self, tensor, src: int) -> None:
        ...

    @abstractmethod
    def send(self, tensor, dst: int) -> None:
        ...

    @abstractmethod
    def recv(self, tensor, src: int) -> None:
        ...


class NullCommunicator(Communicator):
    """单进程（world=1）退化的通信器：所有调用都是空操作。"""

    def broadcast(self, tensor, src: int) -> None:
        return None

    def send(self, tensor, dst: int) -> None:
        return None

    def recv(self, tensor, src: int) -> None:
        return None