from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class AttnModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        ...

    def init_cache(self, batch: int) -> Optional[dict]:
        return None


class MoEModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class ResidualModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, block_outputs: list[torch.Tensor]) -> torch.Tensor:
        ...


class Block(nn.Module, ABC):
    attn: AttnModule
    ffn: nn.Module

    @abstractmethod
    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        ...