from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn


class VisionEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        ...


class ActionDecoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, noisy_actions: torch.Tensor, timestep: torch.Tensor, prefix_kv: Optional[dict] = None) -> torch.Tensor:
        ...


class VLAInference(ABC):
    @abstractmethod
    def encode_observation(self, images: list[torch.Tensor], text: list[int]) -> dict:
        ...
    @abstractmethod
    def sample_actions(self, prefix_kv: dict, num_steps: int = 10) -> torch.Tensor:
        ...