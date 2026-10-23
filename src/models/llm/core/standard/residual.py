from __future__ import annotations

import torch
import torch.nn as nn

from models.llm.core.interfaces import ResidualModule


class StandardResidual(ResidualModule):
    def forward(self, x: torch.Tensor, block_outputs: list[torch.Tensor]) -> torch.Tensor:
        return x + block_outputs[-1] if block_outputs else x