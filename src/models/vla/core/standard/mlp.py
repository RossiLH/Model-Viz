from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardVLAFFN(nn.Module):
    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.up_proj = nn.Linear(dim, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate) * up)