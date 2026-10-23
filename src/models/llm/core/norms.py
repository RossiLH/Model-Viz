from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class HeadRMSNorm(nn.Module):
    def __init__(self, n_heads: int, d_head: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_heads, d_head))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)