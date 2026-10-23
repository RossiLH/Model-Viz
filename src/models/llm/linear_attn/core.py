from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class LinearAttentionBase(nn.Module):
    """Base class for all linear attention variants."""

    def __init__(self, dim: int, n_heads: int, head_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _apply_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """phi(x) = ELU(x) + 1, ensures positive features."""
    return F.elu(x) + 1.0


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def householder_reflection(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply Householder reflection H = I - 2*u*u^T to vector v."""
    return v - 2 * torch.sum(v * u, dim=-1, keepdim=True) * u


def householder_matrix(u: torch.Tensor) -> Callable:
    """Return a function that applies H = I - 2*u*u^T to any input."""
    def apply_h(x):
        return x - 2 * torch.einsum('...i,...j,...jk->...ik', u, u, x)
    return apply_h