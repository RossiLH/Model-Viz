from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.interfaces import AttnModule


class StandardMHA(AttnModule):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self._rope_cache = None
        self.max_seq_len = max_seq_len

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[2]
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len:
            theta = 10000.0 ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
            pos_float = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
            freqs = pos_float * theta.unsqueeze(0)
            cos = freqs.cos().unsqueeze(0).unsqueeze(0)
            sin = freqs.sin().unsqueeze(0).unsqueeze(0)
            self._rope_cache = (cos, sin)
        cos, sin = self._rope_cache
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated

    def init_cache(self, batch: int) -> dict:
        return {
            "k": torch.zeros(batch, self.n_kv_heads, 0, self.head_dim),
            "v": torch.zeros(batch, self.n_kv_heads, 0, self.head_dim),
        }

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        pos = torch.arange(T, device=x.device)
        q = self._rope(q, pos)
        k = self._rope(k, pos)

        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
            cache["k"] = k
            cache["v"] = v

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.full((T, k.shape[2]), float("-inf"), device=x.device), diagonal=1)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)