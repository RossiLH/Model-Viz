from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardVLAAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int = 1, head_dim: int = 16, bidirectional: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.bidirectional = bidirectional
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        if not self.bidirectional:
            mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
            attn = attn + mask
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                attn_mask = attn_mask.squeeze(1)
            attn = attn + attn_mask.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        # 注意: reshape 目标维度是 n_heads * head_dim（而非输入 dim D），
        # 因为 GQA 场景下两者可能不同
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)