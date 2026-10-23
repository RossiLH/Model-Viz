from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vla.core.norms import AdaRMSNorm
from models.vla.core.standard.attention import StandardVLAAttention
from models.vla.core.standard.mlp import StandardVLAFFN


class PiGemmaBlock(nn.Module):
    """Gemma 风格 transformer block（AdaRMSNorm + GQA 注意力 + FFN）

    ``per_token_cond`` 透传给两处 AdaRMSNorm：
    - False（默认）：条件在序列轴均值池化，适用于 π0/π0.5（单标量 timestep）；
    - True：逐 token 独立调制，π0.7 动作专家必须开启，否则训练期 RTC 无法
      把"哪些步是冻结前缀"这一信息传进网络（详见 core/norms.py 的说明）。
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, head_dim: int, mlp_dim: int,
                 bidirectional: bool = True, cond_dim: int | None = None,
                 per_token_cond: bool = False):
        super().__init__()
        self.input_norm = AdaRMSNorm(dim, cond_dim=cond_dim, per_token_cond=per_token_cond)
        self.post_attn_norm = AdaRMSNorm(dim, cond_dim=cond_dim, per_token_cond=per_token_cond)
        self.attn = StandardVLAAttention(dim, n_heads, n_kv_heads, head_dim, bidirectional=bidirectional)
        self.ffn = StandardVLAFFN(dim, mlp_dim)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        h = self.input_norm(x, cond)
        h = self.attn(h, attn_mask=attn_mask)
        x = residual + h
        residual = x
        h = self.post_attn_norm(x, cond)
        h = self.ffn(h)
        x = residual + h
        return x


class PiGemmaModel(nn.Module):
    def __init__(self, dim: int, depth: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 mlp_dim: int, bidirectional: bool = True, cond_dim: int | None = None,
                 per_token_cond: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            PiGemmaBlock(dim, n_heads, n_kv_heads, head_dim, mlp_dim,
                         bidirectional=bidirectional, cond_dim=cond_dim,
                         per_token_cond=per_token_cond)
            for _ in range(depth)
        ])
        self.norm = AdaRMSNorm(dim, cond_dim=cond_dim, per_token_cond=per_token_cond)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond=cond, attn_mask=attn_mask)
        return self.norm(x, cond=cond)