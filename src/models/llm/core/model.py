from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.llm.core.interfaces import AttnModule, Block
from models.llm.core.config import LLMConfig
from models.llm.core.norms import RMSNorm


class SimpleBlock(Block):
    def __init__(self, attn: AttnModule, ffn: nn.Module, dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.attn = attn
        self.ffn = ffn
        self.attn_norm = RMSNorm(dim, norm_eps)
        self.ffn_norm = RMSNorm(dim, norm_eps)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GenericLLM(nn.Module):
    def __init__(self, config: LLMConfig, blocks: list[Block]):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.model.vocab_size, config.model.dim)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(config.model.dim, config.model.norm_eps)

    def forward(self, tokens: torch.Tensor, caches: Optional[list[Optional[dict]]] = None) -> torch.Tensor:
        h = self.tok_embed(tokens)
        for i, block in enumerate(self.blocks):
            cache = caches[i] if caches else None
            h = block(h, cache)
        return self.norm(h)