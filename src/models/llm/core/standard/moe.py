from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.interfaces import MoEModule


class StandardMoE(MoEModule):
    def __init__(self, dim: int, n_experts: int, top_k: int, hidden_dim: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=False),
                nn.Linear(hidden_dim, dim, bias=False),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        logits = self.gate(x)
        top_k_vals, top_k_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_vals, dim=-1)
        out = torch.zeros_like(x)
        for b in range(B):
            for t in range(T):
                for k in range(self.top_k):
                    e = top_k_idx[b, t, k].item()
                    w = weights[b, t, k]
                    out[b, t] += w * self.experts[e](x[b, t].unsqueeze(0)).squeeze(0)
        return out