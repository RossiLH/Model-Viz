from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.norms import RMSNorm
from models.llm.deepseek_v4.config import MTPConfig


# ============================================================
# MTP 标签构造
# ============================================================

def build_mtp_labels(input_ids: torch.Tensor, mtp_depth: int,
                     ignore_index: int = -100) -> torch.Tensor:
    """构造 MTP 标签。

    主因果 LmHead 预测 x_{t+1}；MTP head k 预测 x_{t+k+2}。
    返回 [B, mtp_depth, T]，越界位置填 ignore_index。
    """
    B, T = input_ids.shape
    labels = torch.full((B, mtp_depth, T), ignore_index,
                        device=input_ids.device, dtype=torch.long)
    ids = input_ids.long()
    for k in range(mtp_depth):
        shift = k + 2
        if shift >= T:
            continue
        labels[:, k, : T - shift] = ids[:, shift:]
    return labels


# ============================================================
# MTP Transform（每深度一个）
# ============================================================

class MTPTransform(nn.Module):
    """MTP 变换：RMSNorm -> Linear -> SiLU -> Linear（mini 参考 canonical）。

    输入的深度变换使 MTP head 有独立于主 LmHead 的参数空间，
    而不至于直接复用主隐状态（论文用独立 MTP 模块而非简单共享）。
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None,
                 activation: str = "silu"):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.norm = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.act = {"silu": F.silu, "gelu": F.gelu, "relu": F.relu,
                    "identity": lambda x: x}[activation]
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ============================================================
# MTP Head（多 token 预测头）
# ============================================================

class MultiTokenPredictionHead(nn.Module):
    """多 token 预测头：mtp_depth 个变换 + 词表投影。

    每个 depth 独立 transform + head，预测位置 t+k+2 的 token
    （D4-A7）。可 tie 到主 LmHead 权重（tie_with_lm_head）。
    """

    def __init__(self, config: MTPConfig, dim: int, vocab_size: int):
        super().__init__()
        self.config = config
        self.depth = config.num_nextn_predict_layers
        self.tie_with_lm_head = getattr(config, "tie_with_lm_head", False)
        self.loss_weight = config.mtp_loss_weight

        self.transforms = nn.ModuleList([
            MTPTransform(dim) for _ in range(self.depth)
        ])
        self.heads = nn.ModuleList([
            nn.Linear(dim, vocab_size, bias=False) for _ in range(self.depth)
        ])

    def forward(self, hidden_states: torch.Tensor,
                input_ids: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """MTP 前向。

        Args:
            hidden_states: [B,T,D] 主模型最后层隐状态
            input_ids: [B,T]；提供时同时算加权 loss

        Returns:
            (mtp_logits [B,depth,T,V], mtp_loss)（loss 可为 None）
        """
        logits_list = []
        for k in range(self.depth):
            h = self.transforms[k](hidden_states)
            logits_list.append(self.heads[k](h))
        mtp_logits = torch.stack(logits_list, dim=1)  # [B,depth,T,V]

        loss = None
        if input_ids is not None:
            labels = build_mtp_labels(input_ids, self.depth, self.config.ignore_index)
            loss = self.compute_loss(mtp_logits, labels)
        return mtp_logits, loss

    def compute_loss(self, mtp_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """加权 MTP loss：depth 平均 + loss_weight 缩放。"""
        B, K, T, V = mtp_logits.shape
        losses = [
            F.cross_entropy(mtp_logits[:, k].reshape(-1, V),
                            labels[:, k].reshape(-1),
                            ignore_index=self.config.ignore_index)
            for k in range(K)
        ]
        raw = torch.stack(losses).sum()
        return self.loss_weight * raw


def mtp_loss_from_logits(mtp_logits: torch.Tensor, input_ids: torch.Tensor,
                         loss_weight: float = 0.3,
                         ignore_index: int = -100) -> torch.Tensor:
    """独立工具函数：给定 MTP logits 与 input_ids 直接算加权 loss。"""
    B, K, T, V = mtp_logits.shape
    labels = build_mtp_labels(input_ids, K, ignore_index)
    raw = torch.stack([
        F.cross_entropy(mtp_logits[:, k].reshape(-1, V),
                        labels[:, k].reshape(-1), ignore_index=ignore_index)
        for k in range(K)
    ]).sum()
    return loss_weight * raw