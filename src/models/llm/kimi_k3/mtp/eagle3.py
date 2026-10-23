# -*- coding: utf-8 -*-
"""EAGLE-3 草稿模型：从 MTP 层微调出的投机解码 draft（K3 报告 §4.1.4）。

EAGLE-3（extension-3 of the speculative decoding literature）把预训练的 MTP
层当作起点，微调成一个单解码器层式的草稿模型。目标模型冻结，只训草稿层与
特征融合投影。两个关键机制（对齐 kimi3-master ``serve/eagle3.py`` 全文）：

1. **特征融合 ``FeatureFusion``**：输入是低/中/高三层特征的拼接（分别取自
   第 1、4、最终 Attention Residuals 块的输出——一个 93 层骨干只有这 3 处
   “深度方向混合”出入口），经一个**无 bias** 的 ``W_E3: 3d -> d`` 投影
   融合。``W_E3`` 初值 ``[0 0 I]``：最后一列是单位阵，于是初始融合输出
   **完全等于高层特征 h_h**——正是 MTP 预训练时的输入；模型从“只信高层”
   逐步学到“吸收低/中层特征”，初始化不引入瞬间扰动。

2. **LK 损失**：投机解码的加速比正比于逐 token 接受率 ``sum_x min(p,q)``
   （p 目标、q 草稿）。用减法 KL 当代理**不能保证**最大化接受率（容量受限的
   草稿模型），故 K3 直接优化权重似然 LK 损失
   ::

       L_LK = -log sum_x min(p(x), q(x))

   温度 1，无 ground-truth CE 项。训练时把草稿展开 7 步：除第一步外（最先前
   位置的“目标侧特征”不可得），草稿消费自己早前步的输出——正是推理期
   循环起草（drafting）的过程的镜像。

特征栈语义（在 main 侧由 ``AttentionResiduals`` 每块输出收集）：
- ``eagle3_feature_stack(attn_res_state)``：从 AttnRes 状态缓存里按
  ``(0, 3, -1)``（第 1、第 4、最后）三个 unit 的输出取特征。
  本项目 AttnRes 状态语义：``n_units = 2*n_layers``（每 block 两个 sublayer
  unit）。kimi3-master 里 ``feature_blocks=(0,3,-1)`` 指第 1 / 4 / 末个
  **Attention Residuals 块**的输出。实现上本项目直接给一个通用的
  ``eagle3_feature_stack(state, picks)``：默认 picks 视 n_layers 归一。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .mtp_layer import MTPLayer

__all__ = [
    "Eagle3Draft",
    "Eagle3FeatureFusion",
    "FeatureFusion",
    "acceptance_rate",
    "draft_config",
    "eagle3_feature_stack",
    "lk_loss",
    "speedup_from_acceptance",
]


def acceptance_rate(p: Tensor, q: Tensor) -> Tensor:
    """``sum_x min(p(x), q(x))`` —— 投机解码加速比正比于的量。"""
    return torch.minimum(p, q).sum(-1)


def lk_loss(target_logits: Tensor, draft_logits: Tensor) -> Tensor:
    """Eq. 16，温度 = 1：直接是接受率的负对数，无 ground-truth CE 项。

    优化真正决定加速比的量，而非它的代理（如 KL）。
    """
    p = torch.softmax(target_logits.float(), dim=-1)
    q = torch.softmax(draft_logits.float(), dim=-1)
    return -torch.log(acceptance_rate(p, q).clamp_min(1e-9)).mean()


class Eagle3FeatureFusion(nn.Module):
    """无 bias 的 ``W_E3``，初始 ``[0 0 I]``（官方 FeatureFusion）。

    ``proj: (n_sources*dim) -> dim``，权重初值前 ``(n_sources-1)*dim`` 列全 0、
    末 ``dim`` 列单位阵。初始化后 ``fuse 输出 ≡ 高层特征``。
    """

    def __init__(self, dim: int, n_sources: int = 3):
        super().__init__()
        self.dim = dim
        self.n_sources = n_sources
        self.proj = nn.Linear(dim * n_sources, dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.proj.weight.zero_()
            # [0 ... 0 I]：最后一个源（高层特征）恒等通过。
            self.proj.weight[:, -self.dim:] = torch.eye(self.dim)

    def forward(self, features: Sequence[Tensor]) -> Tensor:
        if len(features) != self.n_sources:
            raise ValueError(f"expected {self.n_sources} feature sources, got {len(features)}")
        return self.proj(torch.cat(list(features), dim=-1))


# 兼容官方命名：FeatureFusion 即 Eagle3FeatureFusion。
FeatureFusion = Eagle3FeatureFusion


@torch.no_grad()
def eagle3_feature_stack(
    state,
    picks: Optional[Sequence[int]] = None,
) -> list[Tensor]:
    """从本项目 AttentionResiduals 状态里摊平 EAGLE-3 可读的特征源列表。

    ``state`` 是 ``Kimi3ForCausalLM`` 前向期间 ``AttentionResiduals`` 的残差
    容器（``modules/attn_res.py::AttnResState``）。本项目 AttnRes 以块（block）
    为单位维持低级表示：``init_state`` 把 token embedding 置为 ``blocks[0]``，
    随后每完成一个块就折叠进 ``blocks``（``add_output``），``sources()`` 返回
    "embedding + 各已完成块表示 + 当前块 partial" 的可读源列表。

    **职责划分（修复 V2 Q5 的重复选择缺陷）**：本函数只负责"把状态对象摊平
    成完整的出口列表"，**不做"选哪三层"**；选择唯一保留在
    ``Eagle3Draft.fuse()``（由 ``DraftConfig.feature_blocks`` 单一来源驱动），
    与官方 ``kimi3-master/serve/eagle3.py::fuse`` 一致。因此：

    - ``picks=None``（默认）：返回**全部**可读源 ``[b_0, b_1, ..., b_N]``，
      供上层自行选择（推荐用法：先摊平，再按 ``feature_blocks`` 挑）。
    - ``picks`` 非空：为向后兼容仍可在此直接按索引挑选（已弃用，新代码
      不要用，避免与 ``fuse()`` 的选择逻辑互斥串联而越界）。

    返回：特征张量列表，每个形状 ``(B, T, C)``。
    """
    if hasattr(state, "outputs"):  # 通用兼容：直接给输出列表的对象
        seq = list(state.outputs)
    elif hasattr(state, "sources"):
        seq = state.sources()
    elif isinstance(state, (list, tuple)):
        seq = list(state)
    else:
        raise TypeError(f"cannot read feature sources from {type(state)}")
    if picks is None:
        return seq
    n = len(seq)
    idx = [i if i >= 0 else n + i for i in picks]
    if any(i >= n for i in idx) or any(i < 0 for i in idx):
        raise IndexError(f"feature picks {picks} out of range for {n} sources")
    return [seq[i] for i in idx]


@dataclass
class DraftConfig:
    """EAGLE-3 草稿配置（对齐官方 kimi3-master DraftConfig）。"""

    unroll_steps: int = 7   # 训练期 rollout 步数（论文测试协议）。
    feature_blocks: tuple = (0, 3, -1)  # 特征取自那些 AttnRes 块。
    qat: bool = True        # 草稿 MoE 权重用 MXFP4/MXFP8 量化（§4.1.4）


class Eagle3Draft(nn.Module):
    """从预训练 MTP 层微调出的草稿模型。

    ``mtp_layer`` 是 MTPLayer（其内部 ``block`` 就是草案层），``lm_head`` 是
    目标模型共享词表头。推理期：每步从目标模型取 ``fused``（特征融合）输入，
    草稿输出 logits [B,1,V]，再与目标做接受/拒绝。
    """

    def __init__(
        self,
        mtp_layer: nn.Module,
        dim: int,
        lm_head: nn.Module,
        cfg: DraftConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or draft_config()
        self.fusion = Eagle3FeatureFusion(dim, n_sources=len(self.cfg.feature_blocks))
        self.layer = mtp_layer
        self.lm_head = lm_head

    def fuse(self, block_features: Sequence[Tensor]) -> Tensor:
        return self.fusion([block_features[i] for i in self.cfg.feature_blocks])

    def forward(self, fused: Tensor, next_embeds: Tensor) -> Tensor:
        h = self.layer(fused, next_embeds)
        return self.lm_head(h)

    @torch.no_grad()
    def draft(
        self, fused: Tensor, embed_fn: Callable[[Tensor], Tensor],
        *, n_steps: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """自动回归起草：第一步后草稿吃自己的输出——推理期 draft 的原样。"""
        n_steps = n_steps or self.cfg.unroll_steps
        tokens, logits_out = [], []
        state = fused
        last = embed_fn(torch.zeros(fused.shape[:2], dtype=torch.long, device=fused.device))
        for _ in range(n_steps):
            logits = self.forward(state, last)
            nxt = logits[:, -1:].argmax(-1)
            tokens.append(nxt)
            logits_out.append(logits[:, -1:])
            last = embed_fn(nxt)
            state = state[:, -1:]
        return torch.cat(tokens, dim=1), torch.cat(logits_out, dim=1)


def speedup_from_acceptance(rate: float, draft_len: int) -> float:
    """lossless 投机采样下每个目标前向的期望 token 数。"""
    if rate >= 1.0:
        return float(draft_len + 1)
    return (1.0 - rate ** (draft_len + 1)) / (1.0 - rate)


draft_config = DraftConfig  # 兼容旧命名