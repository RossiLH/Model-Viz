# -*- coding: utf-8 -*-
"""DeepSeek-V4 0x0F：Anticipatory Routing（前瞻路由）的 CPU 可测子集。

论文语义（§4.2，第 1361~1377 行）：把主干网络与路由网络的更新解耦——第 t 步
用**历史参数 θ_{t−Δ}**（上一状态的路由权重）提前算好路由索引并缓存，第 t 步直接
复用，从而把路由决策从「依赖自身当前状态」的递归里解放出来，并把额外墙钟开销
压到约 20%。配套一个「只在 loss spike 时触发」的自动检测：检测到 loss 突增时
进入 anticipatory 模式运行一段时间，随后回退到标准训练。

本项目是 CPU 教学平台、无完整训练循环，因此实现的是**机制的 CPU 可测子集**：
- ``snapshot()``：冻结当前路由 gate 权重 → 得到历史参数 θ_{t−Δ}；
- ``precompute()``：用**冻结参数**预计算 top-k 索引并**缓存**（按输入签名）；
- ``lookup()``：命中缓存直接返回，未命中则现场用当前参数计算；
- ``observe_loss()`` + ``active``：loss-spike 自动检测 + revert 状态机。

接入方式：``DeepSeekV4MoE.forward(..., anticipatory_router=router)`` 时，
若 router 处于 anticipatory 且有缓存索引，就用缓存索引（跳过 live 路由）；
否则走标准路由。这是非侵入的可选路径，默认关闭（标准行为不变）。
"""
from __future__ import annotations

from typing import Optional

import torch


def _input_signature(x: torch.Tensor, ids: Optional[torch.Tensor]) -> str:
    """输入签名：x 的量化内容哈希 + ids 内容。用于缓存 key。

    教学实现：直接把张量内容哈希进来（测试规模下可复现）；真实系统应改用
    data_ptr / 迭代号等稳定键（论文里是「fetch data of step t in advance」，
    即按训练步号而非内容键控缓存）。
    """
    h = torch.as_tensor([0], dtype=torch.long)
    for t in (x, ids):
        if t is not None:
            v = t.detach().float().view(-1)
            v = (v * 1024).round().to(torch.int64)
            h = (h * 31 + v.sum() % (2 ** 62)) % (2 ** 62)
    return f"{x.shape}_{int(h)}"


class AnticipatoryRouter:
    """Anticipatory Routing 的 CPU 可测子集：冻结权重 + 索引缓存 + spike 检测。

    Args:
        moe: 要包裹的 DeepSeekV4MoE（需要 ._route 与 .gate / .topk）。
        spike_ratio: loss 突增判定 = 当前 loss > EMA × spike_ratio。
        revert_after: 触发后运行多少步自动回退到标准模式。
        ema_alpha: loss EMA 的平滑系数（默认 0.3，近期主导）。
    """

    def __init__(self, moe, spike_ratio: float = 2.0,
                 revert_after: int = 3, ema_alpha: float = 0.3):
        self.moe = moe
        self.spike_ratio = spike_ratio
        self.revert_after = revert_after
        self.ema_alpha = ema_alpha
        self._cache: dict[str, torch.Tensor] = {}
        self._ema: Optional[float] = None
        self._active = False
        self._spike_steps_left = 0

    # ------------------------------------------------------------------
    # 历史参数 θ_{t−Δ}
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, torch.Tensor]:
        """冻结当前路由 gate 权重 → 历史参数快照（detach，不受后续更新影响）。

        论文里 θ_{t−Δ} 是 Δ 步之前的路由参数；这里用「当前权重的一份冻结拷贝」
        表达「路由权重与主干解耦、可以独立于主干更新」的核心。
        """
        return {"gate": [p.detach().clone() for p in self.moe.gate.parameters()]}

    # ------------------------------------------------------------------
    # 索引缓存
    # ------------------------------------------------------------------
    def precompute(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                   params: Optional[dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """用（可选冻结的）参数预计算 top-k 索引并缓存，返回索引 [B,T,topk]。

        params=None 时用当前 live 参数；传 snapshot() 的结果即「用 θ_{t−Δ} 提前算」。
        """
        if params is not None:
            # 临时把 live gate 权重换成冻结副本算索引（只读，不修改 moe 状态）
            saved = [p.data.clone() for p in self.moe.gate.parameters()]
            with torch.no_grad():
                for p, s in zip(self.moe.gate.parameters(), params["gate"]):
                    p.data.copy_(s)
                indices = self._route_indices(x, input_ids)
                for p, s in zip(self.moe.gate.parameters(), saved):
                    p.data.copy_(s)
        else:
            indices = self._route_indices(x, input_ids)
        key = _input_signature(x, input_ids)
        self._cache[key] = indices
        return indices

    def lookup(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        """命中缓存返回缓存索引；未命中则现场用当前参数计算。"""
        key = _input_signature(x, input_ids)
        if key in self._cache:
            return self._cache[key]
        return self._route_indices(x, input_ids)

    def _route_indices(self, x: torch.Tensor,
                       input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """取 top-k 索引（不含偏置的纯路由，供缓存/复用）。"""
        with torch.no_grad():
            topk_idx, _, _ = self.moe._route(x, input_ids)
        return topk_idx

    # ------------------------------------------------------------------
    # loss-spike 自动检测 + 回退状态机
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """当前是否处于 anticipatory 模式（loss spike 后触发、revert_after 步后回退）。"""
        return self._active

    def observe_loss(self, loss: float) -> None:
        """喂一个训练步的 loss，更新 EMA 并做 spike 检测 / 回退。

        注意 spike 判定用「更新前的 EMA」：先用旧 EMA 判断当前 loss 是否突增，
        再把当前 loss 折进 EMA。若先用新 EMA 判定，一次 spike 会把 EMA 拉高，
        反而不满足 loss > EMA×ratio（自我掩盖）。
        """
        if self._ema is None:
            self._ema = loss
            return

        if self._active:
            # 回退倒计时
            self._spike_steps_left -= 1
            if self._spike_steps_left <= 0:
                self._active = False
            # 即便处于 active，也继续更新 EMA（否则回退后 EMA 是旧的）
            self._ema = self.ema_alpha * loss + (1 - self.ema_alpha) * self._ema
            return

        prev_ema = self._ema
        self._ema = self.ema_alpha * loss + (1 - self.ema_alpha) * prev_ema
        if loss > prev_ema * self.spike_ratio and prev_ema > 0:
            self._active = True
            self._spike_steps_left = self.revert_after
