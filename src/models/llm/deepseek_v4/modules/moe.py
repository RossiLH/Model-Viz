from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.interfaces import MoEModule
from models.llm.deepseek_v4.config import MoEV4Config


class SwiGLUExpert(nn.Module):
    """SwiGLU 专家：gate_up 融合投影 + down 投影（带 SwiGLU Clamping）。

    SwiGLU Clamping（0x0F，论文 §4.2 / nano modeling.py:853-855 语义）：
    - **线性分量 up** 钳到 [−swiglu_limit, swiglu_limit]（论文「clamp the linear
      component of SwiGLU to [−10,10]」）；
    - **门控分量 gate** 在 **silu 之前**钳上界 swiglu_limit（论文「cap the upper
      bound of the gate component at 10」）。
    钳位发生在激活前，是为了在 FP8/FP4 低精度下消除异常值、稳定训练；
    swiglu_limit=0 时完全跳过钳位（基线路径）。
    """

    def __init__(self, dim: int, hidden: int, swiglu_limit: float = 10.0):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        if self.limit > 0:
            # 先钳 pre-activation（nano 语义）：gate 上界 10、up 上下界 [−10,10]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        gate = F.silu(gate)
        return self.down(gate * up)


class DeepSeekV4MoE(MoEModule):
    """DeepSeek-V4 MoE 路由（D4-A6，以 SGLang/nano 语义为准）。

    两处关键设计：
    1. **打分 sqrt(softplus)**：`s = sqrt(softplus(W_r·x))`，无界、非竞争
       （区别于 softmax 的零和竞争），探针06实测其大 logit 梯度不饱和。
    2. **aux-loss-free 偏置**（noaux_tc）：`e_score_correction_bias` 只影响
       top-k 的"选择"，不进最终权重 → 无梯度偏差、不干扰主目标。
       过载专家偏置下调、空闲专家上调（探针06验证）。
    3. **Hash 路由**：前 num_hash_layers 层按 token-id 哈希（确定性冷启动）。
       ``hash_stride>0`` 用取模+stride；``hash_stride=0`` 用 LCG 静态表。
    """

    def __init__(self, config: MoEV4Config, dim: int, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.dim = dim
        self.layer_idx = layer_idx
        self.num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.is_hash_layer = layer_idx < config.num_hash_layers

        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(dim, config.moe_intermediate_size,
                         swiglu_limit=config.swiglu_limit)
            for _ in range(self.num_experts)
        ])
        # shared experts：对所有 token 都计算
        self.shared_experts = nn.ModuleList([
            SwiGLUExpert(dim, config.moe_intermediate_size,
                         swiglu_limit=config.swiglu_limit)
            for _ in range(config.n_shared_experts)
        ])
        # aux-loss-free 偏置缓冲
        self.register_buffer("e_score_correction_bias",
                             torch.zeros(self.num_experts))
        # LCG 哈希表（hash_stride=0 时用）
        if config.hash_stride == 0:
            self.tid2eid: Optional[torch.Tensor] = None  # 由外部注入
        else:
            self.tid2eid = None

    # ------------------------------------------------------------------
    # 打分
    # ------------------------------------------------------------------
    def _affinity(self, logits: torch.Tensor) -> torch.Tensor:
        """sqrt(softplus(logits)) —— 无界、非竞争的打分（D4-B3）。"""
        if self.config.scoring_func == "sqrtsoftplus":
            return torch.sqrt(F.softplus(logits))
        if self.config.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        raise ValueError(f"未知 scoring_func: {self.config.scoring_func}")

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def _hash_route(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hash 路由：返回 [B,T,topk] 的 expert 索引。

        hash_stride>0：`(token_id + r*stride) % n_experts`（mini 风格）。
        hash_stride=0：查 LCG 表 tid2eid[token_id, r]（nano 风格，需注入）。
        """
        B, T = input_ids.shape
        if self.config.hash_stride > 0:
            offsets = torch.arange(self.topk, device=input_ids.device) * self.config.hash_stride
            return (input_ids[..., None] + offsets[None, None, :]) % self.num_experts
        # LCG 静态表
        # 注意：表按「见过的最大 token id」惰性建/扩表。必须在 id 超出表长时
        # **扩表**，否则第一个 batch 的 max id 会永久决定表大小，后续出现更大
        # 的 token id 就 IndexError（曾踩坑：vocab=1000 但首 batch max=421）。
        need = int(input_ids.max().item()) + 1
        if self.tid2eid is None or self.tid2eid.shape[0] < need:
            ids = torch.arange(need, device=input_ids.device).unsqueeze(1)
            offs = torch.arange(self.topk, device=input_ids.device).unsqueeze(0)
            self.tid2eid = (ids * 1103515245 + 12345 + offs * 2654435761) % self.num_experts
        return self.tid2eid[input_ids]  # [B,T,topk]

    def _route(self, x: torch.Tensor, input_ids: Optional[torch.Tensor],
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (topk_idx, topk_weights, logits)。

        topk_idx: [B,T,topk]；topk_weights: [B,T,topk]（归一化+缩放）
        """
        B, T, D = x.shape
        logits = self.gate(x)
        affinity = self._affinity(logits)

        if self.is_hash_layer:
            topk_idx = self._hash_route(input_ids)
        else:
            selection = affinity + self.e_score_correction_bias
            topk_idx = selection.topk(self.topk, dim=-1).indices

        # 权重用 affinity（不含 bias → 偏置纯选择、不进权重）
        weights = affinity.gather(-1, topk_idx)
        if self.config.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights * self.config.routed_scaling_factor
        return topk_idx, weights, logits

    # ------------------------------------------------------------------
    # 均衡偏置更新
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_balance_bias(self, topk_idx: torch.Tensor) -> None:
        """aux-loss-free 偏置更新（D4-A6#23，nano update_balance_bias）。

        counts 偏高的专家（过载）偏置下调、偏低的（空闲）上调。
        只影响选择、不进梯度。
        """
        counts = torch.bincount(topk_idx.reshape(-1),
                                minlength=self.num_experts).float()
        if counts.sum() == 0:
            return
        target = counts.mean()
        adjustment = (target - counts) / target.clamp_min(1.0)
        self.e_score_correction_bias.add_(
            adjustment * self.config.router_bias_update_speed)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None,
                return_router_logits: bool = False,
                anticipatory_router=None) -> torch.Tensor:
        """MoE 前向。

        Args:
            x: [B,T,D]
            input_ids: 用于 hash 路由（前 num_hash_layers 层需要）
            return_router_logits: True 时返回 (out, router_logits)
            anticipatory_router: AnticipatoryRouter 实例（0x0F）。非 None 时，
                若命中其缓存索引则用缓存 topk（跳过 live 路由），实现
                「提前算好的路由索引直接复用」；默认 None = 标准路由不变。

        Returns:
            out: [B,T,D]；或 (out, router_logits)
        """
        B, T, D = x.shape
        if self.is_hash_layer and input_ids is None:
            raise ValueError("hash_moe 路由需要 input_ids")

        # Anticipatory Routing：命中缓存 → 直接复用缓存 topk（不重算路由）
        cached_idx = None
        if anticipatory_router is not None and not self.is_hash_layer:
            cached_idx = anticipatory_router.lookup(x, input_ids)
        if cached_idx is not None:
            topk_idx = cached_idx
            with torch.no_grad():
                logits = self.gate(x)
                affinity = self._affinity(logits)
                weights = affinity.gather(-1, topk_idx)
            if self.config.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            weights = weights * self.config.routed_scaling_factor
        else:
            topk_idx, weights, logits = self._route(x, input_ids)

        flat = x.reshape(-1, D)
        flat_idx = topk_idx.reshape(-1, self.topk)
        flat_weights = weights.reshape(-1, self.topk)

        out = flat.new_zeros(flat.shape)
        for eid, expert in enumerate(self.experts):
            route_mask = flat_idx == eid
            token_mask = route_mask.any(dim=-1)
            if not token_mask.any():
                continue
            # 该 token 在此专家上的权重（若有多个路由命中同一专家，取和）
            w = flat_weights.masked_fill(~route_mask, 0.0).sum(dim=-1)
            out[token_mask] += w[token_mask, None] * expert(flat[token_mask])

        # shared experts：所有 token 都计算
        for se in self.shared_experts:
            out += se(flat)

        out = out.view(B, T, D)
        if return_router_logits:
            return out, logits
        return out
