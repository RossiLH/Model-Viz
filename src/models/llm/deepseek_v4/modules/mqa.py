from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.interfaces import AttnModule
from models.llm.deepseek_v4.config import MQAV4Config


class GroupedLinear(nn.Module):
    """分组低秩线性层（Grouped Output Projection 的前置块，D4-A1#6）。

    论文（DeepSeek-V4 §2.3.1）：把 ``n_h`` 个 Q 头的输出切成 ``g`` 组，
    每组独立投影到 ``d_g`` 维瓶颈（``d_g < n_h * head_dim / g``），
    最后拼接再过 ``o_b_proj`` 投回 ``dim``。

    数学上等价于块对角矩阵 ``W = diag(W^G_1, ..., W^G_g)``：
    跨组零耦合（组间无信息交换），跨组混合由 ``o_b_proj`` 完成。

    参考实现：``Reference/DeepSeekV4/nano-deepseek-v4-main/nano_deepseek_v4/modeling.py:81``

    Args:
        in_features: 输入总宽（= attention_width = n_heads * head_dim）
        out_features_per_group: 每组输出瓶颈宽（= o_lora_rank，即论文 d_g）
        groups: 分组数 g
    """

    def __init__(self, in_features: int, out_features_per_group: int, groups: int) -> None:
        super().__init__()
        if in_features % groups != 0:
            raise ValueError("in_features 必须能被 groups 整除")
        self.groups = groups
        self.in_per_group = in_features // groups
        self.out_per_group = out_features_per_group
        # weight 形状 [g, d_g, in_per_group]：g 个独立的低秩子矩阵
        self.weight = nn.Parameter(
            torch.empty(groups, out_features_per_group, self.in_per_group))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_features] -> [..., groups * out_per_group]。"""
        *prefix, width = x.shape
        if width != self.groups * self.in_per_group:
            raise ValueError(
                f"期望输入宽 {self.groups * self.in_per_group}，实际 {width}")
        x = x.reshape(-1, self.groups, self.in_per_group)
        # einsum "ngi,goi->ngo"：每组独立做线性投影（block-diagonal）
        y = torch.einsum("ngi,goi->ngo", x, self.weight)
        return y.reshape(*prefix, self.groups * self.out_per_group)


class MQAV4Attention(AttnModule):
    """DeepSeek-V4 的 MQA 注意力基底（滑窗无压缩层 + 各压缩层共享基座）。

    与标准 MHA（``core/standard/mha.py``）的差异（对应 D4-A1，以 SGLang 语义为准）：
    1. **MQA**：``n_kv_heads=1``，所有 Q 头共享一份 K/V，KV cache 缩小 H 倍。
    2. **Q 低秩两段投影**：``q_a(dim→q_lora_rank) -> q_b(q_lora_rank→H·head_dim)``，
       用极少参数重建全宽 Q。
    3. **K/V 共享低秩瓶颈**：单次 ``kv_proj(dim→head_dim)`` 得 K/V 基底；
       K 带 nope+rope 两段（部分 RoPE），V 独立复用。**不需要逆 RoPE**
       （区别于 nano 的 V=K 技巧，见 D4-A1#3）。
    4. **Partial RoPE**：只旋转 head_dim 的后 ``qk_rope_head_dim`` 维。
    5. **Attention Sink**：每头一个可学习标量 logit，拼接在 softmax 末尾，
       吸收模型对无关上下文的过度注意力。

    本类承担「无压缩的滑窗注意力」层（compress_ratio=0）；HCA/CSA 通过子类
    或组合在 ``forward`` 里追加压缩 KV 分支（见 hca.py / csa.py）。
    """

    def __init__(self, config: MQAV4Config):
        super().__init__()
        self.config = config
        config.validate()

        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.window_size = config.window_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.attention_width = config.attention_width

        # Q 低秩两段投影（D4-B2：以 nano/SGLang 的 q_a/q_b 为准）
        if config.q_lora_rank:
            self.q_a_proj = nn.Linear(self.dim, config.q_lora_rank, bias=False)
            self.q_a_norm = nn.RMSNorm(config.q_lora_rank, eps=1e-6)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.attention_width, bias=False)
        else:
            # 退化：无低秩（q_lora_rank=0），单次全宽投影
            self.q_a_proj = None
            self.q_a_norm = None
            self.q_b_proj = nn.Linear(self.dim, self.attention_width, bias=False)

        # K/V 共享低秩瓶颈（MQA）：dim -> head_dim
        self.kv_proj = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

        # Attention sink：每头一个标量 logit（D4-B7，nano/SGLang 标量形式）
        if config.use_attention_sink:
            self.attention_sink = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.attention_sink = None

        # 输出投影（Grouped Output Projection，D4-A1#6，与 nano/SGLang 对齐）
        # o_a_proj：GroupedLinear，g 个独立的 (attention_width/g → o_lora_rank) 子矩阵
        # o_b_proj：把 g * o_lora_rank 的中间瓶颈投回 dim
        o_g = config.resolved_o_groups
        self.o_a_proj = GroupedLinear(
            self.attention_width, config.o_lora_rank, o_g)
        self.o_b_proj = nn.Linear(
            o_g * config.o_lora_rank, self.dim, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

        # 预计算 cos/sin（对 nope 部分不旋转，只旋转 rope 部分）
        self._register_buffer = None  # 占位，避免误用
        self._cos_sin_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Partial RoPE
    # ------------------------------------------------------------------
    def _rope_cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """预计算 (cos, sin)，shape [seq_len, qk_rope_head_dim/2]。"""
        if self._cos_sin_cache is not None and self._cos_sin_cache[0].shape[0] >= seq_len:
            return self._cos_sin_cache
        theta = 1.0 / (self.config.rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, device=device, dtype=torch.float32)
            / self.qk_rope_head_dim))
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * theta.unsqueeze(0)  # [seq, dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        self._cos_sin_cache = (cos, sin)
        return cos, sin

    def _apply_partial_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                            negate_sin: bool = False) -> torch.Tensor:
        """对 x[..., -rope_dim:] 做（interleaved）RoPE，前部不变。

        Args:
            x: [B, H, T, head_dim]
            cos/sin: [T, rope_dim/2]
            negate_sin: True 时用 -sin（逆旋转）
        """
        rope_dim = self.qk_rope_head_dim
        if rope_dim == 0:
            return x
        # cos/sin: [T, rope_dim/2] -> [1, 1, T, rope_dim]（与 x[B,H,T,Dh] 广播）
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,T,rope_dim]
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
        if negate_sin:
            sin = -sin
        nope, rope = x[..., : self.nope_head_dim], x[..., self.nope_head_dim:]
        # interleaved rotate_half
        even = rope[..., 0::2]
        odd = rope[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)  # [B,H,T,rope_dim]
        rope = rope * cos + rotated * sin
        return torch.cat([nope, rope], dim=-1)

    # ------------------------------------------------------------------
    # 注意力
    # ------------------------------------------------------------------
    def _core_attention(self, q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """标准缩放点积注意力（带 attention sink 与滑窗 mask）。

        Args:
            q: [B, H, T, head_dim]
            keys/values: [B, H, Tk, head_dim]
            mask: [B, 1, T, Tk] 布尔掩码（True=可见）
        """
        B, H, T, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), keys.transpose(-1, -2).float()) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # attention sink：每头一个标量 logit，拼在末尾
        if self.attention_sink is not None:
            sink = self.attention_sink.view(1, H, 1, 1).expand(B, H, T, 1)
            scores = torch.cat([scores, sink.float()], dim=-1)

        probs = F.softmax(scores, dim=-1).to(values.dtype)
        probs = self.dropout(probs)
        if self.attention_sink is not None:
            values = torch.cat([values, torch.zeros_like(values[..., :1, :])], dim=-2)
        return torch.matmul(probs[..., : values.shape[-2]], values)

    def _local_mask(self, q_pos: torch.Tensor, k_pos: torch.Tensor,
                    attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """滑窗 + 因果掩码。q_pos/k_pos 为位置（[B,T] 或 [T]）。"""
        # k_pos <= q_pos（因果）且 k_pos >= q_pos - window + 1（滑窗）
        mask = (k_pos.unsqueeze(1) <= q_pos.unsqueeze(-1)) & (
            k_pos.unsqueeze(1) >= q_pos.unsqueeze(-1) - self.window_size + 1)
        if attention_mask is not None and attention_mask.shape[-1] == k_pos.shape[-1]:
            mask = mask & attention_mask.bool().unsqueeze(1)
        return mask  # [B, T, Tk]

    def _trim_window(self, cache: dict, q_pos: torch.Tensor) -> None:
        """把滑窗缓存裁剪到 window_size（**无损**，C8 的省显存关键）。

        因为 `_local_mask` 完全由 *位置* 决定可见性，且后续 query 位置只增不减，
        所以对「本 chunk 最早的 query」都已不可见的 key，对未来所有 query 也
        永远不可见 —— 可以安全丢弃。

        注意必须用 ``q_pos.min()`` 而非 max：预填充时一个 chunk 内含多个 query
        （位置 0..T-1），若按 max 裁剪会把靠前 query 仍需要的 key 提前删掉，
        导致预填充结果错误（曾踩坑：decode/prefill 偏差从 1e-6 涨到 2.07）。
        解码时 T=1、min==max，仍能把缓存压到 O(window_size)。
        """
        k_pos = cache["pos"]
        if k_pos.shape[1] == 0:
            return
        cutoff = int(q_pos.min().item()) - self.window_size + 1
        if cutoff <= int(k_pos[0, 0].item()):
            return  # 最旧的 token 仍在窗口内，无需裁剪
        keep = (k_pos[0] >= cutoff).nonzero(as_tuple=True)[0]
        if keep.numel() == k_pos.shape[1]:
            return
        start = int(keep[0].item()) if keep.numel() > 0 else k_pos.shape[1]
        cache["k"] = cache["k"][:, :, start:, :]
        cache["v"] = cache["v"][:, :, start:, :]
        cache["pos"] = cache["pos"][:, start:]

    # ------------------------------------------------------------------
    # AttnModule 接口
    # ------------------------------------------------------------------
    def init_cache(self, batch: int) -> dict:
        """初始化推理缓存：MQA 只有一份 K/V（n_kv_heads=1）。"""
        return {
            "k": torch.zeros(batch, 1, 0, self.head_dim),
            "v": torch.zeros(batch, 1, 0, self.head_dim),
            "pos": torch.zeros(batch, 0, dtype=torch.long),
        }

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """MQA 滑窗注意力前向。

        Args:
            x: [B, T, dim]
            cache: 推理缓存 dict（可含历史 K/V 与位置）
            position_ids: [B, T] 或 [T]；None 时默认 0..T-1
            attention_mask: [B, T] 布尔（True=有效 token）

        Returns:
            out: [B, T, dim]
        """
        B, T, D = x.shape
        assert D == self.dim

        # Q 低秩两段
        if self.q_a_proj is not None:
            q_mid = self.q_a_norm(self.q_a_proj(x))
            q = self.q_b_proj(q_mid).view(B, T, self.n_heads, self.head_dim)
        else:
            q = self.q_b_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B, H, T, Dh]

        # K/V 共享低秩
        kv = self.kv_norm(self.kv_proj(x)).unsqueeze(1)  # [B, 1, T, Dh]（MQA）

        # 位置
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)  # [B, T]

        # Partial RoPE：Q 与 K（按 position_ids 取 cos/sin，解码时位置可跳变）
        max_pos = position_ids.max().item() + 1
        cos, sin = self._rope_cos_sin(max_pos, x.device)
        pos0 = position_ids[0]  # [T]（batch 共享）
        cos_q, sin_q = cos[pos0], sin[pos0]
        q = self._apply_partial_rope(q, cos_q, sin_q)
        k = self._apply_partial_rope(kv, cos_q, sin_q)

        # 处理缓存（首次 cache={} 时视为无过去）
        if cache is not None and "k" in cache:
            past_k = cache["k"]
            past_v = cache["v"]
            past_pos = cache["pos"]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, kv], dim=2)
            q_pos = position_ids
            k_pos = torch.cat([past_pos, position_ids], dim=1)
            cache["k"] = k
            cache["v"] = v
            cache["pos"] = k_pos
            self._trim_window(cache, q_pos)
            k, v, k_pos = cache["k"], cache["v"], cache["pos"]
        else:
            v = kv
            q_pos = position_ids
            k_pos = position_ids
            if cache is not None:
                cache["k"] = k
                cache["v"] = v
                cache["pos"] = k_pos
                self._trim_window(cache, q_pos)
                k, v, k_pos = cache["k"], cache["v"], cache["pos"]

        # 广播 MQA：K/V 复制到所有头
        k_exp = k.expand(B, self.n_heads, -1, self.head_dim)
        v_exp = v.expand(B, self.n_heads, -1, self.head_dim)

        # 掩码
        mask = self._local_mask(q_pos, k_pos, attention_mask)  # [B, T, Tk]

        context = self._core_attention(q, k_exp, v_exp, mask)
        # 无逆 RoPE：V 就是未旋转的 kv（SGLang 语义，D4-A1#3）
        context = context.transpose(1, 2).reshape(B, T, self.attention_width)

        out = self.o_b_proj(self.o_a_proj(context))
        return out
