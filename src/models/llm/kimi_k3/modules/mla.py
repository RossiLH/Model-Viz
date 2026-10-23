# -*- coding: utf-8 -*-
"""Gated Multi-head Latent Attention（门控多头潜在注意力，K3 报告 §2.1.2）。

解决的问题：标准 softmax 注意力的 KV cache 随序列长度线性增长。在 1M 上下文下，
cache 显存与带宽会成为瓶颈。MLA 的核心思想是**压缩 + 重构**：把每个 token 的
key/value 先压进一个低维潜在向量 ``c = W_dkv x``（``kv_lora_rank`` 维），只在
cache 里存这个压缩后的 latent；真正参与注意力时再用学到的 up 投影把每 head 的
k/v 展开出来。于是 cache 大小由 ``kv_lora_rank`` 决定，而不是
``num_heads * head_dim``。

K3 在继承 DeepSeek-V2 的 MLA 基础上做了两处改动：

1. **NoPE（完全不施加位置编码）**：MLA 层不应用 RoPE，也没有解耦的 rope head
   维。位置信息完全由与之交错的 KDA 层提供 —— KDA 的递推衰减和 delta 更新天然
   位置敏感（recency 由衰减累积编码）。好处是：把上下文扩到 1M **不需要任何
   位置参数手术**（不用调 frequency base、不用 YaRN）。

2. **全秩输出门（Eq. 7）**：``y = W_o [ sigmoid(W_g x) * o ]``，让每个 token 按
   其输入调制它从全局注意力中读取哪些通道。这里 ``W_g`` 是**全秩**的（输出维等于
   拼接后的 v 维），而非低秩门 —— 低秩门在 KDA 上换成了 head-wise RMSNorm。

数值：训练时注意力输出在 FP32 累加（``attn_out_fp32``），用于修正 flash
attention 的偏向舍入误差（biased rounding）。

本项目实现与 ``kimi3-master/kimi3/modules/mla.py`` 逐位对齐（同公式、同投影命名、
同 cache 结构）；SGLang 在独立 ``kimi-k3`` 分支里把输出门融合成单 CUDA kernel
（``mla_output_gate.cuh``：``out = x * sigmoid(gate)``）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MLAConfig
from .norm import RMSNorm

__all__ = ["GatedMLA", "MLACache"]


class MLACache:
    """解码缓存：**只存压缩后的 latent**。

    这是 MLA 的意义所在：cache 随 ``kv_lora_rank``（如 512）增长，而不是随
    ``num_heads * head_dim``（如 96*128=12288）增长。相比之下标准 attention 的
    KV cache 每个 token 都要存原始 K/V，MLA 只需存压缩表示，省下约一个数量级。

    ``kv_latent`` 形状 ``(batch, seq, kv_lora_rank)``；``append`` 沿 seq 维追加
    新 latent 并返回完整历史（供当前步展开全部 k/v）。
    """

    __slots__ = ("kv_latent", "length")

    def __init__(self, kv_latent: Tensor, length: int = 0):
        self.kv_latent = kv_latent
        self.length = length

    def append(self, latent: Tensor) -> Tensor:
        """追加 ``(batch, new, rank)``，返回完整 latent 历史。"""
        self.kv_latent = (
            latent if self.kv_latent is None else torch.cat([self.kv_latent, latent], dim=1)
        )
        self.length = self.kv_latent.shape[1]
        return self.kv_latent

    def clone(self) -> "MLACache":
        """深拷贝本 cache（投机解码 / beam search 需要在候选续写间分支状态）。

        与 ``KDACache.clone`` 语义一致：返回独立副本，改动副本不影响原对象。
        MLA 的 cache 只是 ``kv_latent`` 一个压缩 latent 张量，深拷贝即可。
        """
        return MLACache(
            None if self.kv_latent is None else self.kv_latent.clone(),
            length=self.length,
        )

    def truncate(self, length: int) -> None:
        """原地截断 latent 到 ``length`` 个位置（回滚到被接受前缀末尾）。

        投机解码部分接受时，草稿被拒绝的位置对应的 latent 必须丢弃，否则
        下一轮会把“未验证的续写”当成历史。MLA 是逐 token 追加的压缩 latent，
        直接按 seq 维切回 ``length`` 即可精确回滚。
        """
        if self.kv_latent is not None:
            self.kv_latent = self.kv_latent[:, :length]
        self.length = length


class GatedMLA(nn.Module):
    """完整的门控 MLA 层。

    参数结构（§2.1.2）：
    - ``q_down / q_norm / q_up``（q-LoRA 分支）：query 也走低秩压缩
      ``dim -> q_lora_rank -> qk_dim``；若 ``q_lora_rank is None`` 则直接用
      ``q_proj: dim -> qk_dim`` 全秩投影。
    - ``kv_down / kv_norm / k_up / v_up``：key/value 的压缩-重构。``kv_down``
      是唯一的压缩投影（``dim -> kv_lora_rank``），``k_up / v_up`` 在注意力时
      把 latent 展开成每 head 的 qk 维 / v 维。
    - ``g_proj``：**全秩输出门** ``sigmoid(W_g x)``（Eq. 7）。
    - ``o_proj``：最终输出投影 ``v_dim -> dim``。

    数据流（forward）：
        q = W_upq(RMSNorm(W_dq x))                  (b,h,t,dqk)
        c = RMSNorm(W_dkv x)                        (b,t,kv_lora)  ← 唯一 cache
        k = W_uk(c)  (b,h,t,dqk)  ;  v = W_uv(c)  (b,h,t,dv)
        o = softmax(q k^T / sqrt(dqk)) v            (b,h,t,dv)  [fp32 累加]
        y = W_o [ sigmoid(W_g x) * o ]              (b,t,dim)
    """

    def __init__(self, dim: int, cfg: MLAConfig, *, layer_idx: int = 0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.layer_idx = layer_idx
        h = cfg.num_heads
        self.num_heads = h
        self.qk_head_dim = cfg.qk_head_dim
        self.v_head_dim = cfg.v_head_dim
        # 注意力缩放因子 1/sqrt(qk_head_dim)。
        self.scale = cfg.qk_head_dim**-0.5

        # query 投影：可选 q-LoRA 低秩分支，或全秩直连。
        if cfg.q_lora_rank is None:
            self.q_proj = nn.Linear(dim, cfg.qk_dim, bias=False)
            self.q_down = self.q_up = self.q_norm = None
        else:
            self.q_proj = None
            self.q_down = nn.Linear(dim, cfg.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(cfg.q_lora_rank, eps)
            self.q_up = nn.Linear(cfg.q_lora_rank, cfg.qk_dim, bias=False)

        # key/value 的压缩与重构。
        self.kv_down = nn.Linear(dim, cfg.kv_lora_rank, bias=False)  # W_dkv：压缩
        self.kv_norm = RMSNorm(cfg.kv_lora_rank, eps)
        self.k_up = nn.Linear(cfg.kv_lora_rank, cfg.qk_dim, bias=False)  # W_uk
        self.v_up = nn.Linear(cfg.kv_lora_rank, cfg.v_dim, bias=False)   # W_uv

        self.g_proj = nn.Linear(dim, cfg.v_dim, bias=False)   # 全秩门，Eq. 7
        self.o_proj = nn.Linear(cfg.v_dim, dim, bias=False)

    def _queries(self, x: Tensor) -> Tensor:
        """计算 query。q-LoRA 分支：``W_upq(RMSNorm(W_dq x))``；否则 ``W_q x``。"""
        if self.q_proj is not None:
            return self.q_proj(x)
        assert self.q_down is not None and self.q_norm is not None and self.q_up is not None
        return self.q_up(self.q_norm(self.q_down(x)))

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        cache: MLACache | None = None,
    ) -> Tensor:
        b, t, _ = x.shape
        h, dqk, dv = self.num_heads, self.qk_head_dim, self.v_head_dim

        # query -> (b, h, t, dqk)。
        q = self._queries(x).view(b, t, h, dqk).transpose(1, 2)

        # 压缩 latent；有 cache 则追加历史（decode 时只传当前 token，靠 append 累积）。
        latent = self.kv_norm(self.kv_down(x))
        if cache is not None:
            latent = cache.append(latent)
        kv_len = latent.shape[1]

        # 从 latent 展开每 head 的 k/v。
        k = self.k_up(latent).view(b, kv_len, h, dqk).transpose(1, 2)
        v = self.v_up(latent).view(b, kv_len, h, dv).transpose(1, 2)

        # 训练时在 FP32 累加注意力，修正低精度舍入偏置。
        if self.cfg.attn_out_fp32 and self.training:
            q, k, v = q.float(), k.float(), v.float()

        is_causal = attn_mask is None and kv_len == t
        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.scale
            )
        else:
            # torch < 2.0（本仓库 CPU 环境）：手动因果 softmax 注意力。
            attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
            if is_causal:
                mask = torch.triu(torch.full((t, kv_len), float('-inf'), device=x.device), diagonal=1)
                attn = attn + mask
            elif attn_mask is not None:
                attn = attn + attn_mask
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
        # (b,h,t,dv) -> (b,t,h*dv)，降回输入 dtype。
        out = out.transpose(1, 2).reshape(b, t, h * dv).to(x.dtype)

        # Eq. 7：依赖输入的、逐通道的、全秩输出门。
        # sigmoid(W_g x) 逐通道在 (0,1)，决定该 token 从注意力输出里读取多少。
        out = out * torch.sigmoid(self.g_proj(x))
        return self.o_proj(out)

    def init_cache(self, batch: int, *, device=None, dtype=torch.float32) -> MLACache:
        """为 decode 初始化空 latent cache：``(batch, 0, kv_lora_rank)``。"""
        return MLACache(
            torch.zeros(batch, 0, self.cfg.kv_lora_rank, device=device, dtype=dtype)
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, qk={self.qk_head_dim}, "
            f"v={self.v_head_dim}, q_lora={self.cfg.q_lora_rank}, "
            f"kv_lora={self.cfg.kv_lora_rank}, nope=True"
        )
