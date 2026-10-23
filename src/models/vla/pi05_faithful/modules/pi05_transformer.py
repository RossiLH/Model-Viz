"""PI0.5 Transformer 模块

本模块实现了 PI0.5 的 Transformer 层，对应官方 openpi-main 中的 gemma.py。

官方实现参考：
- 文件: Reference/openpi-main/src/openpi/models/gemma.py
- 类: TransformerBlock (官方 GemmaBlock)
- 门控残差: gemma.py:453 _gated_residual
- adaRMS: gemma.py:113-131 RMSNorm
- FFN (GeGLU): gemma.py:253-271 FeedForward

PI0.5 相比标准 Transformer 的关键创新：
1. adaRMSNorm 条件化：时间步经 adaRMSNorm 注入（scale/shift/gate 调制）
   **注意：官方 pi0.py:210 传 adarms_cond=[None, adarms_cond] —— 只有 action expert
   被条件化，VLM（prefix）传 None。**
2. 门控残差连接：x + y * gate，其中 gate 由 adaRMSNorm 从时间步 cond 产生
   （官方 gemma.py:129-131 的 gate + gemma.py:453 _gated_residual）
3. 双专家联合处理：PaliGemma 和 Action Expert 在每一层共享联合注意力

关键对齐（论文 arXiv:2504.16054 附录 A-E）：
- KV cache 必须逐层独立（每层有自己的一份 cache，禁止跨层传递）
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# 复用 VLA core 中的 AdaRMSNorm（相对导入，无需 sys.path 干预）
from ...core.norms import AdaRMSNorm

from .dual_expert_attention import DualExpertJointAttention, make_attn_mask


class GeGLU(nn.Module):
    """Gated GELU MLP（对应官方 gemma.py:253-271 FeedForward 的 GeGLU 激活）

    官方结构:
        gate = gelu(x @ w_gating)      # 门控分支
        up = x @ w1                    # 升维分支
        out = (gate * up) @ w2         # 逐元素相乘后降维

    比普通 GELU 多一个"门控"分支，是 Gemma/PaliGemma 的标准 FFN 激活。
    """

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, mlp_dim, bias=False) # 门控分支
        self.up = nn.Linear(dim, mlp_dim, bias=False)   # 升维分支
        self.down = nn.Linear(mlp_dim, dim, bias=False) # 降维投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.gelu(self.gate(x))
        up = self.up(x)
        return self.down(gate * up) # gate ⊙ up，再投影回 dim


class RMSNormFinal(nn.Module):
    """最终 RMSNorm（对应官方 gemma.py:410 的 final_norm，仍是 RMSNorm）

    官方最终 norm 是 RMSNorm（非 LayerNorm），且接受 adarms_cond。
    这里对 suffix 使用条件化、对 prefix 使用无条件（官方 pi0.py:410
    final_norms 分别以各自 adarms_cond 调用）。
    """

    def __init__(self, dim: int, cond_dim: int | None = None):
        super().__init__()
        self.norm = AdaRMSNorm(dim, cond_dim=cond_dim, return_gate=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(x, cond)


class Pi05TransformerBlock(nn.Module):
    """PI0.5 Transformer Block

    对应官方: GemmaBlock (Reference/openpi-main/src/openpi/models/gemma.py)

    每个 Block 包含:
    - 输入 adaRMSNorm（**仅 suffix 条件化**，prefix 传 None）
    - 双专家联合注意力（块掩码，单向流）
    - 门控残差连接（gate 来自 adaRMSNorm 的条件调制）
    - 后注意力 adaRMSNorm（**仅 suffix 条件化**）
    - FFN（PaliGemma 和 Action Expert 各自独立的 GeGLU）
    - 门控残差连接

    官方结构:
        x_norm, gate = adaRMS(x, cond)
        attn_out = joint_attention(x_norm)
        x = x + attn_out * gate          # 门控残差（gate 由 cond 产生）
        x_norm, gate = adaRMS(x, cond)
        ffn_out = ffn(x_norm)
        x = x + ffn_out * gate           # 门控残差
    """

    def __init__(self, config):
        """初始化 Transformer Block

        对应官方: GemmaBlock.__init__()

        Args:
            config: Pi05FaithfulConfig
        """
        super().__init__()

        self.config = config

        # ===== 输入 adaRMSNorm（条件化） =====
        # 对应官方: RMSNorm(cond=adarms_cond)，通过 cond_proj 生成 scale/shift/gate
        # prefix 用无条件（官方 adarms_cond=[None, ...]），suffix 用条件化
        self.input_norm_prefix = AdaRMSNorm(
            dim=config.paligemma_width,  # 无条件（cond=None）
        )
        self.input_norm_suffix = AdaRMSNorm(
            dim=config.action_width,
            cond_dim=config.action_width,  # adaRMS cond 宽度 = action expert 宽度
            return_gate=True,
        )

        # ===== 双专家联合注意力 =====
        self.attention = DualExpertJointAttention(config)

        # ===== 后注意力 adaRMSNorm（条件化） =====
        self.post_attn_norm_prefix = AdaRMSNorm(dim=config.paligemma_width)  # 无条件
        self.post_attn_norm_suffix = AdaRMSNorm(
            dim=config.action_width,
            cond_dim=config.action_width,
            return_gate=True,
        )

        # ===== Feed-Forward Network（各自独立，GeGLU） =====
        self.ffn_prefix = GeGLU(config.paligemma_width, config.paligemma_mlp_dim)
        self.ffn_suffix = GeGLU(config.action_width, config.action_mlp_dim)

    @staticmethod
    def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate) -> torch.Tensor:
        """门控残差（对应官方 gemma.py:453 _gated_residual）

        官方:
            def _gated_residual(x, y, gate):
                if gate is None:
                    return x + y      # 无 adaRMS 条件时退化为标准残差
                return x + y * gate   # 有条件时按 gate 缩放

        Args:
            x: 残差输入
            y: 子层输出
            gate: (B, 1, D) 门控（来自 adaRMSNorm 的 cond 调制）或 None

        Returns:
            残差结果
        """
        if gate is None:
            return x + y
        return x + y * gate

    def forward(
        self,
        prefix: Optional[torch.Tensor],
        suffix: torch.Tensor,
        timestep_cond: Optional[torch.Tensor],
        mask: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播

        对应官方: GemmaBlock.__call__()

        数据流:
            prefix, suffix: 输入
                | adaRMSNorm（prefix 无条件，suffix 用 cond）
            prefix_norm, (suffix_norm, gate_suffix)
                | 双专家联合注意力
            prefix_attn, suffix_attn
                | 门控残差（prefix 静态残差，suffix 用条件 gate）
            prefix, suffix
                | adaRMSNorm
            prefix_norm, (suffix_norm, gate_suffix)
                | FFN（各自独立 GeGLU）
            prefix_ffn, suffix_ffn
                | 门控残差
            prefix, suffix: 输出

        Args:
            prefix: (B, prefix_len, paligemma_width) PaliGemma token，可为 None（使用缓存）
            suffix: (B, suffix_len, action_width) Action Expert token
            timestep_cond: (B, action_width) 时间步条件（adaRMS 调制，仅用于 suffix）
            mask: (B, q_len, kv_len) 二维块掩码
            kv_cache: 可选的 (cached_k, cached_v)

        Returns:
            prefix_out: (B, prefix_len, paligemma_width) 或 None
            suffix_out: (B, suffix_len, action_width)
            new_kv_cache: (new_k, new_v) 本次新增的 KV
        """
        # ===== 输入 adaRMSNorm =====
        # prefix 无条件（官方 adarms_cond=[None, ...]），suffix 条件化并返回 gate
        prefix_norm = self.input_norm_prefix(prefix) if prefix is not None else None
        suffix_norm, gate_in = self.input_norm_suffix(suffix, cond=timestep_cond)

        # ===== 双专家联合注意力 =====
        prefix_attn, suffix_attn, new_kv_cache = self.attention(
            prefix_norm, suffix_norm, mask, kv_cache
        )

        # ===== 门控残差（注意力） =====
        # prefix：无条件时无 gate，退化为标准残差 x + y（官方 _gated_residual）
        # suffix：x + attn * gate_in（gate 由时间步 cond 产生）
        if prefix is not None:
            prefix = self._gated_residual(prefix, prefix_attn, None)
        suffix = self._gated_residual(suffix, suffix_attn, gate_in)

        # ===== 后注意力 adaRMSNorm =====
        prefix_norm = self.post_attn_norm_prefix(prefix) if prefix is not None else None
        suffix_norm, gate_ffn = self.post_attn_norm_suffix(suffix, cond=timestep_cond)

        # ===== Feed-Forward Network（各自独立 GeGLU） =====
        prefix_ffn = self.ffn_prefix(prefix_norm) if prefix_norm is not None else None
        suffix_ffn = self.ffn_suffix(suffix_norm)

        # ===== 门控残差（FFN） =====
        if prefix is not None:
            prefix = self._gated_residual(prefix, prefix_ffn, None)
        suffix = self._gated_residual(suffix, suffix_ffn, gate_ffn)

        return prefix, suffix, new_kv_cache


class Pi05Transformer(nn.Module):
    """PI0.5 Transformer

    对应官方: 由多个 GemmaBlock 堆叠而成，用于同时处理 prefix 和 suffix。

    关键：**KV cache 逐层独立**。每个 block 有自己的一份 cache，
    禁止把上一层的 cache 顺序传给下一层。
    """

    def __init__(self, config):
        """初始化 Transformer

        对应官方: TransformerBlock 堆叠 (Reference/openpi-main/src/openpi/models/gemma.py)

        Args:
            config: Pi05FaithfulConfig
        """
        super().__init__()

        self.config = config

        # 堆叠 Transformer blocks
        self.blocks = nn.ModuleList([
            Pi05TransformerBlock(config)
            for _ in range(config.paligemma_depth)
        ])

        # 最终 RMSNorm（官方 gemma.py:410 仍是 RMSNorm）
        self.final_norm_prefix = RMSNormFinal(config.paligemma_width)
        self.final_norm_suffix = RMSNormFinal(config.action_width, cond_dim=config.action_width)

    def forward(
        self,
        prefix: Optional[torch.Tensor],
        suffix: torch.Tensor,
        timestep_cond: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, list]:
        """前向传播

        Args:
            prefix: (B, prefix_len, paligemma_width) 或 None
            suffix: (B, suffix_len, action_width)
            timestep_cond: (B, action_width) 时间步条件
            mask: (B, q_len, kv_len) 二维块掩码
            kv_cache: list，长度 = depth，每项是 (cached_k, cached_v) 或 None

        Returns:
            prefix_out / suffix_out / new_kv_caches（list，逐层独立）
        """
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            # 逐层独立：取本层自己的 cache（第 i 层取 kv_cache[i]）
            layer_kv = None
            if kv_cache is not None:
                layer_kv = kv_cache[i]
            prefix, suffix, layer_new_kv = block(
                prefix, suffix, timestep_cond, mask, layer_kv
            )
            new_kv_caches.append(layer_new_kv)

        # 最终 RMSNorm（prefix 无条件，suffix 条件化）
        if prefix is not None:
            prefix = self.final_norm_prefix(prefix)
        suffix = self.final_norm_suffix(suffix, cond=timestep_cond)

        return prefix, suffix, new_kv_caches

    def prefix_forward(
        self,
        prefix: torch.Tensor,
    ) -> list:
        """仅编码 prefix，产出每层 KV cache（推理时只做一次，去噪各步复用）

        对应官方 pi0.py:237:  `_, kv_cache = llm([prefix_tokens, None], ...)`
        之后去噪循环内不再重算 prefix，只喂 suffix + 复用 cache。

        实现：以空 suffix（长度 0）跑一遍，每层只缓存 prefix 的 K/V。
        （suffix 为空时，所有 suffix 分支在 (B,0,W) 上运算并得到空张量，不影响结果。）

        Args:
            prefix: (B, prefix_len, paligemma_width)

        Returns:
            kv_caches: list，长度 = depth，每项 (k, v)，仅含 prefix 的 key/value
        """
        B, plen, _ = prefix.shape
        device = prefix.device
        p_mask = torch.ones(B, plen, dtype=torch.bool, device=device)
        mask = make_attn_mask(p_mask, None, plen, 0)  # (B, plen, plen)

        empty_suffix = torch.zeros(B, 0, self.config.action_width, device=device)

        kv_caches = []
        x = prefix
        for block in self.blocks:
            x, _, new_kv = block(x, empty_suffix, None, mask, None)
            kv_caches.append(new_kv)
        return kv_caches

    def prefix_fast_forward(
        self,
        combined: torch.Tensor,
        prefix_len: int,
    ):
        """三段式 Stage 1：prefix + FAST 联合前向，缓存仅 prefix 块 KV + 返回 FAST hidden

        对应设计规格 §2 / §4.5：
        - 输入 combined = cat([prefix_tokens, fast_embeds])，形状 (B, P+F, width)
        - 用三段掩码（prefix 双向 + FAST 因果）跑全部层
        - 每层缓存 **只取 prefix 块** 的 K/V（供连续动作 cross-attn，不含 FAST）
        - 返回最终 norm 后 FAST 位置的 hidden states（供 da_head 做 CE）

        Args:
            combined: (B, P+F, paligemma_width) prefix 与 FAST 的嵌入拼接
            prefix_len: prefix 块长度 P

        Returns:
            fast_hidden: (B, F, paligemma_width) FAST 位置最终输出
            prefix_kv_cache: list[(k, v)]，每项 (B, P, num_kv_heads, head_dim)
        """
        from .three_segment import make_three_segment_mask

        B, total_len, _ = combined.shape
        fast_len = total_len - prefix_len
        device = combined.device

        p_mask = torch.ones(B, prefix_len, dtype=torch.bool, device=device)
        f_mask = torch.ones(B, fast_len, dtype=torch.bool, device=device)
        mask = make_three_segment_mask(p_mask, f_mask, None, prefix_len, fast_len, 0)

        empty_suffix = torch.zeros(B, 0, self.config.action_width, device=device)

        prefix_kv_cache = []
        x = combined
        for block in self.blocks:
            x, _, new_kv = block(x, empty_suffix, None, mask, None)
            k, v = new_kv
            # 只缓存 prefix 块 KV（FAST 不缓存，连续动作不允许看 FAST）
            prefix_kv_cache.append((k[:, :prefix_len], v[:, :prefix_len]))

        x = self.final_norm_prefix(x)                 # (B, P+F, width)
        fast_hidden = x[:, prefix_len:]               # (B, F, width)
        return fast_hidden, prefix_kv_cache
