"""双专家联合注意力模块

本模块实现了 PI0.5 的核心创新 —— 双专家联合注意力，对应官方 openpi-main 中的 Attention 类。

官方实现参考：
- 文件: Reference/openpi-main/src/openpi/models/gemma.py
- 类: Attention
- 调用: Attention.__call__()
- 掩码构造: Reference/openpi-main/src/openpi/models/pi0.py 的 make_attn_mask

官方核心逻辑：
1. 两个 expert（PaliGemma 语言模型 + Action Expert）分别投影出各自的 Q/K/V
2. 将两个 expert 的 Q/K/V 拼接成一份，做一次联合注意力计算
3. 按 expert 分割回各自维度，分别通过各自的输出投影
4. 支持 KV cache（推理时 prefix 部分只计算一次并缓存）

关键语义（论文 arXiv:2504.16054 附录 A-E 强制）：
    "A full prefix mask is used on images, prompt tokens, and proprioceptive
     state; FAST action tokens attend to this prefix and auto-regressively on
     previous action tokens. ... In effect, information flows unidirectionally
     from the VLM to the action expert; no VLM embedding attends to the action
     expert."

即：attention 掩码必须是非对称的块掩码 ——
    - prefix 内部：双向可见（全 prefix mask）
    - suffix 看 prefix：可见
    - suffix 看 suffix：可见（动作块整体预测）
    - prefix 看 suffix：**禁止**（VLM 不 attend action expert）
掩码必须作用在 **key 轴**（被 mask 的 token 不应被其他 token 看见）。
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_attn_mask(
    prefix_mask: torch.Tensor,
    suffix_mask: Optional[torch.Tensor],
    prefix_len: int,
    suffix_len: int,
) -> torch.Tensor:
    """构造双专家块掩码（对应官方 pi0.py:19 make_attn_mask 的块下三角语义）

    返回形状 (B, total_len, total_len) 的布尔矩阵，True=可见。
    块结构（prefix 在上，suffix 在下）：
        key:     prefix    suffix
    query
    prefix  -> [  True      False ]   # prefix 只看 prefix，不看 suffix（论文强制）
    suffix  -> [  True      True  ]   # suffix 看 prefix + 看自己

    Args:
        prefix_mask: (B, prefix_len) prefix 的 padding 掩码
        suffix_mask: (B, suffix_len) 或 None；None 时 suffix 全部可见
        prefix_len: prefix 长度
        suffix_len: suffix 长度

    Returns:
        (B, total_len, total_len) 布尔掩码
    """
    B = prefix_mask.shape[0]
    device = prefix_mask.device

    # 约定：掩码形状 (B, query, key) —— dim1 是 query 轴，dim2 是 key 轴
    #   prefix_mask.unsqueeze(2) -> (B, pl, 1) 作为 query 侧
    #   prefix_mask.unsqueeze(1) -> (B, 1, pl) 作为 key 侧
    if suffix_mask is None:
        suffix_mask = torch.ones(B, suffix_len, dtype=torch.bool, device=device)

    # 左上：prefix(query) -> prefix(key)，全双向可见，仅受 padding 约束
    block_pp = prefix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)  # (B, pl, pl)
    # 右上：prefix(query) -> suffix(key)，论文禁止，全 False
    block_ps = torch.zeros(B, prefix_len, suffix_len, dtype=torch.bool, device=device)
    # 左下：suffix(query) -> prefix(key)，可见
    block_sp = suffix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)  # (B, sl, pl)
    # 右下：suffix(query) -> suffix(key)，双向可见，仅受 padding 约束
    block_ss = suffix_mask.unsqueeze(2) & suffix_mask.unsqueeze(1)  # (B, sl, sl)

    top = torch.cat([block_pp, block_ps], dim=2)  # (B, pl, total)
    bottom = torch.cat([block_sp, block_ss], dim=2)  # (B, sl, total)
    return torch.cat([top, bottom], dim=1)  # (B, total, total)


class DualExpertJointAttention(nn.Module):
    """双专家联合注意力

    对应官方: Attention (Reference/openpi-main/src/openpi/models/gemma.py)

    PaliGemma（语言模型）和 Action Expert 共享注意力计算，
    而非各自独立计算注意力，这是 PI0.5 的核心创新之一。
    """

    def __init__(self, config):
        """初始化联合注意力

        对应官方: Attention.__init__()

        Args:
            config: Pi05FaithfulConfig，包含:
                - paligemma_width: PaliGemma 隐藏维度
                - paligemma_num_heads: PaliGemma 注意力头数
                - paligemma_num_kv_heads: PaliGemma KV 头数（GQA）
                - action_width: Action Expert 隐藏维度
                - action_num_heads: Action Expert 注意力头数
                - action_num_kv_heads: Action Expert KV 头数（GQA）
                - head_dim: 每个注意力头的维度
        """
        super().__init__()

        self.config = config
        self.head_dim = config.head_dim

        # 两个 expert 使用相同的 head 数（官方要求，便于联合注意力计算）
        self.num_heads = config.paligemma_num_heads
        self.num_kv_heads = config.paligemma_num_kv_heads

        # GQA 组数: 每个 KV 头对应的 Q 头数量
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # ===== PaliGemma (Prefix) 投影 =====
        # 对应官方: 语言模型 expert 的 q/k/v 投影
        # 将 prefix 序列 (paligemma_width) 映射到注意力空间 (num_heads * head_dim)
        self.prefix_q_proj = nn.Linear(
            config.paligemma_width,
            config.paligemma_num_heads * config.head_dim,
            bias=False,
        )
        # KV 使用更少的头（GQA），节省内存
        self.prefix_k_proj = nn.Linear(
            config.paligemma_width,
            config.paligemma_num_kv_heads * config.head_dim,
            bias=False,
        )
        self.prefix_v_proj = nn.Linear(
            config.paligemma_width,
            config.paligemma_num_kv_heads * config.head_dim,
            bias=False,
        )

        # ===== Action Expert (Suffix) 投影 =====
        # 对应官方: 动作 expert 的 q/k/v 投影
        self.suffix_q_proj = nn.Linear(
            config.action_width,
            config.action_num_heads * config.head_dim,
            bias=False,
        )
        self.suffix_k_proj = nn.Linear(
            config.action_width,
            config.action_num_kv_heads * config.head_dim,
            bias=False,
        )
        self.suffix_v_proj = nn.Linear(
            config.action_width,
            config.action_num_kv_heads * config.head_dim,
            bias=False,
        )

        # ===== 输出投影 =====
        # 对应官方: 各 expert 的 o_proj，将注意力输出映射回各自的隐藏维度
        self.prefix_out_proj = nn.Linear(
            config.paligemma_num_heads * config.head_dim,
            config.paligemma_width,
            bias=False,
        )
        self.suffix_out_proj = nn.Linear(
            config.action_num_heads * config.head_dim,
            config.action_width,
            bias=False,
        )

        # 预计算 RoPE 频率
        self._init_rope_frequencies()

    def _init_rope_frequencies(self):
        """初始化 RoPE 频率

        对应官方: gemma.py 中的 RoPE 频率初始化

        RoPE 使用正弦函数生成随位置变化的旋转角度，
        频率按 head_dim 的一半指数递减（类似位置编码）。
        """
        dim = self.head_dim // 2  # 每对维度一个频率
        # 频率: 1 / (10000^(2i/d))，与 Transformer 位置编码一致
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, dtype=torch.float) / dim))
        self.register_buffer("rope_freqs", freqs, persistent=False)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """应用旋转位置编码 (RoPE)

        对应官方: _apply_rope (Reference/openpi-main/src/openpi/models/gemma.py)

        RoPE 将每个维度对 (x_real, x_imag) 按位置角度旋转:
            out_real = x_real * cos(theta) - x_imag * sin(theta)
            out_imag = x_real * sin(theta) + x_imag * cos(theta)
        其中 theta = position * frequency

        Args:
            x: (B, seq_len, num_heads, head_dim) 待编码的张量
            positions: (B, seq_len) 位置索引

        Returns:
            (B, seq_len, num_heads, head_dim) 编码后的张量
        """
        B, S, H, D = x.shape

        # 计算每个位置的角度: (B, S, D//2)
        angles = positions.unsqueeze(-1) * self.rope_freqs.unsqueeze(0)
        sin = angles.sin()
        cos = angles.cos()

        # 将最后一维拆成 (D//2, 2) 分别代表实部和虚部
        x_reshaped = x.view(B, S, H, D // 2, 2)
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        # 旋转公式
        # cos/sin 形状 (B, S, D//2)，unsqueeze(2) 变为 (B, S, 1, D//2)
        # 与 x_real (B, S, H, D//2) 广播相乘（对每个位置和 head 应用同一旋转角）
        out_real = x_real * cos.unsqueeze(2) - x_imag * sin.unsqueeze(2)
        out_imag = x_real * sin.unsqueeze(2) + x_imag * cos.unsqueeze(2)

        # 重新拼回 (B, S, H, D)
        out = torch.stack([out_real, out_imag], dim=-1).view(B, S, H, D)
        return out

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """扩展 KV 以匹配 Q 的 head 数（GQA）

        对应官方: gemma.py 中 kv 头数少于 q 头数时的扩展逻辑

        GQA（分组查询注意力）中，多个 Q 头共享一个 KV 头。
        本函数将 KV 从头维度 (num_kv_heads) 复制扩展到 (num_heads)。

        Args:
            x: (B, S, num_kv_heads, head_dim)

        Returns:
            (B, S, num_heads, head_dim)
        """
        if self.num_kv_groups == 1:
            return x  # 一对一，无需扩展

        B, S, H_kv, D = x.shape
        # 将每个 KV 头复制 num_kv_groups 次
        x = x.unsqueeze(2).expand(B, S, self.num_kv_groups, H_kv, D)
        x = x.reshape(B, S, H_kv * self.num_kv_groups, D)
        return x

    def forward(
        self,
        prefix: Optional[torch.Tensor],
        suffix: torch.Tensor,
        mask: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """前向传播

        对应官方: Attention.__call__()

        数据流:
            1. prefix (PaliGemma) 投影 Q/K/V
            2. suffix (Action Expert) 投影 Q/K/V
            3. 拼接 Q/K/V 做联合注意力
            4. 应用 RoPE（位置编码）
            5. 合并 KV cache（推理时缓存 prefix 的 K/V）
            6. GQA 扩展 KV 头
            7. Scaled Dot-Product Attention
            8. 按 expert 分割回各自维度
            9. 输出投影 + 更新 KV cache

        Args:
            prefix: (B, prefix_len, paligemma_width) PaliGemma token，可为 None（使用缓存）
            suffix: (B, suffix_len, action_width) Action Expert token
            mask: (B, q_len, kv_len) **二维块掩码**（Bool，True=可见）。
                由 make_attn_mask 生成，蕴含"prefix 不看 suffix"的单向约束。
                kv_len 需等于最终 key 长度（含 cache 时 = cached_len + q_len）。
            kv_cache: 可选的 (cached_k, cached_v)，形状 (B, cached_len, num_kv_heads, head_dim)

        Returns:
            prefix_out: (B, prefix_len, paligemma_width) 或 None（当 prefix 为 None）
            suffix_out: (B, suffix_len, action_width)
            new_kv_cache: (new_k, new_v) 本次新增的 KV（用于下次缓存）
        """
        B = suffix.shape[0]
        device = suffix.device

        # ===== 处理 Prefix (PaliGemma) =====
        if prefix is not None:
            prefix_len = prefix.shape[1]
            # 投影 Q/K/V 并重塑为多头形式
            q_prefix = self.prefix_q_proj(prefix)  # (B, prefix_len, num_heads * head_dim)
            k_prefix = self.prefix_k_proj(prefix)  # (B, prefix_len, num_kv_heads * head_dim)
            v_prefix = self.prefix_v_proj(prefix)  # (B, prefix_len, num_kv_heads * head_dim)
            q_prefix = q_prefix.view(B, prefix_len, self.num_heads, self.head_dim)
            k_prefix = k_prefix.view(B, prefix_len, self.num_kv_heads, self.head_dim)
            v_prefix = v_prefix.view(B, prefix_len, self.num_kv_heads, self.head_dim)
        else:
            # prefix 为 None（推理增量），仅处理 suffix
            prefix_len = 0

        # ===== 处理 Suffix (Action Expert) =====
        suffix_len = suffix.shape[1]
        q_suffix = self.suffix_q_proj(suffix)  # (B, suffix_len, num_heads * head_dim)
        k_suffix = self.suffix_k_proj(suffix)  # (B, suffix_len, num_kv_heads * head_dim)
        v_suffix = self.suffix_v_proj(suffix)  # (B, suffix_len, num_kv_heads * head_dim)
        q_suffix = q_suffix.view(B, suffix_len, self.num_heads, self.head_dim)
        k_suffix = k_suffix.view(B, suffix_len, self.num_kv_heads, self.head_dim)
        v_suffix = v_suffix.view(B, suffix_len, self.num_kv_heads, self.head_dim)

        # ===== 拼接做联合注意力 =====
        # 对应官方: 将两个 expert 的 q/k/v 拼接
        if prefix is not None:
            q = torch.cat([q_prefix, q_suffix], dim=1)  # (B, total_len, num_heads, head_dim)
            k = torch.cat([k_prefix, k_suffix], dim=1)  # (B, total_len, num_kv_heads, head_dim)
            v = torch.cat([v_prefix, v_suffix], dim=1)  # (B, total_len, num_kv_heads, head_dim)
            total_len = prefix_len + suffix_len
        else:
            q = q_suffix
            k = k_suffix
            v = v_suffix
            total_len = suffix_len

        # ===== 应用 RoPE =====
        # 对应官方: q/k 应用旋转位置编码
        # 新 token 的位置：若已有 cache，则从 cached_len 开始（增量解码位置续接）
        if kv_cache is not None:
            start_pos = kv_cache[0].shape[1]
        else:
            start_pos = 0
        positions = (
            torch.arange(start_pos, start_pos + total_len, device=device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        q = self._apply_rope(q, positions)
        k = self._apply_rope(k, positions)

        # ===== KV Cache 合并 =====
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            cached_len = cached_k.shape[1]
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)
            kv_len = cached_len + total_len
        else:
            cached_len = 0
            kv_len = total_len

        # ===== GQA 扩展 KV =====
        # 将 KV 从 num_kv_heads 扩展到 num_heads
        k_expanded = self._expand_kv(k)  # (B, kv_len, num_heads, head_dim)
        v_expanded = self._expand_kv(v)  # (B, kv_len, num_heads, head_dim)

        # ===== 注意力计算 =====
        # 对应官方: scaled dot-product attention
        q_t = q.transpose(1, 2)  # (B, num_heads, total_len, head_dim)
        k_t = k_expanded.transpose(1, 2)  # (B, num_heads, kv_len, head_dim)
        v_t = v_expanded.transpose(1, 2)  # (B, num_heads, kv_len, head_dim)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) / scale  # (B, H, total_len, kv_len)

        # 应用注意力掩码 —— 作用在 KEY 轴（被 mask 的 key 不被任何 query 看见）
        # softmax 在 key 维（dim=-1），故掩码需覆盖全部 key（含缓存 keys）
        if mask is not None:
            if mask.dim() == 2:
                # 向后兼容：1D 行掩码 (B, total_len)
                # 扩展为 key 轴二维：缓存 keys 全可见 + 新 keys 按行掩码
                row_mask = mask.unsqueeze(1)  # (B, 1, total_len)
                mask = torch.cat(
                    [torch.ones(B, total_len, cached_len, dtype=torch.bool, device=device), row_mask],
                    dim=2,  # (B, total_len, kv_len)
                )
            elif mask.dim() == 3 and kv_cache is not None:
                # 3D 掩码 (B, total_len, suffix_len)：只覆盖当前新增 keys，
                # 需在 key 维前拼接缓存 keys（全可见）
                mask = torch.cat(
                    [torch.ones(B, total_len, cached_len, dtype=torch.bool, device=device), mask],
                    dim=2,  # (B, total_len, cached_len+suffix_len)= (B, total_len, kv_len)
                )
            mask_expanded = mask.unsqueeze(1)  # (B, 1, total_len, kv_len)
            attn_weights = attn_weights.masked_fill(~mask_expanded, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v_t)  # (B, num_heads, total_len, head_dim)

        # 重塑: (B, total_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, total_len, self.num_heads * self.head_dim)

        # ===== 分割回两个 expert =====
        if prefix is not None:
            prefix_attn = attn_output[:, :prefix_len]
            suffix_attn = attn_output[:, prefix_len:]
            prefix_out = self.prefix_out_proj(prefix_attn)
            suffix_out = self.suffix_out_proj(suffix_attn)
        else:
            prefix_out = None
            suffix_out = self.suffix_out_proj(attn_output)

        # ===== 更新 KV Cache =====
        # 缓存本次新增的 KV（合并前的原始 k/v，num_kv_heads 形式）
        new_kv_cache = (k, v)

        return prefix_out, suffix_out, new_kv_cache
