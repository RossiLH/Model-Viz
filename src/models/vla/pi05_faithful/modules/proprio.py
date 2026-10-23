"""Proprio（关节状态）接入模块

设计规格 §3：方案 (a) 状态离散化接入 prefix。
OpenTau / CBC 的做法是把状态 discretize 成 0-255 的 bins，转字符串拼进 prompt：
"Task: ..., State: s1 s2 ...;\nAction: "。

本实现（自包含，无语言 tokenizer）：
- 复用 `StateTokenizer.bins` 做相同的 digitize；
- 但直接返回 bin 索引作为整数 token id（每状态维度一个 token，范围 [0, num_bins)），
  经模型 `text_embed` 嵌入后拼到 prefix **末尾**（即图像 token 与文本 token 之后，
  最终顺序：图像 → 文本 → state）。
- 语义与 OpenTau/CBC 的「离散状态文本」等价，省去字符串↔id 往返。

默认 `proprio_enabled=False`，且 `encode_prefix` 仅当 `state` 非 None 时拼接
（向后兼容：不传 state 或开关关闭时行为与现状完全一致）。
"""
import numpy as np
import torch

from .state_tokenizer import StateTokenizer


class ProprioEncoder:
    """状态离散化器：连续状态 → bin 索引 token id"""

    def __init__(self, num_bins=256, state_range=(-1.0, 1.0)):
        """初始化

        Args:
            num_bins: 离散化 bins 数量（默认 256，对应 0-255）
            state_range: 状态值范围（默认 -1 到 1）
        """
        self.num_bins = num_bins
        self.tokenizer = StateTokenizer(num_bins=num_bins, state_range=state_range)

    def encode(self, state):
        """将连续状态离散化为 token id

        Args:
            state: (B, D) 或 (D,) 状态向量

        Returns:
            (B, D) long token id，每个值落在 [0, num_bins)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_np = state.detach().cpu().numpy()
        bins = self.tokenizer.bins
        # digitize 返回 bins[i-1] <= x < bins[i] 的 i；bins[0] == state_range[0]，
        # 故左端点恰好落在 bin 1，减 1 使左端点 → 0、右端点 → num_bins-1
        digitized = np.digitize(state_np, bins) - 1
        digitized = np.clip(digitized, 0, self.num_bins - 1)
        return torch.tensor(digitized, dtype=torch.long, device=state.device)
