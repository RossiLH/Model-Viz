# -*- coding: utf-8 -*-
"""VisionProjector：把 pixel-shuffle 后的视觉特征映射回骨干 token 嵌入空间。

对应 K3 技术报告 §2.4。设计上是**刻意小**的适配器：
论文把容量放在视觉塔和骨干里，而不是放在两者之间的适配器上 —— 因为骨干从第一步
就与视觉 token 联合训练，不需要一个"翻译层"帮它理解视觉。

本实现与 ``Reference/kimi3-master/kimi3/vision/projector.py`` 逐位对齐：
``norm -> fc1 -> gelu -> fc2``，输入宽 ``dim * pixel_shuffle^2``，输出宽 ``model_dim``。
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ViTConfig
from ..modules.norm import RMSNorm

__all__ = ["VisionProjector"]


class VisionProjector(nn.Module):
    """轻量 MLP 投影器：``(B, T, dim*shuffle^2) -> (B, T, model_dim)``。

    结构刻意简单：
      x -> RMSNorm(in) -> Linear(in, hidden, bias=True) -> GELU
        -> Linear(hidden, model_dim, bias=True)
    两个 Linear 都带 bias（这里是投影器，不是 MoonViT 塔本体，不适用"无 bias"约定）。
    """

    def __init__(self, vit_cfg: ViTConfig, model_dim: int, eps: float = 1e-6):
        super().__init__()
        in_dim = vit_cfg.dim * vit_cfg.pixel_shuffle**2   # 视觉塔输出宽
        hidden = vit_cfg.projector_hidden or model_dim     # 隐藏宽，默认等于骨干宽
        self.norm = RMSNorm(in_dim, eps)
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, model_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, in_dim) -> (B, T, model_dim)
        return self.fc2(F.gelu(self.fc1(self.norm(x))))