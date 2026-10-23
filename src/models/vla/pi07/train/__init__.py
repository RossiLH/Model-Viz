# -*- coding: utf-8 -*-
"""PI0.7 训练工具包（模型专属，仅 PI0.7 用）。

与 core/train/ 的分工：
- core/train/：跨 VLA 模型共享（RunningStats、Normalize、lr_schedule、PI0.5 trainer）
- pi07/train/：PI0.7 专属（视频窗口数据转换、元数据标准化、两级 dropout、Pi07Trainer）
"""

from models.vla.pi07.train.config import Pi07TrainConfig
from models.vla.pi07.train.trainer import Pi07Trainer

__all__ = ["Pi07TrainConfig", "Pi07Trainer"]
