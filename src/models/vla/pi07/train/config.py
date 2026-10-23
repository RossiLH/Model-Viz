# -*- coding: utf-8 -*-
"""PI0.7 训练配置（Pi07TrainConfig）。

继承 core/train/config.py 的 TrainConfig，在此基础上加 PI0.7 专属字段。
未改动的超参（优化器、LR、EMA 等）全部从父类继承，保持与 PI0.5 同默认值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from models.vla.core.train.config import TrainConfig


@dataclass
class Pi07TrainConfig(TrainConfig):
    """PI0.7 训练的全部超参与开关。

    与 PI0.5 的关键差异（仅列出新增 / 覆写字段）：

    视频历史：
        - n_obs_steps: 视频历史帧数（论文 T=6）
        - state_dim: 状态维度，每帧一个 token（论文 state_dim=32）
        - image_size: 图像边长（用于视频 resize）

    元数据条件化（对齐 Pi07MetadataConfig）：
        - metadata_enabled: 是否开启元数据条件化
        - n_metadata: 元数据标量字段数（speed, quality, mistake, control_mode）
        - metadata_drop_prob: 整块丢弃概率（论文 15%）
        - per_field_drop_prob: 逐字段丢弃概率（论文 5%）
        - speed_bin_size: speed 分箱宽度（论文 500 步）
        - quality_min / quality_max: quality 取值范围 [1, 5]

    子目标图像（对齐 Pi07SubgoalConfig）：
        - subgoal_enabled: 是否开启子目标图像条件
        - subgoal_keep_prob: 注入比例（论文 25%）
        - subtask_drop_given_subgoal: 有子目标时子任务文本丢弃率（论文 30%）

    动作块（覆写父类默认值）：
        - action_horizon: 50（论文 H=50，PI0.5 默认 16）
        - model_action_dim: 32（PI0.5 默认 7）
    """

    # ---------------- 视频历史 ----------------
    # 视频历史帧数（论文 T=6）
    n_obs_steps: int = 6
    # 状态维度（每帧投影一个 token，论文 state_dim=32）
    state_dim: int = 32
    # 图像边长（用于视频 resize；与视觉编码器 image_size 一致）
    image_size: int = 224

    # ---------------- 元数据 ----------------
    # 是否启用元数据条件化
    metadata_enabled: bool = True
    # 元数据标量字段数（speed, quality, mistake, control_mode = 4）
    n_metadata: int = 4
    # 元数据整块丢弃概率（论文 §V-E：15%）
    metadata_drop_prob: float = 0.15
    # 单字段独立丢弃概率（论文：5%）
    per_field_drop_prob: float = 0.05
    # speed 分箱宽度（步）
    speed_bin_size: int = 500
    # quality 取值范围
    quality_min: int = 1
    quality_max: int = 5
    # 元数据归一化尺度：speed 除以该值缩到 ~1 量级
    speed_scale: float = 8000.0

    # ---------------- 子目标图像 ----------------
    # 是否启用子目标图像条件
    subgoal_enabled: bool = True
    # 子目标注入比例（论文 25%）
    subgoal_keep_prob: float = 0.25
    # 有子目标时子任务文本丢弃率（论文 30%）
    subtask_drop_given_subgoal: float = 0.30

    # ---------------- 覆写父类默认值 ----------------
    # 动作块长度（论文 H=50）
    action_horizon: int = 50
    # 模型动作维度
    model_action_dim: int = 32

    # ---------------- 输出目录 ----------------
    output_dir: str = "outputs/pi07_train"

    # ---------------- 验证时元数据档位（论文 §VII） ----------------
    # quality 恒取 5（最高）
    inference_quality: int = 5
    # mistake 恒取 False
    inference_mistake: bool = False
    # speed 取该任务 episode 长度的第 15 百分位
    inference_speed_percentile: float = 15.0


__all__ = ["Pi07TrainConfig"]
