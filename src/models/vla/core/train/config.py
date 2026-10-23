# -*- coding: utf-8 -*-
"""PI0.5 训练配置（TrainConfig）。

与官方 `openpi/training/config.py:TrainConfig` 对齐：
- 单一 dataclass，PyDantic-free（不引入新依赖）
- 字段全部是基本类型，YAML 友好

只列本项目真实使用的字段（不抄全表），并对每个字段加详细中文注释。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class TrainConfig:
    """PI0.5 训练的所有超参与开关。

    设计原则：
        - 字段命名与官方一致，便于将来对照
        - 任何"是否启用某特性"都用 None / 0 区分"没开"和"开了用 0"
        - 不引入 pydantic / hydra 等额外依赖（PyYAML 都不要，足够简单）

    必填字段：model (Pi05FaithfulPolicyModel 实例)、norm_stats (dict[str, NormStats])
    """

    # ---------------- 数据 ----------------
    # LeRobot 数据集路径（也可以是 npz 自定义格式，由 DataLoader 决定）
    dataset_path: str = ""

    # 训练时动作 chunk 长度（未来 H 步）
    action_horizon: int = 16

    # state / actions 零填充的目标维度（必须等于 model.action_dim）
    # 如果 model.action_dim=7，pad 后的 state/actions 最后一维就是 7
    model_action_dim: int = 7

    # DeltaActions 用的 mask（前 True 维做 delta，最后 False 维保持绝对）
    # 官方 Libero 是 [True]*6 + [False]（6 关节 + 1 夹爪绝对）
    delta_mask: Optional[Sequence[bool]] = None

    # 训练 batch size
    batch_size: int = 32

    # 训练/验证划分比例（基于 episode 数）
    val_split: float = 0.05

    # ---------------- 优化器 ----------------
    # AdamW 配置（对齐 `openpi/training/optimizer.py:66-83`）
    opt_b1: float = 0.9
    opt_b2: float = 0.95
    opt_eps: float = 1e-8
    opt_weight_decay: float = 1e-10
    clip_grad_norm: float = 1.0

    # LR 调度（warmup_cosine）
    warmup_steps: int = 1000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    # EMA decay（0 表示关闭）
    ema_decay: float = 0.99

    # ---------------- 训练循环 ----------------
    max_steps: int = 100_000
    log_interval: int = 100
    save_interval: int = 1000
    seed: int = 42

    # 设备
    device: str = "cpu"

    # ---------------- checkpoint ----------------
    output_dir: str = "outputs/pi05_train"
    resume_from: Optional[str] = None  # 之前的 checkpoint 路径

    # ---------------- norm stats ----------------
    # 已算好的 NormStats 路径；为空时假定 model 已自带 stats
    norm_stats_path: Optional[str] = None

    # ---------------- 调试开关 ----------------
    # 启用后会多打几行用于排错（默认 False 保持训练循环轻量）
    debug: bool = False


@dataclass
class AdamWConfig:
    """AdamW 优化器配置（与 `openpi/training/optimizer.py:AdamW` 对齐）。

    单独抽出来是为了让 TrainConfig 不臃肿。
    """
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10


def build_optimizer_params(model, config: TrainConfig) -> list:
    """构造 torch.optim.AdamW 的 params 列表。

    与官方 `AdamW.create()` 一致：不区分 weight_decay mask（官方 PI0/PI0.5
    训练时对所有参数统一 weight_decay = 1e-10，所以这里不分组）。
    """
    return list(model.parameters())


__all__ = ["TrainConfig", "AdamWConfig", "build_optimizer_params"]
