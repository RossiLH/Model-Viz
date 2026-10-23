# -*- coding: utf-8 -*-
"""PI0.7 专属变换（视频状态归一化 / 元数据标准化 / 两级 prompt dropout）。

与 core/train/transforms.py 的分工：
- core/：跨 VLA 模型共享（Normalize / Delta / Pad / Resize）
- pi07/train/：PI0.7 专属（视频状态归一化、元数据、dropout）

所有 transform 都是可调用对象：接收 dict，返回变换后的 dict。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from models.vla.core.train.norm_stats import NormStats
from models.vla.pi07.modules.metadata import sample_prompt_dropout


@dataclass(frozen=True)
class NormalizeVideoState:
    """对 state_history (T, D) 做分位数 / z-score 归一化。

    把 (T, D) 展平成 (T, D) 后逐行复用同一 NormStats（与 state 归一化
    用同一套统计量），然后还原形状。视频帧是 uint8 跳过（与 core 的
    Normalize 跳 uint8 同理）。

    为什么不复用 core.Normalize：core.Normalize 假设 state 是 (D,) 单向量，
    直接调用会把整段 (T, D) 当一行向量处理，q01 形状不匹配。
    """

    norm_stats: Dict[str, NormStats]
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        if "state_history" not in data:
            return data
        sh = data["state_history"]  # (T, D)
        T, D = sh.shape
        flat = sh.reshape(-1, D)  # (T, D)

        key = "state" if "state" in self.norm_stats else None
        if key is None or "state" not in self.norm_stats:
            return data

        stats = self.norm_stats["state"]
        if self.use_quantiles:
            if stats.q01 is None or stats.q99 is None:
                return data
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            normalized = (flat - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            if stats.mean is None or stats.std is None:
                return data
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            normalized = (flat - mean) / (std + 1e-6)

        data["state_history"] = normalized.reshape(T, D).astype(sh.dtype)
        return data


@dataclass(frozen=True)
class NormalizeMetadata:
    """元数据四字段标准化到 ~[0, 1] 量级。

    为什么必须标准化：metadata_proj 是 nn.Linear(1, width)，
    speed 原始值（几千步）与 mistake 的 0/1 差 3-4 个数量级，
    不缩放会让 speed 那一维的梯度完全主导元数据通路。

    字段顺序：[speed, quality, mistake, control_mode]
    - speed:   / speed_scale（默认 8000，论文 prompt 示例值）
    - quality: (q - quality_min) / (quality_max - quality_min)
    - mistake: 0/1 原样
    - control_mode: 0/1 原样
    """

    n_metadata: int = 4
    speed_scale: float = 8000.0
    quality_min: int = 1
    quality_max: int = 5

    def __call__(self, data: dict) -> dict:
        if "metadata" not in data:
            return data
        meta = data["metadata"].copy().astype(np.float32)
        if self.n_metadata >= 1:
            meta[0] = meta[0] / self.speed_scale
        if self.n_metadata >= 2:
            q_range = max(1, self.quality_max - self.quality_min)
            meta[1] = (meta[1] - self.quality_min) / q_range
        data["metadata"] = meta
        return data


def apply_prompt_dropout(
    batch: Dict[str, torch.Tensor],
    *,
    metadata_drop_prob: float = 0.15,
    per_field_drop_prob: float = 0.05,
    subgoal_keep_prob: float = 0.25,
    subtask_drop_given_subgoal: float = 0.30,
    n_metadata_fields: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """在 batch 维度上采样两级 dropout 掩码（论文 V-E）。

    落点：collate 之后、model.forward 之前。
    为什么不放在逐样本 transform 里：sample_prompt_dropout 返回 (B,) 掩码，
    需要 batch 维度；逐样本 generator 状态不同步会导致统计特性偏差。

    返回新的 batch 字典（原 batch 不变），仅修改：
    - metadata_masks: 按 dropout 概率置零（整块 + 逐字段）
    - subgoal_mask:    与原 subgoal_mask 做 AND（数据里有 + dropout 保留）
    - response_masks:  与 subtask_drop 做 AND（有子目标时才丢子任务）
    """
    bsize = batch["videos"].shape[0]
    keep = sample_prompt_dropout(
        bsize,
        metadata_drop_prob=metadata_drop_prob,
        per_field_drop_prob=per_field_drop_prob,
        n_metadata_fields=n_metadata_fields,
        subgoal_keep_prob=subgoal_keep_prob,
        subtask_drop_given_subgoal=subtask_drop_given_subgoal,
        generator=generator,
    )

    out = dict(batch)

    # 元数据：原 mask AND dropout 保留掩码
    if "metadata_masks" in out and out["metadata_masks"] is not None:
        meta_keep = keep["metadata_fields"].to(
            device=out["metadata_masks"].device
        )
        out["metadata_masks"] = out["metadata_masks"] & meta_keep

    # 子目标：数据里有 AND dropout 保留
    if "subgoal_mask" in out and out["subgoal_mask"] is not None:
        sg_keep = keep["subgoal"].to(device=out["subgoal_mask"].device)
        out["subgoal_mask"] = out["subgoal_mask"] & sg_keep

    # 子任务：response_masks AND subtask 保留
    if "response_masks" in out and out["response_masks"] is not None:
        sub_keep = keep["subtask"].to(device=out["response_masks"].device)
        out["response_masks"] = out["response_masks"] & sub_keep[:, None]

    return out


def make_inference_metadata(
    bsize: int,
    *,
    n_metadata: int = 4,
    quality: int = 5,
    mistake: bool = False,
    control_mode: float = 0.0,
    speed_value: float = 0.15,
    speed_scale: float = 8000.0,
    quality_min: int = 1,
    quality_max: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """构造推理/验证时的固定档位元数据（论文 VII）。

    speed:   取任务 episode 长度的第 15 百分位（默认 speed_value=0.15，
             这里直接传已归一化的 0-1 值，调用方负责算百分位）
    quality: 恒取 5（最高）
    mistake: 恒取 False
    """
    meta = torch.zeros(bsize, n_metadata, dtype=torch.float32, device=device)
    if n_metadata >= 1:
        meta[:, 0] = speed_value
    if n_metadata >= 2:
        q_norm = (quality - quality_min) / max(1, quality_max - quality_min)
        meta[:, 1] = q_norm
    if n_metadata >= 3:
        meta[:, 2] = float(mistake)
    if n_metadata >= 4:
        meta[:, 3] = control_mode
    meta_masks = torch.ones(bsize, n_metadata, dtype=torch.bool, device=device)
    return {"metadata": meta, "metadata_masks": meta_masks}
