# -*- coding: utf-8 -*-
"""学习率调度器（PyTorch 复刻 optax.warmup_cosine_decay_schedule）。

为什么用 PyTorch 自带 LambdaLR 而不是 optax
-----------------------------------------
本项目跑 CPU + PyTorch，没有 optax（JAX 系）。LambdaLR 是 PyTorch 最轻量的
"传入 step 返回乘子"接口，正好对应 optax 的 schedule 语义。映射关系：

    optax.warmup_cosine_decay_schedule(init, peak, warmup, decay, end)
    → LR = init + (peak - init) * (step / warmup)                          step <= warmup
        端 = end                                                          step >= warmup + decay
        中段 = end + (peak - end) * 0.5 * (1 + cos(pi * t / decay))        warmup < step < warmup + decay

init 是 peak / (warmup_steps + 1) —— 这是 optax 的小坑：它假设第一个 step
是 step 0，所以想从 ~0 起步就只能取 peak / (warmup+1) 当作 init。我们沿用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR


@dataclass
class CosineDecayWithWarmupConfig:
    """与官方 `CosineDecaySchedule` 配置对齐。

    训练 PI0.5 时的官方默认值（来自 `openpi/training/optimizer.py:18-22`）：
        warmup_steps = 1000
        peak_lr      = 2.5e-5
        decay_steps  = 30_000
        decay_lr     = 2.5e-6

    含义：
        前 1000 步线性从 0 → 2.5e-5
        接下来 30000 步余弦下降到 2.5e-6
        之后保持 2.5e-6
    """

    warmup_steps: int = 1000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6


def warmup_cosine_decay_schedule(
    optimizer,
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float,
    *,
    last_epoch: int = -1,
) -> LambdaLR:
    """构造一个 PyTorch LambdaLR，行为对齐 optax.warmup_cosine_decay_schedule。

    Args:
        optimizer: 目标 PyTorch optimizer。LambdaLR 会把 `lr * lr_lambda(step)`
                   写进 param_groups。这里要求 `optimizer.defaults['lr']` 等于
                   peak_value，以便"lr_lambda(step) 返回的乘子"有直观意义：
                       实际 LR = peak_value * lr_lambda(step)
        init_value:  warmup 起点（一般是 peak/(warmup_steps+1)）
        peak_value:  warmup 终点 / decay 起点
        warmup_steps: 线性 warmup 步数
        decay_steps:  从 peak 到 end 的余弦衰减步数
        end_value:    衰减终点，之后保持
        last_epoch:   初始 epoch，-1 表示从 0 开始

    Returns:
        LambdaLR 调度器，调用 scheduler.step() 时按 step 推进
    """
    if warmup_steps < 1:
        raise ValueError(f"warmup_steps 必须 >= 1，得到 {warmup_steps}")
    if decay_steps < 1:
        raise ValueError(f"decay_steps 必须 >= 1，得到 {decay_steps}")

    def lr_lambda(step: int) -> float:
        """LambdaLR 把返回值当乘子，与 optimizer.defaults['lr'] 相乘。

        所以我们返回的是"该 step 应有的 LR" / "optimizer 的初始 lr"，
        而不是直接返回 LR 绝对值。"""
        # 1) warmup 阶段：线性从 init 到 peak
        if step < warmup_steps:
            target = init_value + (peak_value - init_value) * (step / warmup_steps)
        # 2) 衰减结束：保持 end_value
        elif step >= warmup_steps + decay_steps:
            target = end_value
        # 3) 中段：余弦下降到 end_value
        else:
            t = (step - warmup_steps) / decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            target = end_value + (peak_value - end_value) * cosine
        # 归一化：乘子 = 目标 / base
        return target / peak_value

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def attach_lr_scheduler(
    optimizer,
    config: CosineDecayWithWarmupConfig,
    *,
    last_epoch: int = -1,
) -> LambdaLR:
    """便捷函数：把 config 直接装到 PyTorch optimizer 上。

    与官方对齐：
        init_value = peak / (warmup_steps + 1)

    注意：此函数会按 config.peak_lr 覆盖 optimizer 的 base lr。

    PyTorch 的 LambdaLR 实际从 `group['initial_lr']`（若不存在则取
    `group['lr']` 写入）读 base，因此必须三处都改：
        - optimizer.defaults['lr']
        - group['initial_lr']（LambdaLR 的真实 base 来源）
        - group['lr']（step() 时实际写回的字段）
    """
    optimizer.defaults.setdefault("initial_lr", optimizer.defaults["lr"])
    optimizer.defaults["lr"] = config.peak_lr
    for group in optimizer.param_groups:
        # 强制刷新 initial_lr（如果之前已经存在，要更新成新 base）
        group["initial_lr"] = config.peak_lr
        group["lr"] = config.peak_lr
    init_value = config.peak_lr / (config.warmup_steps + 1)
    return warmup_cosine_decay_schedule(
        optimizer,
        init_value=init_value,
        peak_value=config.peak_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=config.decay_steps,
        end_value=config.decay_lr,
        last_epoch=last_epoch,
    )


__all__ = [
    "CosineDecayWithWarmupConfig",
    "warmup_cosine_decay_schedule",
    "attach_lr_scheduler",
]
