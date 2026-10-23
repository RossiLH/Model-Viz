"""学习率调度器：Cosine with Warmup，支持 Muon/AdamW 双 LR。

与 kimi3-master schedules.py 和 kimi-k3-pytorch-main scheduler.py 对齐。
"""
from __future__ import annotations

import math
from typing import Optional


def cosine_with_warmup(
    step: int,
    total_steps: int,
    *,
    warmup_ratio: float = 0.01,
    min_lr_ratio: float = 0.1,
) -> float:
    """Cosine 学习率调度 + 线性 Warmup。

    与 kimi3-master schedules.py:17-27 一致。

    Args:
        step: 当前步数（0-indexed）
        total_steps: 总训练步数
        warmup_ratio: warmup 步数占比（默认1%）
        min_lr_ratio: 最小学习率与峰值的比值

    Returns:
        学习率乘子，在 [min_lr_ratio, 1.0] 范围内
    """
    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def wsd_schedule(
    step: int,
    total_steps: int,
    *,
    warmup_ratio: float = 0.01,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup-Stable-Decay 学习率调度（消融用）。

    与 kimi3-master schedules.py:30-46 一致。
    """
    warmup = max(1, int(total_steps * warmup_ratio))
    decay_start = int(total_steps * (1.0 - decay_ratio))
    if step < warmup:
        return (step + 1) / warmup
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)


class CosineWarmupLR:
    """Cosine with Warmup 学习率调度器，支持 Muon/AdamW 双 LR。

    与 kimi-k3-pytorch-main scheduler.py 的 WarmupCosineLR 对齐。
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
        min_muon_lr: Optional[float] = None,
        schedule: str = "cosine",
    ):
        """
        Args:
            optimizer: HybridMuonAdamW 或标准 PyTorch 优化器
            total_steps: 总训练步数
            warmup_steps: warmup 步数
            min_lr: AdamW 最小学习率
            min_muon_lr: Muon 最小学习率（None 则使用 min_lr）
            schedule: 调度类型 ("cosine" 或 "wsd")
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.min_muon_lr = min_muon_lr
        self.schedule = schedule
        self.step_num = 0

        # 检测是否为混合优化器
        self.is_hybrid = hasattr(optimizer, "muon") and hasattr(optimizer, "adamw")

        if self.is_hybrid:
            self.base_adamw_lrs = [float(g["lr"]) for g in optimizer.adamw.param_groups]
            self.base_muon_lrs = [float(g["lr"]) for g in optimizer.muon.param_groups]
        else:
            self.base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    def _compute_lr(self, base_lr: float, min_lr: float, step: int) -> float:
        """计算指定步数的学习率。"""
        if self.schedule == "wsd":
            multiplier = wsd_schedule(
                step, self.total_steps,
                warmup_ratio=self.warmup_steps / max(1, self.total_steps),
                min_lr_ratio=min_lr / max(1e-12, base_lr),
            )
        else:
            multiplier = cosine_with_warmup(
                step, self.total_steps,
                warmup_ratio=self.warmup_steps / max(1, self.total_steps),
                min_lr_ratio=min_lr / max(1e-12, base_lr),
            )
        return base_lr * multiplier

    def step(self) -> None:
        """推进一步调度。"""
        self.step_num += 1
        if self.is_hybrid:
            muon_min = self.min_lr if self.min_muon_lr is None else self.min_muon_lr
            for i, g in enumerate(self.optimizer.adamw.param_groups):
                g["lr"] = self._compute_lr(self.base_adamw_lrs[i], self.min_lr, self.step_num)
            for i, g in enumerate(self.optimizer.muon.param_groups):
                g["lr"] = self._compute_lr(self.base_muon_lrs[i], muon_min, self.step_num)
        else:
            for i, g in enumerate(self.optimizer.param_groups):
                g["lr"] = self._compute_lr(self.base_lrs[i], self.min_lr, self.step_num)

    def get_lr_dict(self) -> dict:
        """返回当前学习率字典（用于日志）。"""
        if self.is_hybrid:
            return {
                "step": self.step_num,
                "muon_lr": float(self.optimizer.muon.param_groups[0]["lr"]),
                "adamw_lr": float(self.optimizer.adamw.param_groups[0]["lr"]),
            }
        return {
            "step": self.step_num,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }


def lr_lambda_for(optim_cfg, total_steps: int):
    """返回 step -> multiplier 的可调用对象（对齐 kimi3-master optim/schedules.py:49-63）。

    根据 optim_cfg.schedule 选择 cosine_with_warmup 或 wsd_schedule，
    返回一个接收 step（0-indexed）返回 lr multiplier 的 lambda。

    Args:
        optim_cfg: OptimConfig 配置对象，需包含 schedule/warmup_ratio/min_lr_ratio
        total_steps: 总训练步数

    Returns:
        step -> multiplier 的可调用对象
    """
    if optim_cfg.schedule == "wsd":
        return lambda s: wsd_schedule(
            s,
            total_steps,
            warmup_ratio=optim_cfg.warmup_ratio,
            min_lr_ratio=optim_cfg.min_lr_ratio,
        )
    return lambda s: cosine_with_warmup(
        s,
        total_steps,
        warmup_ratio=optim_cfg.warmup_ratio,
        min_lr_ratio=optim_cfg.min_lr_ratio,
    )
