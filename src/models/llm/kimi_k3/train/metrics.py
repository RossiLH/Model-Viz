"""训练指标（对齐 kimi3-master train/metrics.py）。

除 loss 外，K3 重点关注两类指标：
- MoE 负载均衡（Quantile Balancing 是否生效，论文 §2.3.3）
- KDA 衰减统计（下界参数化是否在预期范围内，Eq. 5）
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["RunningMean", "MetricLog", "expert_balance", "decay_stats"]


class RunningMean:
    """滑动窗口均值计算器。

    维护最近 ``window`` 个值的列表，超过时弹出最早值。
    用于平滑训练 loss 等指标。
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.values: list[float] = []

    def add(self, value: float) -> None:
        """添加一个值，超窗时丢弃最早值。"""
        self.values.append(value)
        if len(self.values) > self.window:
            self.values.pop(0)

    @property
    def mean(self) -> float:
        """当前窗口均值，空窗口返回 NaN。"""
        return sum(self.values) / len(self.values) if self.values else math.nan

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class MetricLog:
    """追加式 JSONL 日志 + 内存历史，便于断言。"""

    path: Path | None = None
    history: list[dict] = field(default_factory=list)
    start: float = field(default_factory=time.time)

    def log(self, **kw) -> dict:
        """记录一条指标，可选写入 JSONL 文件。"""
        record = {"wall": round(time.time() - self.start, 3), **kw}
        self.history.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
        return record

    def series(self, key: str) -> list[float]:
        """提取指定 key 的所有值（跳过不含该 key 的记录）。"""
        return [r[key] for r in self.history if key in r]

    def smoothed(self, key: str, window: int = 10) -> list[float]:
        """对指定 key 做滑动窗口均值平滑。"""
        values = self.series(key)
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            chunk = values[lo : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out


def expert_balance(model) -> dict[str, float]:
    """聚合所有 MoE 层的负载均衡统计。

    遍历模型中所有 router，收集 max_min_ratio、cv、max_over_mean，
    返回最大比值、平均 cv、最大 over-mean 三个聚合指标。
    """
    ratios, cvs, over = [], [], []
    for router in model.model.routers():
        stats = router.load_balance_stats()
        ratios.append(stats["max_min_ratio"])
        cvs.append(stats["cv"])
        over.append(stats["max_over_mean"])
    if not ratios:
        return {}
    return {
        "balance/max_min_ratio": max(ratios),
        "balance/mean_cv": sum(cvs) / len(cvs),
        "balance/max_over_mean": max(over),
    }


def decay_stats(model) -> dict[str, float]:
    """报告 KDA 层的实际 retention factor 分布。

    Eq. 5 保证 alpha 在 (exp(g_min), 1) 范围内；观察模型收敛后 alpha
    在该范围内的位置，可判断线性层保留了多少历史信息。
    """
    from models.llm.kimi_k3.modules.kda import KimiDeltaAttention

    mins, maxs, means = [], [], []
    for module in model.modules():
        if isinstance(module, KimiDeltaAttention):
            with_bias = module.alpha_bias.detach()
            scale = module.a_log.detach().exp().repeat_interleave(module.head_dim_k)
            g = module.cfg.g_min * (scale * with_bias).sigmoid()
            alpha = g.exp()
            mins.append(float(alpha.min()))
            maxs.append(float(alpha.max()))
            means.append(float(alpha.mean()))
    if not mins:
        return {}
    return {
        "kda/alpha_min": min(mins),
        "kda/alpha_max": max(maxs),
        "kda/alpha_mean": sum(means) / len(means),
    }
