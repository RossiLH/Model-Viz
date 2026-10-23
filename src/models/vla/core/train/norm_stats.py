# -*- coding: utf-8 -*-
"""流式归一化统计量（均值、标准差、分位数）

设计动机：
官方 openpi 的 RunningStats 用 Welford 在线算法算 mean/std，用直方图估分位数，
全部 O(1) 内存，O(N) 时间，支持流式增量更新，可以处理装不进内存的大数据集。
本模块完全复刻同一逻辑，保证同一份数据算出的 q01/q99 与官方逐位可比。

为什么不用 numpy.quantile：
1. numpy.quantile 需要全量数据排序后计算，对于大数据集 O(N log N) 且占内存；
2. 官方是流式的，为了对齐必须复刻流式版本；
3. 直方图分位数是近似算法，但 5000 bin 下精度足够。
"""

from dataclasses import dataclass
from typing import Dict, Optional

import json
import numpy as np


@dataclass
class NormStats:
    """归一化统计量。

    所有数组的形状都是 (D,)，D 是向量维度。
    mean/std 用作 z-score；q01/q99 用作分位数归一化。
    两种归一化通常不会同时启用，由调用方决定。
    """
    # 让 mean/std 也可空（z-score 场景）—— PI0.5 走 quantile 路线时不需要 mean/std
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None


class RunningStats:
    """流式计算均值、方差、分位数的统计量累加器。

    实现要点：
    1. mean / var 用 Welford 在线算法（数值稳定，不需要累积所有样本）；
    2. 分位数用直方图近似，默认 5000 个 bin，与官方一致；
    3. min/max 变化时重新分布直方图 bin 边界，保证 bin 始终覆盖所有已见数据。
    """

    def __init__(self, num_quantile_bins: int = 5000):
        """初始化空统计量。

        Args:
            num_quantile_bins: 用于分位数估计的直方图 bin 数，越大越精确。
        """
        self._num_bins = int(num_quantile_bins)
        self._count = 0
        self._dim = None
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None

    def update(self, batch: np.ndarray) -> None:
        """用一批数据更新统计量。

        Args:
            batch: 任意形状的数组，最后一维是特征维 (..., D)。
                   会被展平成 (N, D) 后逐元素统计。
        """
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 0:
            batch = batch.reshape(1, 1)
        elif batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        else:
            batch = batch.reshape(-1, batch.shape[-1])

        num_elements, dim = batch.shape
        if num_elements == 0:
            return

        if self._dim is None:
            self._dim = dim
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch ** 2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._count = num_elements
            self._histograms = []
            self._bin_edges = []
            for i in range(dim):
                lo = float(self._min[i])
                hi = float(self._max[i])
                if hi - lo < 1e-12:
                    hi = lo + 1.0
                edges = np.linspace(lo - 1e-10, hi + 1e-10, self._num_bins + 1)
                hist, _ = np.histogram(batch[:, i], bins=edges)
                self._histograms.append(hist.astype(np.float64))
                self._bin_edges.append(edges)
            return

        if dim != self._dim:
            raise ValueError(
                f"维度不一致：新数据 {dim} 维，已有 {self._dim} 维")

        new_min = np.min(batch, axis=0)
        new_max = np.max(batch, axis=0)
        min_changed = bool(np.any(new_min < self._min))
        max_changed = bool(np.any(new_max > self._max))
        if min_changed or max_changed:
            old_min = self._min.copy()
            old_max = self._max.copy()
            self._min = np.minimum(self._min, new_min)
            self._max = np.maximum(self._max, new_max)
            self._adjust_histograms(old_min, old_max)

        batch_mean = np.mean(batch, axis=0)
        batch_msq = np.mean(batch ** 2, axis=0)
        total = self._count + num_elements
        self._mean += (batch_mean - self._mean) * (num_elements / total)
        self._mean_of_squares += (batch_msq - self._mean_of_squares) * (num_elements / total)
        self._count = total

        for i in range(self._dim):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist.astype(np.float64)

    def _adjust_histograms(self, old_min: np.ndarray, old_max: np.ndarray) -> None:
        """当 min/max 变化时，重新分布直方图到更宽的 bin 边界。

        用旧 bin 左边界作为样本值，重新做直方图到新边界。
        这是近似做法（丢失 bin 内部分布信息），但误差很小。
        """
        for i in range(self._dim):
            if (float(self._min[i]) == float(old_min[i])
                    and float(self._max[i]) == float(old_max[i])):
                continue
            lo = float(self._min[i])
            hi = float(self._max[i])
            if hi - lo < 1e-12:
                hi = lo + 1.0
            new_edges = np.linspace(lo - 1e-10, hi + 1e-10, self._num_bins + 1)
            old_left = self._bin_edges[i][:-1]
            new_hist, _ = np.histogram(
                old_left, bins=new_edges, weights=self._histograms[i])
            self._histograms[i] = new_hist.astype(np.float64)
            self._bin_edges[i] = new_edges

    def get_statistics(self) -> NormStats:
        """返回当前累积的统计量。

        Returns:
            NormStats: 含 mean / std / q01 / q99
        """
        if self._count < 2:
            raise ValueError("样本数不足 2，无法计算统计量")

        variance = self._mean_of_squares - self._mean ** 2
        std = np.sqrt(np.maximum(0.0, variance))

        q01, q99 = self._compute_quantiles([0.01, 0.99])

        return NormStats(
            mean=self._mean.copy(),
            std=std.copy(),
            q01=q01,
            q99=q99,
        )

    def _compute_quantiles(self, quantiles):
        """从直方图估计分位数。

        Args:
            quantiles: 分位数列表，如 [0.01, 0.99]

        Returns:
            每个分位数对应一个 (D,) 数组
        """
        results = []
        for q in quantiles:
            target = q * self._count
            values = np.zeros(self._dim, dtype=np.float64)
            for i in range(self._dim):
                cumsum = np.cumsum(self._histograms[i])
                idx = int(np.searchsorted(cumsum, target))
                idx = max(0, min(idx, len(self._bin_edges[i]) - 2))
                cum_before = float(cumsum[idx - 1]) if idx > 0 else 0.0
                bin_count = float(self._histograms[i][idx])
                if bin_count > 0:
                    frac = (target - cum_before) / bin_count
                else:
                    frac = 0.0
                values[i] = (
                    self._bin_edges[i][idx]
                    + frac * (self._bin_edges[i][idx + 1] - self._bin_edges[i][idx])
                )
            results.append(values)
        return results


def save_norm_stats(path: str, stats: Dict[str, NormStats]) -> None:
    """把归一化统计量存成 JSON 文件。

    JSON 格式与官方 openpi 的 norm_stats.json 兼容，
    可以直接加载官方预计算的统计量。

    Args:
        path:  输出文件路径（.json）
        stats: {key: NormStats} 字典，如 {"state": ..., "actions": ...}
    """
    out = {}
    for key, s in stats.items():
        entry = {
            "mean": s.mean.tolist(),
            "std": s.std.tolist(),
        }
        if s.q01 is not None:
            entry["q01"] = s.q01.tolist()
        if s.q99 is not None:
            entry["q99"] = s.q99.tolist()
        out[key] = entry

    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"norm_stats": out}, f, indent=2, ensure_ascii=False)


def load_norm_stats(path: str) -> Dict[str, NormStats]:
    """从 JSON 文件加载归一化统计量。

    Args:
        path: JSON 文件路径

    Returns:
        {key: NormStats} 字典
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("norm_stats", data)
    result = {}
    for key, entry in raw.items():
        mean = np.asarray(entry["mean"], dtype=np.float64)
        std = np.asarray(entry["std"], dtype=np.float64)
        q01 = np.asarray(entry["q01"], dtype=np.float64) if "q01" in entry else None
        q99 = np.asarray(entry["q99"], dtype=np.float64) if "q99" in entry else None
        result[key] = NormStats(mean=mean, std=std, q01=q01, q99=q99)
    return result
