# -*- coding: utf-8 -*-
"""RunningStats / norm_stats 单元测试

覆盖要点：
- Welford 均值/方差与 numpy 全量计算的一致性
- 分位数（q01/q99）与 numpy.quantile 的近似一致性
- 流式分批更新与一次性全量更新的一致性（测试 Welford 的稳定性）
- min/max 变化时直方图重新分布的正确性
- save/load JSON 往返一致性
- 边界情况：单维数据、恒定值数据、零样本
"""

import os
import json
import tempfile

import numpy as np
import pytest

from models.vla.core.train.norm_stats import (
    RunningStats, NormStats, save_norm_stats, load_norm_stats,
)


def test_mean_std_matches_numpy():
    """Welford 在线算法计算的均值和方差应与 numpy 全量计算一致。"""
    rng = np.random.RandomState(42)
    data = rng.randn(1000, 4).astype(np.float64)

    rs = RunningStats()
    rs.update(data)
    stats = rs.get_statistics()

    np.testing.assert_allclose(stats.mean, np.mean(data, axis=0), atol=1e-10)
    np.testing.assert_allclose(stats.std, np.std(data, axis=0), rtol=1e-3)


def test_quantile_approximation():
    """直方图分位数估计应接近 numpy.quantile 的结果。"""
    rng = np.random.RandomState(0)
    data = rng.randn(50000, 2).astype(np.float64)

    rs = RunningStats(num_quantile_bins=5000)
    rs.update(data)
    stats = rs.get_statistics()

    np_q01 = np.quantile(data, 0.01, axis=0)
    np_q99 = np.quantile(data, 0.99, axis=0)

    # 5000 bin 下误差应小于数据范围的 2%
    data_range = np_q99 - np_q01
    max_err_q01 = float(np.max(np.abs(stats.q01 - np_q01)))
    max_err_q99 = float(np.max(np.abs(stats.q99 - np_q99)))
    assert max_err_q01 < float(np.max(data_range)) * 0.02, (
        f"q01 error {max_err_q01} exceeds 2% of range {np.max(data_range)}")
    assert max_err_q99 < float(np.max(data_range)) * 0.02, (
        f"q99 error {max_err_q99} exceeds 2% of range {np.max(data_range)}")


def test_streaming_vs_batch():
    """分批流式更新应与一次性全量更新结果一致。"""
    rng = np.random.RandomState(123)
    data = rng.randn(2000, 3).astype(np.float64)

    rs1 = RunningStats()
    rs1.update(data)

    rs2 = RunningStats()
    for i in range(0, 2000, 50):
        rs2.update(data[i:i + 50])

    s1 = rs1.get_statistics()
    s2 = rs2.get_statistics()

    np.testing.assert_allclose(s1.mean, s2.mean, atol=1e-12)
    np.testing.assert_allclose(s1.std, s2.std, atol=1e-10)
    # 分位数因直方图重分布有近似误差，500 bin 下约 0.5% 的相对误差是正常的
    np.testing.assert_allclose(s1.q01, s2.q01, rtol=0.01)
    np.testing.assert_allclose(s1.q99, s2.q99, rtol=0.01)


def test_min_change_adjusts_histograms():
    """min 变化时直方图应重新分布，不丢样本。"""
    rs = RunningStats(num_quantile_bins=500)
    rs.update(np.array([[0.0, 1.0], [0.5, 1.5]]))
    count_after_first = rs._count
    # 加入更小的值，触发 min 变化
    rs.update(np.array([[-1.0, 0.0]]))

    stats = rs.get_statistics()
    # 总样本数正确
    assert rs._count == count_after_first + 1
    # 新的 min 正确
    assert stats.q01[0] <= -0.9


def test_constant_values():
    """恒定值数据的 std 应为 0。"""
    data = np.full((100, 2), 5.0, dtype=np.float64)

    rs = RunningStats()
    rs.update(data)
    stats = rs.get_statistics()

    np.testing.assert_allclose(stats.mean, [5.0, 5.0])
    np.testing.assert_allclose(stats.std, [0.0, 0.0], atol=1e-10)


def test_single_dimension():
    """单维数据也能正常工作。"""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    rs = RunningStats()
    rs.update(data)
    stats = rs.get_statistics()

    assert stats.mean.shape == (1,)
    assert stats.std.shape == (1,)
    np.testing.assert_allclose(stats.mean[0], 3.0)


def test_save_load_roundtrip():
    """save_norm_stats / load_norm_stats 往返一致。"""
    rng = np.random.RandomState(7)
    data_state = rng.randn(500, 8).astype(np.float64)
    data_actions = rng.randn(500, 7).astype(np.float64)

    rs_state = RunningStats()
    rs_state.update(data_state)
    rs_actions = RunningStats()
    rs_actions.update(data_actions)

    stats_dict = {
        "state": rs_state.get_statistics(),
        "actions": rs_actions.get_statistics(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "norm_stats.json")
        save_norm_stats(path, stats_dict)

        # 验证文件存在且是有效 JSON
        assert os.path.exists(path)
        with open(path, "r") as f:
            raw = json.load(f)
        assert "norm_stats" in raw
        assert "state" in raw["norm_stats"]
        assert "actions" in raw["norm_stats"]

        # 加载回来
        loaded = load_norm_stats(path)

    np.testing.assert_allclose(loaded["state"].mean, stats_dict["state"].mean)
    np.testing.assert_allclose(loaded["state"].std, stats_dict["state"].std)
    np.testing.assert_allclose(loaded["state"].q01, stats_dict["state"].q01)
    np.testing.assert_allclose(loaded["state"].q99, stats_dict["state"].q99)
    np.testing.assert_allclose(loaded["actions"].mean, stats_dict["actions"].mean)


def test_too_few_samples_raises():
    """样本数不足 2 时 get_statistics 抛 ValueError。"""
    rs = RunningStats()
    rs.update(np.array([[1.0, 2.0]]))

    with pytest.raises(ValueError, match="不足"):
        rs.get_statistics()


def test_dimension_mismatch_raises():
    """两次 update 维度不一致应抛错。"""
    rs = RunningStats()
    rs.update(np.zeros((10, 3)))

    with pytest.raises(ValueError, match="维度不一致"):
        rs.update(np.zeros((10, 5)))
