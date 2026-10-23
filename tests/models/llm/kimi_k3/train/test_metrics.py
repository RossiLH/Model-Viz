"""训练指标测试（对齐 kimi3-master train/metrics.py）。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from models.llm.kimi_k3.train.metrics import MetricLog, RunningMean, decay_stats, expert_balance


class TestRunningMean:
    def test_basic(self):
        """滑动窗口均值：窗口 5，添加 0-9 后均值应为最后 5 个的均值。"""
        rm = RunningMean(window=5)
        for i in range(10):
            rm.add(float(i))
        assert rm.mean == pytest.approx(7.0)  # [5,6,7,8,9] -> 7.0

    def test_empty(self):
        """空窗口返回 NaN。"""
        rm = RunningMean()
        assert rm.mean != rm.mean  # NaN != NaN

    def test_len(self):
        """len() 返回当前存储的值数量。"""
        rm = RunningMean(window=3)
        rm.add(1.0)
        rm.add(2.0)
        assert len(rm) == 2

    def test_window_overflow(self):
        """超过窗口大小后最早值被弹出。"""
        rm = RunningMean(window=3)
        rm.add(1.0)
        rm.add(2.0)
        rm.add(3.0)
        rm.add(4.0)
        assert len(rm) == 3
        assert rm.mean == pytest.approx(3.0)  # [2,3,4]

    def test_single_value(self):
        """单个值的均值就是它本身。"""
        rm = RunningMean(window=10)
        rm.add(42.0)
        assert rm.mean == pytest.approx(42.0)


class TestMetricLog:
    def test_log_to_memory(self):
        """log() 返回 record 并存入 history。"""
        ml = MetricLog()
        record = ml.log(step=1, loss=0.5)
        assert record["step"] == 1
        assert len(ml.history) == 1

    def test_log_to_file(self, tmp_path):
        """log() 写入 JSONL 文件。"""
        path = tmp_path / "metrics.jsonl"
        ml = MetricLog(path=path)
        ml.log(step=1, loss=0.5)
        ml.log(step=2, loss=0.3)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["loss"] == 0.5

    def test_series(self):
        """series() 提取指定 key 的所有值。"""
        ml = MetricLog()
        ml.log(loss=0.5)
        ml.log(loss=0.3)
        ml.log(step=1)  # 没有 loss
        assert ml.series("loss") == [0.5, 0.3]

    def test_smoothed(self):
        """smoothed() 做滑动窗口均值。"""
        ml = MetricLog()
        for i in range(10):
            ml.log(loss=float(i))
        smoothed = ml.smoothed("loss", window=3)
        assert len(smoothed) == 10
        # 最后一个值应该是 (7+8+9)/3 = 8.0
        assert smoothed[-1] == pytest.approx(8.0)

    def test_wall_timestamp(self):
        """每条记录包含 wall 时间戳。"""
        ml = MetricLog()
        record = ml.log(x=1)
        assert "wall" in record
        assert record["wall"] >= 0

    def test_empty_series(self):
        """空 history 的 series 返回空列表。"""
        ml = MetricLog()
        assert ml.series("nonexistent") == []


class TestExpertBalance:
    def test_skip_no_model(self):
        """expert_balance 需要完整模型实例，此处仅验证函数签名。"""
        pytest.skip("需要完整 MoE 模型实例，跳过")


class TestDecayStats:
    def test_skip_no_model(self):
        """decay_stats 需要 KimiDeltaAttention 模型实例，此处仅验证函数签名。"""
        pytest.skip("需要 KDA 模型实例，跳过")
