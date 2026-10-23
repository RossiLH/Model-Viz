"""学习率调度器测试。"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from models.llm.kimi_k3.optim.lr_scheduler import CosineWarmupLR, cosine_with_warmup, lr_lambda_for, wsd_schedule
from models.llm.kimi_k3.optim.muon import Muon


class TestCosineWithWarmup:
    """cosine_with_warmup 函数测试。"""

    def test_warmup_phase(self):
        """warmup 阶段线性增长。"""
        assert cosine_with_warmup(0, 1000, warmup_ratio=0.01) < 0.2
        assert cosine_with_warmup(5, 1000, warmup_ratio=0.01) > cosine_with_warmup(0, 1000, warmup_ratio=0.01)

    def test_peak_at_warmup_end(self):
        """warmup 结束时接近1.0。"""
        lr = cosine_with_warmup(10, 1000, warmup_ratio=0.01)
        assert abs(lr - 1.0) < 0.15

    def test_decay_to_min(self):
        """训练结束时衰减到 min_lr_ratio。"""
        lr = cosine_with_warmup(999, 1000, warmup_ratio=0.01, min_lr_ratio=0.1)
        assert abs(lr - 0.1) < 0.05

    def test_output_range(self):
        """输出始终在 [min_lr_ratio, 1.0] 范围内。"""
        for step in range(0, 1000, 10):
            lr = cosine_with_warmup(step, 1000, warmup_ratio=0.01, min_lr_ratio=0.1)
            assert 0.1 <= lr <= 1.0


class TestWsdSchedule:
    """wsd_schedule 函数测试。"""

    def test_warmup(self):
        """warmup 阶段线性增长。"""
        assert wsd_schedule(0, 1000) < 0.2

    def test_stable_phase(self):
        """稳定阶段返回1.0。"""
        lr = wsd_schedule(500, 1000, warmup_ratio=0.01, decay_ratio=0.1)
        assert abs(lr - 1.0) < 0.01

    def test_decay_phase(self):
        """衰减阶段下降。"""
        lr_900 = wsd_schedule(900, 1000, warmup_ratio=0.01, decay_ratio=0.1)
        lr_950 = wsd_schedule(950, 1000, warmup_ratio=0.01, decay_ratio=0.1)
        assert lr_900 > lr_950


class TestCosineWarmupLR:
    """CosineWarmupLR 调度器测试。"""

    def test_standard_optimizer(self):
        """标准优化器调度。"""
        model = nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineWarmupLR(opt, total_steps=100, warmup_steps=10)

        scheduler.step()
        # 第1步应在 warmup 中
        lr = opt.param_groups[0]["lr"]
        assert 0 < lr <= 0.1

    def test_hybrid_optimizer(self):
        """混合优化器双 LR 调度。"""
        model = nn.Linear(4, 4, bias=True)
        muon = Muon(
            [{"params": [model.weight], "use_muon": True, "num_heads": None}],
            lr=0.1,
        )
        adamw = torch.optim.AdamW(
            [{"params": [model.bias]}],
            lr=0.01,
        )
        from models.llm.kimi_k3.optim.hybrid import HybridMuonAdamW
        hybrid = HybridMuonAdamW(muon, adamw)

        scheduler = CosineWarmupLR(
            hybrid, total_steps=100, warmup_steps=10,
            min_lr=0.001, min_muon_lr=0.01,
        )

        for _ in range(20):
            scheduler.step()

        lr_dict = scheduler.get_lr_dict()
        assert "muon_lr" in lr_dict
        assert "adamw_lr" in lr_dict
        assert lr_dict["muon_lr"] > 0
        assert lr_dict["adamw_lr"] > 0

    def test_get_lr_dict(self):
        """get_lr_dict 返回正确结构。"""
        model = nn.Linear(4, 4)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = CosineWarmupLR(opt, total_steps=100, warmup_steps=10)
        scheduler.step()
        d = scheduler.get_lr_dict()
        assert "step" in d
        assert "lr" in d


@dataclass
class _FakeOptimCfg:
    """模拟 OptimConfig 的轻量数据类，用于 lr_lambda_for 测试。"""
    schedule: str = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1


class TestLrLambdaFor:
    """lr_lambda_for 工厂函数测试。"""

    def test_cosine_default(self):
        """默认 schedule 返回 cosine 调度。"""
        cfg = _FakeOptimCfg(schedule="cosine")
        fn = lr_lambda_for(cfg, total_steps=1000)
        # warmup 阶段
        assert fn(0) < 0.2
        # 训练末尾
        assert abs(fn(999) - 0.1) < 0.05

    def test_wsd_schedule(self):
        """wsd schedule 返回 WSD 调度。"""
        cfg = _FakeOptimCfg(schedule="wsd")
        fn = lr_lambda_for(cfg, total_steps=1000)
        # warmup 阶段
        assert fn(0) < 0.2
        # 稳定阶段（中间）
        assert abs(fn(500) - 1.0) < 0.01
        # 衰减末尾
        assert fn(999) < 1.0

    def test_returns_callable(self):
        """返回值可调用且返回 float。"""
        cfg = _FakeOptimCfg()
        fn = lr_lambda_for(cfg, 500)
        assert callable(fn)
        assert isinstance(fn(0), float)

    def test_output_range(self):
        """cosine 调度输出始终在 [min_lr_ratio, 1.0]。"""
        cfg = _FakeOptimCfg(schedule="cosine", min_lr_ratio=0.1)
        fn = lr_lambda_for(cfg, 1000)
        for step in range(0, 1000, 10):
            val = fn(step)
            assert 0.1 <= val <= 1.0, f"step={step}, val={val}"

    def test_wsd_output_range(self):
        """wsd 调度输出始终在 [min_lr_ratio, 1.0]。"""
        cfg = _FakeOptimCfg(schedule="wsd", min_lr_ratio=0.1)
        fn = lr_lambda_for(cfg, 1000)
        for step in range(0, 1000, 10):
            val = fn(step)
            assert 0.1 <= val <= 1.0, f"step={step}, val={val}"

    def test_warmup_monotonic(self):
        """warmup 阶段乘子严格递增。"""
        cfg = _FakeOptimCfg(schedule="cosine", warmup_ratio=0.05)
        fn = lr_lambda_for(cfg, 1000)
        vals = [fn(i) for i in range(50)]
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
