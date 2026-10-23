"""混合优化器测试。"""
import torch
import torch.nn as nn

from models.llm.kimi_k3.optim.muon import Muon
from models.llm.kimi_k3.optim.hybrid import HybridMuonAdamW


class TestHybridMuonAdamW:
    """混合优化器测试。"""

    def _build_hybrid(self):
        """构建混合优化器。"""
        model = nn.Linear(8, 4, bias=True)
        muon = Muon(
            [{"params": [model.weight], "use_muon": True, "num_heads": None}],
            lr=0.01,
        )
        adamw = torch.optim.AdamW(
            [{"params": [model.bias], "weight_decay": 0.0}],
            lr=0.001,
        )
        return HybridMuonAdamW(muon, adamw), model

    def test_step_executes(self):
        """正常梯度下 step 执行。"""
        opt, model = self._build_hybrid()
        opt.zero_grad()
        loss = (model(torch.randn(4, 8)) ** 2).mean()
        loss.backward()
        report = opt.step()
        assert report.executed
        assert report.nonfinite_parameters == 0

    def test_nan_gradient_skips(self):
        """NaN 梯度跳过整步。"""
        opt, model = self._build_hybrid()
        opt.zero_grad()
        loss = (model(torch.randn(4, 8)) ** 2).mean()
        loss.backward()
        # 注入 NaN 梯度
        model.weight.grad = torch.full_like(model.weight.grad, float("nan"))
        report = opt.step()
        assert not report.executed
        assert report.nonfinite_parameters > 0

    def test_set_lr(self):
        """set_lr 同时设置两个优化器的学习率。"""
        opt, model = self._build_hybrid()
        opt.set_lr(0.05, muon_lr=0.1)
        assert opt.muon.param_groups[0]["lr"] == 0.1
        assert opt.adamw.param_groups[0]["lr"] == 0.05

    def test_state_dict_roundtrip(self):
        """状态序列化/反序列化。"""
        opt, model = self._build_hybrid()
        # 先执行一步
        opt.zero_grad()
        loss = (model(torch.randn(4, 8)) ** 2).mean()
        loss.backward()
        opt.step()

        # 序列化
        state = opt.state_dict()
        assert "muon" in state
        assert "adamw" in state
        assert state["step_number"] == 1

        # 反序列化
        opt2, model2 = self._build_hybrid()
        opt2.load_state_dict(state)
        assert opt2.step_number == 1

    def test_zero_grad(self):
        """zero_grad 清除两个优化器的梯度。"""
        opt, model = self._build_hybrid()
        loss = (model(torch.randn(4, 8)) ** 2).mean()
        loss.backward()
        opt.zero_grad()
        assert model.weight.grad is None
        assert model.bias.grad is None
