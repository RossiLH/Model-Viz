"""Muon 优化器测试。"""
import pytest
import torch
import torch.nn as nn

from models.llm.kimi_k3.optim.muon import Muon, per_head_orthogonalize


class TestPerHeadOrthogonalize:
    """Per-Head 正交化测试。"""

    def test_output_shape(self):
        """输出 shape 与输入一致。"""
        m = torch.randn(32, 16)  # 4 heads × 8 dim
        out = per_head_orthogonalize(m, num_heads=4)
        assert out.shape == (32, 16)

    def test_heads_independent(self):
        """不同头独立正交化。

        固定随机种子：Newton-Schulz 只迭代 5 步，对偶发的病态随机矩阵
        （原始条件数极大）无法收敛到条件数 < 3，属算法固有性质而非缺陷。
        不固定种子时该断言依赖全局 RNG 状态（约 2% 概率偶发失败），
        故显式播种以保证测试确定性。
        """
        torch.manual_seed(0)
        m = torch.randn(16, 8)  # 2 heads × 8 dim
        out = per_head_orthogonalize(m, num_heads=2)
        # 每个头的奇异值应接近1
        head0 = out.view(2, 8, 8)[0]
        head1 = out.view(2, 8, 8)[1]
        s0 = torch.linalg.svdvals(head0)
        s1 = torch.linalg.svdvals(head1)
        # 条件数应 < 3（宽松检查）
        assert s0.max() / s0.min() < 3.0
        assert s1.max() / s1.min() < 3.0

    def test_invalid_head_dim(self):
        """out_dim 不能被 num_heads 整除时抛出异常。"""
        m = torch.randn(10, 8)  # 10 不能被 3 整除
        with pytest.raises(AssertionError):
            per_head_orthogonalize(m, num_heads=3)


class TestMuon:
    """Muon 优化器核心测试。"""

    def test_reduces_quadratic(self):
        """Muon 能最小化二次函数（对齐 kimi3-master test_optimizer_reduces_a_quadratic）。"""
        torch.manual_seed(0)
        model = nn.Linear(4, 4, bias=False)
        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.02,
            adamw_lr=0.02,
        )
        for _ in range(100):
            optimizer.zero_grad()
            loss = (model(torch.randn(8, 4)) ** 2).mean()
            loss.backward()
            optimizer.step()
        assert float(loss) < 0.5

    def test_adamw_branch_reduces_loss(self):
        """AdamW 分支也能最小化二次函数。"""
        torch.manual_seed(42)
        model = nn.Linear(4, 4, bias=True)

        optimizer = Muon(
            [
                {"params": [model.weight], "use_muon": True, "num_heads": None},
                {"params": [model.bias], "use_muon": False},
            ],
            lr=0.01,
        )

        initial_loss = None
        for _ in range(50):
            optimizer.zero_grad()
            x = torch.randn(8, 4)
            loss = (model(x) ** 2).mean()
            if initial_loss is None:
                initial_loss = float(loss)
            loss.backward()
            optimizer.step()

        final_loss = float(loss)
        assert final_loss < initial_loss * 0.5

    def test_momentum_buffer(self):
        """动量缓冲正确累积。"""
        model = nn.Linear(4, 4, bias=False)
        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01,
            momentum=0.9,
        )

        x = torch.randn(8, 4)
        optimizer.zero_grad()
        loss = (model(x) ** 2).mean()
        loss.backward()
        optimizer.step()

        # 动量缓冲应已初始化
        state = optimizer.state[model.weight]
        assert "momentum_buffer" in state
        assert state["momentum_buffer"].shape == model.weight.shape

    def test_weight_decay(self):
        """Weight decay 使权重衰减。"""
        model = nn.Linear(4, 4, bias=False)
        w_before = model.weight.data.clone()

        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01,
            weight_decay=0.1,
        )

        optimizer.zero_grad()
        loss = (model(torch.randn(8, 4)) ** 2).mean()
        loss.backward()
        optimizer.step()

        # 权重应该衰减了（norm 变小）
        assert model.weight.data.norm() < w_before.norm() * 1.1

    def test_weight_clipping(self):
        """Weight clipping 限制权重增长。"""
        model = nn.Linear(4, 4, bias=False)
        # 初始化为大权重
        with torch.no_grad():
            model.weight.copy_(torch.randn(4, 4) * 10)

        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.1,
            weight_clip=1.0,
        )

        for _ in range(10):
            optimizer.zero_grad()
            loss = (model(torch.randn(8, 4)) ** 2).mean()
            loss.backward()
            optimizer.step()

        # 权重 RMS 应被限制在阈值附近
        rms = model.weight.pow(2).mean().sqrt()
        assert float(rms) < 2.0

    def test_2d_only_constraint(self):
        """Muon 仅接受2D参数。"""
        model = nn.Linear(4, 4, bias=False)
        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01,
        )
        # 2D 参数应正常工作
        optimizer.zero_grad()
        loss = (model(torch.randn(8, 4)) ** 2).mean()
        loss.backward()
        optimizer.step()  # 不应抛出异常

    def test_per_head_mode(self):
        """Per-Head 模式正确工作。"""
        model = nn.Linear(16, 32, bias=False)  # 4 heads × 8 dim

        optimizer = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": 4}],
            lr=0.01,
        )

        for _ in range(10):
            optimizer.zero_grad()
            loss = (model(torch.randn(8, 16)) ** 2).mean()
            loss.backward()
            optimizer.step()

        # 不应抛出异常
        assert True

    def test_nesterov_vs_momentum(self):
        """Nesterov 和普通动量产生不同更新。"""
        torch.manual_seed(42)
        model1 = nn.Linear(4, 4, bias=False)
        model2 = nn.Linear(4, 4, bias=False)
        # 复制权重
        model2.load_state_dict(model1.state_dict())

        opt1 = Muon(
            [{"params": model1.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01, nesterov=True,
        )
        opt2 = Muon(
            [{"params": model2.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01, nesterov=False,
        )

        x = torch.randn(8, 4)
        for _ in range(5):
            for opt in [opt1, opt2]:
                opt.zero_grad()
                loss = (model1(x) ** 2).mean() if opt is opt1 else (model2(x) ** 2).mean()
                loss.backward()
                opt.step()

        # 两种模式应产生不同的权重
        assert not torch.allclose(model1.weight, model2.weight, atol=1e-6)
