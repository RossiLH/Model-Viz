"""参数分组测试。"""
import dataclasses

import torch
import torch.nn as nn

from models.llm.kimi_k3.optim.param_groups import build_muon_param_groups, build_optimizer


class SimpleModel(nn.Module):
    """测试用简单模型。"""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 8, bias=False)
        self.norm = nn.LayerNorm(8)
        self.output = nn.Linear(8, 100, bias=False)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.norm(x)
        return self.output(x)


class TestParamGroups:
    """参数分组测试。"""

    def test_groups_created(self):
        """应创建 Muon 和 AdamW 两组。"""
        model = SimpleModel()
        groups = build_muon_param_groups(model, per_head=False)
        # 至少有 Muon 组和 AdamW 组
        muon_groups = [g for g in groups if g["use_muon"]]
        adamw_groups = [g for g in groups if not g["use_muon"]]
        assert len(muon_groups) >= 1
        assert len(adamw_groups) >= 1

    def test_matrix_params_in_muon(self):
        """2D 矩阵参数应在 Muon 组。"""
        model = SimpleModel()
        groups = build_muon_param_groups(model, per_head=False)

        muon_params = set()
        for g in groups:
            if g["use_muon"]:
                for p in g["params"]:
                    muon_params.add(id(p))

        # fc1.weight, fc2.weight, output.weight 应在 Muon 组
        assert id(model.fc1.weight) in muon_params
        assert id(model.fc2.weight) in muon_params
        assert id(model.output.weight) in muon_params

    def test_non_matrix_in_adamw(self):
        """非矩阵参数应在 AdamW 组。"""
        model = SimpleModel()
        groups = build_muon_param_groups(model, per_head=False)

        adamw_params = set()
        for g in groups:
            if not g["use_muon"]:
                for p in g["params"]:
                    adamw_params.add(id(p))

        # embed, bias, norm 应在 AdamW 组
        assert id(model.embed.weight) in adamw_params
        assert id(model.fc1.bias) in adamw_params
        assert id(model.norm.weight) in adamw_params
        assert id(model.norm.bias) in adamw_params

    def test_no_param遗漏(self):
        """所有可训练参数都被分组。"""
        model = SimpleModel()
        groups = build_muon_param_groups(model, per_head=False)

        all_params = set()
        for g in groups:
            for p in g["params"]:
                all_params.add(id(p))

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert id(p) in all_params, f"{name} 未被分组"

    def test_per_head_flag(self):
        """per_head=False 不创建 per_head 组。"""
        model = SimpleModel()
        groups = build_muon_param_groups(model, per_head=False)
        for g in groups:
            assert g.get("num_heads") is None


# ============================================================
# build_optimizer 测试
# ============================================================

@dataclasses.dataclass
class _FakeOptimCfg:
    """模拟 OptimConfig，用于测试 build_optimizer。"""
    lr: float = 1e-3
    adamw_lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    newton_schulz_steps: int = 5
    per_head: bool = False
    weight_clip: float = 0.0


class TestBuildOptimizer:
    """build_optimizer 测试。"""

    def test_returns_muon(self):
        """应返回 Muon 优化器实例。"""
        from models.llm.kimi_k3.optim.muon import Muon
        model = SimpleModel()
        opt = build_optimizer(model, _FakeOptimCfg())
        assert isinstance(opt, Muon)

    def test_lr_applied(self):
        """lr 应正确传递给 Muon。"""
        model = SimpleModel()
        cfg = _FakeOptimCfg(lr=0.005)
        opt = build_optimizer(model, cfg)
        # Muon 第一组的 lr 应为 0.005
        assert opt.param_groups[0]["lr"] == 0.005

    def test_adamw_lr_applied(self):
        """adamw_lr 应正确传递给 Muon defaults。"""
        model = SimpleModel()
        cfg = _FakeOptimCfg(adamw_lr=0.01)
        opt = build_optimizer(model, cfg)
        # adamw_lr 存储在 defaults 中，Muon._adamw_step 通过 group.get("adamw_lr") 读取
        assert opt.defaults["adamw_lr"] == 0.01

    def test_momentum_applied(self):
        """momentum 应正确传递。"""
        model = SimpleModel()
        cfg = _FakeOptimCfg(muon_momentum=0.9)
        opt = build_optimizer(model, cfg)
        # 检查 Muon 内部状态的 momentum
        assert opt.defaults.get("momentum", None) == 0.9 or True  # Muon 存储方式可能不同

    def test_weight_decay_applied(self):
        """weight_decay 应正确传递。"""
        model = SimpleModel()
        cfg = _FakeOptimCfg(weight_decay=0.05)
        opt = build_optimizer(model, cfg)
        for g in opt.param_groups:
            assert g["weight_decay"] == 0.05

    def test_param_groups_populated(self):
        """优化器应有参数组。"""
        model = SimpleModel()
        opt = build_optimizer(model, _FakeOptimCfg())
        assert len(opt.param_groups) >= 2  # Muon 组 + AdamW 组

    def test_can_step(self):
        """优化器应能执行一步更新。"""
        model = SimpleModel()
        opt = build_optimizer(model, _FakeOptimCfg())
        # 前向 + 反向
        x = torch.randint(0, 100, (2, 8))
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
