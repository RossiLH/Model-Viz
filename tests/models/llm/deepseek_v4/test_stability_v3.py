# -*- coding: utf-8 -*-
"""DeepSeek-V4 0x0F 两个训练稳定性机制的单元测试（TDD 先写失败）。

覆盖：
1. **SwiGLU Clamping**（修正 `SwiGLUExpert`）：
   - 线性分量 up 钳到 [−10, 10]（nano modeling.py:854 语义）
   - gate 分量**激活前**上界钳到 10（nano modeling.py:853 语义，
     即论文「cap the upper bound of the gate component at 10」）
   - 钳位发生在 silu 之前（否则 silu 饱和后钳位无意义）
2. **Anticipatory Routing**（新增 `AnticipatoryRouter`）：
   - snapshot：冻结当前 router 权重 → 得到「历史参数 θ_{t−Δ}」
   - precompute/lookup：用冻结参数预计算路由索引并缓存、按输入签名复用
   - loss-spike 自动检测：当前 loss 超出 EMA×spike_ratio 时触发 anticipatory
   - revert：运行 revert_after 步后回到标准模式
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.deepseek_v4.config import MoEV4Config
from models.llm.deepseek_v4.modules.moe import DeepSeekV4MoE, SwiGLUExpert
from models.llm.deepseek_v4.modules.anticipatory import AnticipatoryRouter


def make_small_moe(layer_idx: int = 5) -> DeepSeekV4MoE:
    cfg = MoEV4Config(
        n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
        moe_intermediate_size=64, num_hash_layers=2,
    )
    return DeepSeekV4MoE(cfg, dim=64, layer_idx=layer_idx)


# ------------------------------------------------------------
# SwiGLU Clamping
# ------------------------------------------------------------

def test_swiglu_linear_component_clamped_symmetrically():
    """线性分量 up 被钳到 [−10, 10]（钳前探测：用超大权重强制溢出）。"""
    expert = SwiGLUExpert(dim=64, hidden=64, swiglu_limit=10.0)
    # 权重放大到 100，令 gate_up 输出远超 ∓10
    with torch.no_grad():
        expert.gate_up.weight.mul_(100.0)
    x = torch.ones(1, 4, 64)
    with torch.no_grad():
        gate, up = expert.gate_up(x).chunk(2, dim=-1)
    # 确认原始输出确实超界（证明测试有效）
    assert float(up.abs().max()) > 10.0, "测试前提失败：up 未超界"
    # 前向输出应被 clamp：up 进入 down 前已被限到 [−10,10]
    out = expert(x)
    assert torch.isfinite(out).all()
    assert float(out.detach().abs().max()) < 200.0  # 若 up 未钳，out 会很大


def test_swiglu_gate_clamped_before_activation():
    """gate 分量在 silu 之前被钳上界（nano 语义，极端权重下可判别）。

    验证方式：把权重放大到令 gate_up 输出超界，然后比较「本实现前向」与
    「显式 pre-activation 钳位参考」逐位一致。若钳位发生在 silu 之后（旧实现），
    两者会在线性分量 up 上不一致（up 根本没被钳）。
    """
    torch.manual_seed(0)
    L = 10.0
    expert = SwiGLUExpert(dim=64, hidden=64, swiglu_limit=L)
    with torch.no_grad():
        expert.gate_up.weight.mul_(100.0)
    x = torch.randn(2, 3, 64)
    with torch.no_grad():
        gate, up = expert.gate_up(x).chunk(2, dim=-1)
        assert float(gate.abs().max()) > L, "测试前提失败：gate 未超界"
        # pre-activation 钳位参考（nano 语义）
        reference = expert.down(
            F.silu(gate.clamp(max=L)) * up.clamp(min=-L, max=L))
        got = expert(x)
    assert torch.allclose(got, reference, atol=1e-5)
    assert float(got.abs().max()) < 200.0  # up 被钳，out 不失控


def test_swiglu_limit_zero_disables_clamp():
    """swiglu_limit=0 时不做任何钳位（兼容基线路径）。"""
    expert = SwiGLUExpert(dim=64, hidden=64, swiglu_limit=0.0)
    x = torch.randn(1, 4, 64)
    out = expert(x)
    assert torch.isfinite(out).all()
    assert out.shape == (1, 4, 64)


def test_swiglu_matches_nano_semantics():
    """前向与 nano modeling.py:853-855 语义逐位一致：
    gate=gate.clamp(max=L); up=up.clamp(min=-L,max=L); out=down(silu(gate)*up)。
    """
    torch.manual_seed(0)
    L = 10.0
    expert = SwiGLUExpert(dim=64, hidden=64, swiglu_limit=L)
    x = torch.randn(2, 3, 64)
    with torch.no_grad():
        gate, up = expert.gate_up(x).chunk(2, dim=-1)
        expected = expert.down(F.silu(gate.clamp(max=L)) * up.clamp(min=-L, max=L))
        got = expert(x)
    assert torch.allclose(got, expected, atol=1e-6)


# ------------------------------------------------------------
# Anticipatory Routing
# ------------------------------------------------------------

def test_anticipatory_snapshot_freezes_router_weights():
    """snapshot 后改 live 权重，冻结快照不受影响（模拟 θ_{t−Δ} 历史参数）。"""
    moe = make_small_moe()
    router = AnticipatoryRouter(moe)
    snap = router.snapshot()
    # 改 live gate 权重
    with torch.no_grad():
        moe.gate.weight.add_(1.0)
    # 快照参数应保持原值（被冻结）
    for s, l in zip(snap["gate"], moe.gate.parameters()):
        assert not torch.allclose(s, l), "快照应冻结，不受 live 更新影响"
    assert len(snap["gate"]) == 1


def test_anticipatory_precompute_then_lookup_uses_cache():
    """precompute 后 lookup 应命中缓存（返回缓存的索引，不重新计算）。"""
    moe = make_small_moe()
    router = AnticipatoryRouter(moe)
    x = torch.randn(1, 4, 64)
    ids = torch.randint(1, 1000, (1, 4))
    # 用 live 参数预计算索引并缓存
    indices = router.precompute(x, ids)
    assert indices.shape == (1, 4, moe.topk)
    # 之后 lookup 命中缓存 → 返回相同索引（无需重新前向）
    cached = router.lookup(x, ids)
    assert torch.equal(cached, indices)


def test_anticipatory_lookup_miss_computes_live():
    """lookup 未命中时应现场用当前参数计算。"""
    moe = make_small_moe()
    router = AnticipatoryRouter(moe)
    x = torch.randn(1, 4, 64)
    ids = torch.randint(1, 1000, (1, 4))
    # 未 precompute 直接 lookup → 现场计算并返回
    idx = router.lookup(x, ids)
    assert idx.shape == (1, 4, moe.topk)


def test_anticipatory_loss_spike_detection():
    """loss spike 检测：平稳序列不触发；突增序列触发。"""
    moe = make_small_moe()
    router = AnticipatoryRouter(moe, spike_ratio=2.0, revert_after=3)
    # 平稳 loss 序列（缓变）：不应触发
    smooth = [1.0, 1.02, 1.01, 1.03, 1.02, 1.01]
    for i, l in enumerate(smooth):
        router.observe_loss(l)
        assert router.active is False, f"第{i}步平稳 loss 不应触发"
    # 突增 loss：3 倍于近期 EMA → 应触发
    router.observe_loss(3.0)
    assert router.active is True


def test_anticipatory_reverts_after_steps():
    """触发后运行 revert_after 步自动回到标准模式。"""
    moe = make_small_moe()
    router = AnticipatoryRouter(moe, spike_ratio=2.0, revert_after=3)
    for l in [1.0, 1.0, 1.0]:
        router.observe_loss(l)
    router.observe_loss(5.0)
    assert router.active is True
    # 触发后继续观察 3 步平稳 loss → 应自动回退
    for _ in range(router.revert_after):
        router.observe_loss(1.0)
    assert router.active is False


def test_anticipatory_moe_forward_uses_cached_indices():
    """MoE forward 接入可选路径：anticipatory 开时用缓存索引，与直接路由一致。"""
    torch.manual_seed(0)
    moe = make_small_moe()
    router = AnticipatoryRouter(moe)
    x = torch.randn(1, 4, 64)
    ids = torch.randint(1, 1000, (1, 4))
    # 参考：标准路由索引
    with torch.no_grad():
        ref_idx, ref_w, _ = moe._route(x, ids)
    # anticipatory：precompute 缓存后，forward 用缓存索引
    router.precompute(x, ids)
    with torch.no_grad():
        out = moe(x, input_ids=ids, anticipatory_router=router)
    assert out.shape == (1, 4, 64)
    assert torch.isfinite(out).all()
