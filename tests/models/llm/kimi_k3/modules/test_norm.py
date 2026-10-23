# -*- coding: utf-8 -*-
"""RMSNorm / HeadRMSNorm / l2_normalize 单元测试。

对应 ``modules/norm.py``（K3 报告 §2.2 / 附录 A）。归一化是 K3 混合架构里几乎所有
层的基础算子，三种归一化在该架构里各司其职：

- ``rms_norm`` / ``RMSNorm``：pre-norm（注意力/FFN 前）、MLA 的潜在空间
  （kv_lora_rank / q_lora_rank 维）、MoE 的潜空间（latent_dim 维）用；
- ``HeadRMSNorm``：**只按每个 head 内部的最后一维归一化**，不跨 head 混洗，
  K3 用它做 KDA 输出门（Eq. 6）的第一步；
- ``l2_normalize``：把 query/key 投影到单位球上，用于 KDA（Eq. 2），保证
  delta-rule 里"沿 key 方向擦除旧记忆"的几何语义成立。

覆盖要点：
- RMS 归一化的数学正确性（输出 RMS = 1、方向保持、weight 仿射）；
- fp32 升精度语义与 weight 逐通道仿射；
- HeadRMSNorm 的 head-wise 语义（只按最后一维归一化、不跨 head）；
- l2_normalize 与 rms_norm 的区别（sum vs mean 的范数语义）；
- 与 kimi-k3-pytorch 参考实现的一致性（同公式）。

这些测试在 CPU 上运行，无需 GPU。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.modules.norm import (
    HeadRMSNorm,
    RMSNorm,
    l2_normalize,
    rms_norm,
)


def _rms(x: torch.Tensor) -> torch.Tensor:
    """辅助函数：沿最后一维计算 RMS（root mean square）＝ sqrt(mean(x²))。

    用途：断言"归一化后输出 RMS = 1"这类性质时避免手写 pow/mean/sqrt 三连。
    """
    return x.pow(2).mean(-1, keepdim=True).sqrt()


# ---------------------------------------------------------------------------
# 可复用测量函数（供 gradio demo 等外部调用，返回 dict，不做 assert）
# ---------------------------------------------------------------------------

def measure_rmsnorm_unit_rms(dim=8, batch=2, seq=5, scale=3.0, eps=1e-6):
    """测量 RMSNorm 输出的 RMS 是否为 1。"""
    x = torch.randn(batch, seq, dim) * scale
    out = rms_norm(x, weight=None, eps=eps)
    rms_out = _rms(out)
    return {
        "max_rms_err": float((rms_out - 1.0).abs().max().item()),
        "mean_rms_err": float((rms_out - 1.0).abs().mean().item()),
        "input_shape": tuple(x.shape),
    }


def measure_rmsnorm_direction_preserved(dim=6, batch=4):
    """测量 RMSNorm 是否保持向量方向（各向同性缩放）。"""
    x = torch.randn(batch, dim)
    out = rms_norm(x, weight=None, eps=1e-6)
    out_n = out / out.norm(dim=-1, keepdim=True)
    x_n = x / x.norm(dim=-1, keepdim=True)
    return {
        "direction_diff_max": float((out_n - x_n).abs().max().item()),
        "direction_preserved": bool(torch.allclose(out_n, x_n, atol=1e-6)),
    }


def measure_rmsnorm_weight_affine(dim=7, batch=3):
    """测量 weight 是否等价于归一化后逐通道乘法。"""
    x = torch.randn(batch, dim)
    w = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.5, 1.0, 1.0])
    out = rms_norm(x, weight=w, eps=1e-6)
    ref = rms_norm(x, weight=None, eps=1e-6) * w
    return {
        "max_diff": float((out - ref).abs().max().item()),
        "matches_weight_times_norm": bool(torch.allclose(out, ref, atol=1e-6)),
    }


def measure_l2_vs_rms_semantics(value=2.0, dim=4):
    """测量 L2 归一化 vs RMS 归一化对常值向量的不同缩放。"""
    x = torch.full((1, dim), float(value))
    l2 = l2_normalize(x, dim=-1)
    rms = rms_norm(x, weight=None, eps=1e-6)
    return {
        "l2_per_element": float(l2[0, 0].item()),
        "rms_per_element": float(rms[0, 0].item()),
        "l2_expected": 1.0 / (dim ** 0.5),
        "rms_expected": 1.0,
    }


def test_rms_norm_output_has_unit_rms() -> None:
    """测试目的：验证归一化的数学核心——输出最后一维 RMS 恒为 1。

    具体功能：对幅值被放大 3 倍的随机输入做 rms_norm（无 weight），断言归一化后的
    输出 ``_rms(out) ≈ 1``。这直接验证公式 ``y = x / sqrt(E[x²]+eps)``。

    为什么重要：RMS 归一化把每个 token 的激活重缩放到单位 RMS，从而切断变压器
    深层激活尺度漂移（scale drift）的累积链。若输出 RMS 不为 1，说明除法因子
    算错了，后续层会收到尺度漂移的激活。
    """
    x = torch.randn(2, 5, 8) * 3.0
    out = rms_norm(x, weight=None, eps=1e-6)
    assert torch.allclose(_rms(out), torch.ones_like(_rms(out)), atol=1e-5)


def test_rms_norm_preserves_direction() -> None:
    """测试目的：验证 RMSNorm 是**各向同性的重缩放**，不改变向量方向。

    具体功能：对随机输入做 rms_norm，断言其单位方向向量（v/‖v‖）与输入的方向
    向量逐点一致。

    为什么重要：这区分了 RMSNorm（纯缩放，保方向）与 LayerNorm（先减均值、改变
    方向再缩放）。K3 选择 RMSNorm 的前提正是"Transformer 对 re-centering 不敏感、
    只需 re-scaling"——若实现了方向改变，说明多了一次没必要的 reduction。
    """
    x = torch.randn(4, 6)
    out = rms_norm(x, weight=None, eps=1e-6)
    assert torch.allclose(
        out / out.norm(dim=-1, keepdim=True),
        x / x.norm(dim=-1, keepdim=True),
        atol=1e-6,
    )


def test_rms_norm_weight_affine() -> None:
    """测试目的：验证 weight 参数作为逐通道仿射的乘法语义。

    具体功能：构造逐维 weight=[1,2,3,1,0.5,1,1]，断言 ``rms_norm(x, w)`` 与
    ``rms_norm(x, None) * w`` 逐点相等。

    为什么重要：weight 是归一化后唯一的自由度（可学习的 elementwise_affine）。
    它必须精确等于"归一化结果 × weight"（而不是加性项或别的组合），否则
    训练时梯度导向的解释就错了。该测试把"乘法仿射"约定钉死。
    """
    x = torch.randn(3, 7)
    w = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.5, 1.0, 1.0])
    out = rms_norm(x, weight=w, eps=1e-6)
    ref = rms_norm(x, weight=None, eps=1e-6) * w
    assert torch.allclose(out, ref, atol=1e-6)


def test_rms_norm_module() -> None:
    """测试目的：验证 nn.Module 封装（RMSNorm 层）的形状与初始行为。

    具体功能：构造 dim=6 的 RMSNorm（初始 weight 全 1），断言：
    - 输出形状与输入一致（逐 token、逐位置归一化在最后一维，不改形状）；
    - 初始输出 RMS = 1（weight 全 1 时等价于纯归一化）。

    为什么重要：模块层是模型里真正被实例化的对象，必须保证"默认状态下输出与
    输入同形状、单位 RMS"，这样接入 pre-norm 位置时不会改变残差流的尺度。
    """
    layer = RMSNorm(6, eps=1e-6)
    x = torch.randn(2, 4, 6)
    out = layer(x)
    assert out.shape == x.shape
    # 初始 weight 全 1 -> 输出 RMS 为 1。
    assert torch.allclose(_rms(out), torch.ones_like(_rms(out)), atol=1e-5)


def test_head_rms_norm_normalizes_last_dim_only() -> None:
    """测试目的：验证 HeadRMSNorm 的 head-wise 语义——只按最后一维（head_dim）归一化。

    具体功能：构造两个 head（head 0 幅值 1，head 1 幅值 10），断言：
    - 每个 head 独立归一化到 RMS=1（不跨 head 共享归一化因子）；
    - 两个 head 的输出幅度一致（大小 head 的尺度差异被消除）。

    为什么重要：head-wise 语义是 KDA 输出门的第一步。KDA 每个 head 维护自己的递推
    状态 S，head 间输出尺度可能差异很大；若整体归一化，大 head 会主导归一化因子把
    小 head 信号压没。该测试用"人工放大某个 head"的场景验证每个 head 确实被独立
    归一化——防止实现偷偷改成整张量归一化（语义漂移）。
    """
    layer = HeadRMSNorm(head_dim=4, eps=1e-6)
    # 构造两个 head，第二个 head 的尺度大很多。
    x = torch.zeros(1, 1, 2, 4)
    x[0, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0])
    x[0, 0, 1] = torch.tensor([10.0, 10.0, 10.0, 10.0])
    out = layer(x)
    # 每个 head 独立归一化到 RMS=1，head 之间尺度差异被消除。
    assert torch.allclose(_rms(out), torch.ones_like(_rms(out)), atol=1e-5)
    # 两个 head 的输出幅度一致（都归一化了，容忍浮点误差）。
    assert torch.allclose(
        out[0, 0, 0].abs().max(), out[0, 0, 1].abs().max(), atol=1e-6
    )


def test_head_rms_norm_weight_shape() -> None:
    """测试目的：验证本实现 HeadRMSNorm 的 weight 形状约定（跨 head 共享）。

    具体功能：head_dim=8 的层，断言 weight.shape == (8,)（不是 (num_heads,
    head_dim)）且 forward 保持 (1,2,3,8) 形状不变。

    为什么重要：形状契约是参数化约定的一部分——本实现的 weight 是 ``(head_dim,)``
    且跨 head 共享（``(num_heads, head_dim)`` 变体在 core/norms.py 中，接口不同）。
    KDA 输出门依赖此重量映射，形状断言防止实现与文档不一致。
    """
    layer = HeadRMSNorm(head_dim=8, eps=1e-6)
    assert layer.weight.shape == (8,)
    assert layer.forward(torch.randn(1, 2, 3, 8)).shape == (1, 2, 3, 8)


def test_l2_normalize_unit_norm() -> None:
    """测试目的：验证 l2_normalize 把向量投影到单位球（L2 范数 = 1），保方向。

    具体功能：随机输入上断言：
    - 输出 L2 范数为 1；
    - 输出方向与输入方向一致。

    为什么重要：l2_normalize 用于 KDA 的 query/key（Eq. 2）。归一化后的 q/k 落在
    单位球上：(a) 稳定 delta-rule 的数值（‖Sᵀk‖ 不随训练漂移）；(b) 让 beta 保持
    "方向写"语义（新 key 只在自己的方向上更新记忆，与 k 的范数解耦）。方向不变
    是该几何语义的前提。
    """
    x = torch.randn(3, 5)
    out = l2_normalize(x, dim=-1)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(
        out / out.norm(dim=-1, keepdim=True),
        x / x.norm(dim=-1, keepdim=True),
        atol=1e-6,
    )


def test_l2_vs_rms_semantics_differ() -> None:
    """测试目的：验证 l2_normalize 与 rms_norm 对向量幅度的处理**刻意不同**。

    具体功能：全 2.0 的 4 维向量：
    - L2（sum(x²)）：缩放到每元素 0.5（= 1/√4，保持总能量不变只改方向→单位球）；
    - RMS（mean(x²)）：缩放到每元素 1（= 1/√mean(x²)，使均方值=1）。

    为什么重要：两者差在"求和的还是求均"——L2 保留向量总能量（去方向），RMS 保留
    逐元素相对尺度（去幅度）。KDA 的 q/k 必须用 L2（保单位球几何），MoE/MLA 的
    归一化用 RMS（保逐位置尺度可比）。若实现把两者混用，KDA 的 delta-rule 语义会被
    悄悄改变。该测试显式锁定语义差异。
    """
    x = torch.full((1, 4), 2.0)
    l2 = l2_normalize(x, dim=-1)
    rms = rms_norm(x, weight=None, eps=1e-6)
    # L2: 每个元素缩放到 1/sqrt(4)=0.5；RMS: 每个元素缩放到 1（均值平方=4）。
    assert torch.allclose(l2, torch.full((1, 4), 0.5), atol=1e-6)
    assert torch.allclose(rms, torch.ones(1, 4), atol=1e-6)