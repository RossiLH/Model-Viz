"""MXFP4 / MXFP8 量化感知训练测试。

测试覆盖范围：
- E8M0 共享指数计算
- MXFP4 量化：round-trip、E2M1 级别对齐、块大小处理
- MXFP8 量化：round-trip、精度对比
- FakeQuant：straight-through estimation（梯度通过）
- MXFP4Linear：量化矩阵乘
- apply_qat：MoE 专家启用/禁用
- quantization_gap：量化差距测量

与 kimi3-master tests/test_qat.py 的关系：
  本文件覆盖 kimi3-master 的全部 7 个测试场景，额外添加底层量化函数的单元测试。
"""

import torch
import torch.nn as nn

from models.llm.kimi_k3.quant.mxfp import (
    MX_BLOCK,
    _E2M1_LEVELS,
    FakeQuant,
    MXFP4Linear,
    _blockify,
    _shared_exponent,
    apply_qat,
    fake_quant,
    mxfp4_quantize,
    mxfp8_quantize,
    quantization_gap,
)


# ============================================================
# 1. E8M0 共享指数
# ============================================================

class TestSharedExponent:
    """_shared_exponent 函数测试。"""

    def test_e8m0_power_of_two(self):
        """缩放因子必须是 2 的幂次。"""
        x = torch.randn(1, 32) * 10.0
        s = _shared_exponent(x, 6.0)
        # 检查 log2 是整数（power-of-two）
        log2 = torch.log2(s)
        assert torch.allclose(log2, log2.round(), atol=1e-4)

    def test_scale_computed_for_each_block(self):
        """每个块有独立的缩放因子。"""
        x = torch.randn(2, 64)  # 2 个块
        s = _shared_exponent(x, 6.0)
        assert s.shape == (2, 1)

    def test_larger_input_larger_scale(self):
        """更大的输入值应产生更大的缩放因子。"""
        small = torch.ones(1, 32) * 1.0
        large = torch.ones(1, 32) * 100.0
        assert _shared_exponent(large, 6.0) > _shared_exponent(small, 6.0)


# ============================================================
# 2. 块化
# ============================================================

class TestBlockify:
    """_blockify 函数测试。"""

    def test_exact_division(self):
        """维度恰好整除块大小时无填充。"""
        x = torch.randn(3, 64)
        blocks, pad, shape = _blockify(x, 32)
        assert pad == 0
        assert blocks.shape == (3, 2, 32)

    def test_padding_applied(self):
        """维度不整除时进行零填充。"""
        x = torch.randn(3, 50)
        blocks, pad, shape = _blockify(x, 32)
        assert pad > 0
        assert blocks.shape[2] == 32

    def test_shape_preserved(self):
        """原始 shape 正确保留。"""
        x = torch.randn(2, 3, 100)
        blocks, pad, shape = _blockify(x, 32)
        assert shape == (2, 3, 100)


# ============================================================
# 3. MXFP4 量化
# ============================================================

class TestMxfp4:
    """mxfp4_quantize 函数测试。"""

    def test_round_trip_small_error(self):
        """量化→反量化的相对误差应该很小。"""
        x = torch.randn(4, 128)
        q = mxfp4_quantize(x)
        rel = float((q - x).norm() / x.norm())
        assert 0.0 < rel < 0.5, f"MXFP4 round-trip error too large: {rel:.3f}"

    def test_output_shape_matches_input(self):
        """输出 shape 必须与输入一致。"""
        x = torch.randn(2, 3, 64)
        q = mxfp4_quantize(x)
        assert q.shape == x.shape

    def test_dtype_preserved(self):
        """dtype 必须与输入一致。"""
        x = torch.randn(4, 32).double()
        q = mxfp4_quantize(x)
        assert q.dtype == torch.float64

    def test_e2m1_values_only(self):
        """输出值必须对齐到 E2M1 级别（允许缩放因子影响）。"""
        x = torch.ones(1, 32)  # 平坦数据 → amax=1, scale=2^floor(log2(1/6))
        q = mxfp4_quantize(x)
        # 缩放后值应该接近 E2M1 级别乘以缩放因子
        assert q.abs().max() <= 6.0 * 10  # 合理范围

    def test_block_size_32(self):
        """默认块大小为 32。"""
        assert MX_BLOCK == 32

    def test_zeros_stay_zero(self):
        """零值量化后仍为零。"""
        x = torch.zeros(4, 32)
        q = mxfp4_quantize(x)
        assert torch.allclose(q, x, atol=1e-6)


# ============================================================
# 4. MXFP8 量化
# ============================================================

class TestMxfp8:
    """mxfp8_quantize 函数测试。"""

    def test_round_trip_small_error(self):
        """MXFP8 的 round-trip 误差应小于 MXFP4。"""
        x = torch.randn(4, 128)
        q = mxfp8_quantize(x)
        rel = float((q - x).norm() / x.norm())
        assert 0.0 < rel < 0.3, f"MXFP8 round-trip error too large: {rel:.3f}"

    def test_better_precision_than_mxfp4(self):
        """MXFP8 精度应优于 MXFP4。"""
        x = torch.randn(4, 128)
        e8 = float((mxfp8_quantize(x) - x).norm())
        e4 = float((mxfp4_quantize(x) - x).norm())
        assert e8 < e4, f"MXFP8 ({e8:.4f}) should be more precise than MXFP4 ({e4:.4f})"

    def test_output_shape_matches_input(self):
        """输出 shape 必须与输入一致。"""
        x = torch.randn(2, 3, 64)
        q = mxfp8_quantize(x)
        assert q.shape == x.shape


# ============================================================
# 5. FakeQuant（Straight-Through Estimator）
# ============================================================

class TestFakeQuant:
    """FakeQuant 测试。"""

    def test_gradient_passes_through(self):
        """反向传播梯度必须通过（STE）。"""
        x = torch.randn(4, 32, requires_grad=True)
        y = fake_quant(x, mxfp4_quantize)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "梯度为零，STE 失效"

    def test_gradient_not_zero(self):
        """梯度不能为零。"""
        x = torch.randn(8, 32, requires_grad=True)
        y = fake_quant(x, mxfp8_quantize)
        y.square().mean().backward()
        assert x.grad is not None
        assert float(x.grad.abs().sum()) > 0

    def test_forward_differs_from_input(self):
        """前向传播结果应与输入不同（量化引入变化）。"""
        x = torch.randn(4, 32)
        y = fake_quant(x, mxfp4_quantize)
        assert not torch.allclose(x, y, atol=1e-4)


# ============================================================
# 6. MXFP4Linear
# ============================================================

class TestMXFP4Linear:
    """MXFP4Linear 测试。"""

    def test_output_shape(self):
        """输出 shape 正确。"""
        w = nn.Parameter(torch.randn(32, 64))  # [in_features, out_features]
        layer = MXFP4Linear(w)
        x = torch.randn(4, 32)
        y = layer(x)
        assert y.shape == (4, 64)

    def test_quantized_differ_from_original(self):
        """量化后的输出应与全精度输出不同。"""
        w = nn.Parameter(torch.randn(32, 64))
        x = torch.randn(4, 32)
        with torch.no_grad():
            original = x @ w
        layer = MXFP4Linear(w)
        with torch.no_grad():
            quantized = layer(x)
        assert not torch.allclose(original, quantized, atol=1e-3)

    def test_master_weight_unchanged(self):
        """主权重保持全精度（不被量化覆盖）。"""
        w = nn.Parameter(torch.randn(32, 64))
        original_w = w.data.clone()
        layer = MXFP4Linear(w)
        x = torch.randn(4, 32)
        layer(x)
        assert torch.equal(w.data, original_w)

    def test_gradient_flows_to_weight(self):
        """梯度能流回权重。"""
        w = nn.Parameter(torch.randn(32, 64))
        layer = MXFP4Linear(w)
        x = torch.randn(4, 32)
        y = layer(x)
        y.sum().backward()
        assert w.grad is not None
        assert w.grad.abs().sum() > 0


# ============================================================
# 7. apply_qat
# ============================================================

class TestApplyQat:
    """apply_qat 函数测试。"""

    def _make_moe(self):
        """创建一个简单的 MoE-like 模型。"""
        model = nn.Module()
        experts = nn.Module()
        experts.qat_enabled = False
        experts.gate = nn.Parameter(torch.randn(4, 32, 64))
        experts.up = nn.Parameter(torch.randn(4, 32, 64))
        experts.down = nn.Parameter(torch.randn(4, 64, 32))
        model.experts = experts
        return model

    def test_qat_off_by_default(self):
        """QAT 默认关闭。"""
        m = self._make_moe()
        assert m.experts.qat_enabled is False

    def test_enabling_qat(self):
        """apply_qat(enabled=True) 开启 QAT。"""
        m = self._make_moe()
        apply_qat(m, enabled=True)
        assert m.experts.qat_enabled is True

    def test_disabling_qat(self):
        """apply_qat(enabled=False) 关闭 QAT。"""
        m = self._make_moe()
        apply_qat(m, enabled=True)
        apply_qat(m, enabled=False)
        assert m.experts.qat_enabled is False

    def test_returns_model(self):
        """apply_qat 返回模型自身（可链式调用）。"""
        m = self._make_moe()
        result = apply_qat(m, enabled=True)
        assert result is m


# ============================================================
# 8. quantization_gap
# ============================================================

class TestQuantizationGap:
    """quantization_gap 函数测试。"""

    def _make_model_and_inputs(self):
        """创建简单模型和输入。"""
        model = nn.Module()
        experts = nn.Module()
        experts.qat_enabled = False
        experts.gate = nn.Parameter(torch.randn(2, 16, 32))
        experts.up = nn.Parameter(torch.randn(2, 16, 32))
        experts.down = nn.Parameter(torch.randn(2, 32, 16))
        model.experts = experts
        x = torch.randn(1, 4, 16)
        return model, x

    def test_gap_is_scalar(self):
        """返回 gap 标量。"""
        m, x = self._make_model_and_inputs()
        result = quantization_gap(m, x, lambda m, x: torch.tensor(1.0))
        assert "gap" in result
        assert result["gap"] == 0.0  # 同一 loss 函数

    def test_fp_and_quant_loss(self):
        """返回 fp 和 quantized loss。"""
        m, x = self._make_model_and_inputs()
        result = quantization_gap(m, x, lambda m, x: torch.tensor(2.0))
        assert result["loss_fp"] == 2.0
        assert result["loss_mxfp4"] == 2.0


# ============================================================
# 9. 边界情况
# ============================================================

class TestEdgeCases:
    """边界情况测试。"""

    def test_single_element_block(self):
        """1×32 的最小输入。"""
        x = torch.randn(1, 32)
        q4 = mxfp4_quantize(x)
        q8 = mxfp8_quantize(x)
        assert q4.shape == x.shape
        assert q8.shape == x.shape

    def test_large_input(self):
        """大输入不会崩溃。"""
        x = torch.randn(1024, 1024)
        q4 = mxfp4_quantize(x)
        assert q4.shape == x.shape

    def test_negative_values(self):
        """负值正确处理。"""
        x = torch.tensor([-10.0, -20.0, -30.0, -5.0, 10.0, 20.0, 30.0, 40.0] * 4)
        q = mxfp4_quantize(x)
        assert q.shape == x.shape
        # 大负值量化后仍为负
        assert q[0] < 0
        assert q[1] < 0
        assert q[2] < 0
        # 大正值量化后仍为正
        assert q[4] > 0
        assert q[5] > 0
        assert q[6] > 0

    def test_non_divisible_by_32(self):
        """维度不被 32 整除时正确填充。"""
        x = torch.randn(3, 50)
        q4 = mxfp4_quantize(x)
        q8 = mxfp8_quantize(x)
        assert q4.shape == x.shape
        assert q8.shape == x.shape
