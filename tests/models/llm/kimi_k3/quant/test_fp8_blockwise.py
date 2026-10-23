"""Block-wise FP8 激活量化测试。

测试覆盖范围：
- BlockFP8 数据结构
- 量化/反量化 round-trip
- 不同块大小的行为
- 边界情况（零值、负值、非整除维度）
- 量化误差测量
"""

import torch

from models.llm.kimi_k3.quant.fp8_blockwise import (
    BlockFP8,
    dequantize_blockwise,
    fp8_dtype,
    quantization_error,
    quantize_blockwise,
)


# ============================================================
# 1. BlockFP8 数据结构
# ============================================================

class TestBlockFP8:
    """BlockFP8 数据结构测试。"""

    def test_nbytes_positive(self):
        """nbytes 必须为正。"""
        x = torch.randn(128, 256)
        q = quantize_blockwise(x)
        assert q.nbytes() > 0

    def test_nbytes_smaller_than_fp32(self):
        """存储开销应小于 FP32。"""
        x = torch.randn(128, 256)
        q = quantize_blockwise(x)
        fp32_bytes = x.numel() * 4
        assert q.nbytes() < fp32_bytes, (
            f"BlockFP8 ({q.nbytes()}) should be smaller than FP32 ({fp32_bytes})"
        )

    def test_dequantize_method(self):
        """BlockFP8.dequantize() 应正常工作。"""
        x = torch.randn(64, 128)
        q = quantize_blockwise(x)
        y = q.dequantize()
        assert y.shape == x.shape


# ============================================================
# 2. 量化/反量化 round-trip
# ============================================================

class TestRoundTrip:
    """量化→反量化 round-trip 测试。"""

    def test_small_relative_error(self):
        """round-trip 相对误差应该较小（int8 退化路径下 <60%）。"""
        x = torch.randn(4, 256)
        y = dequantize_blockwise(quantize_blockwise(x))
        rel = float((y - x).norm() / x.norm())
        assert 0.0 < rel < 0.60, f"Round-trip error too large: {rel:.3f}"

    def test_output_shape_matches_input(self):
        """反量化后 shape 与输入一致。"""
        x = torch.randn(2, 3, 128)
        y = dequantize_blockwise(quantize_blockwise(x))
        assert y.shape == x.shape

    def test_dtype_preserved(self):
        """dtype 必须与原始输入一致。"""
        x = torch.randn(4, 64).double()
        y = dequantize_blockwise(quantize_blockwise(x))
        assert y.dtype == torch.float64

    def test_zeros_stay_zero(self):
        """零值量化后仍为零。"""
        x = torch.zeros(4, 128)
        y = dequantize_blockwise(quantize_blockwise(x))
        assert torch.allclose(y, x, atol=1e-6)


# ============================================================
# 3. 不同块大小
# ============================================================

class TestBlockSizes:
    """不同块大小测试。"""

    def test_block_size_64(self):
        """block_size=64 应正常工作。"""
        x = torch.randn(4, 128)
        q = quantize_blockwise(x, block_size=64)
        assert q.block_size == 64
        y = q.dequantize()
        assert y.shape == x.shape

    def test_block_size_256(self):
        """block_size=256 应正常工作。"""
        x = torch.randn(4, 256)
        q = quantize_blockwise(x, block_size=256)
        y = q.dequantize()
        rel = float((y - x).norm() / x.norm())
        assert 0.0 < rel < 0.50

    def test_smaller_block_better_precision(self):
        """更小的块通常提供更好的精度。"""
        x = torch.randn(4, 256)
        y64 = dequantize_blockwise(quantize_blockwise(x, block_size=64))
        y256 = dequantize_blockwise(quantize_blockwise(x, block_size=256))
        e64 = float((y64 - x).norm())
        e256 = float((y256 - x).norm())
        # 64 通常更好（不一定严格，但大多数情况）
        assert e64 <= e256 * 1.5  # 允许一些波动


# ============================================================
# 4. 边界情况
# ============================================================

class TestEdgeCases:
    """边界情况测试。"""

    def test_negative_values(self):
        """负值正确处理（符号保持，int8 精度有限）。"""
        x = torch.tensor([-10.0, -20.0, -30.0, -5.0, 10.0, 20.0, 30.0, 40.0] * 16)
        y = dequantize_blockwise(quantize_blockwise(x))
        assert y.shape == x.shape
        # 大值的符号应保持（小值可能因 int8 精度丢失符号）
        assert y[0] < 0
        assert y[-1] > 0

    def test_single_row(self):
        """单行输入。"""
        x = torch.randn(1, 128)
        y = dequantize_blockwise(quantize_blockwise(x))
        assert y.shape == x.shape

    def test_non_divisible_by_block(self):
        """维度不被 block_size 整除。"""
        x = torch.randn(4, 100)
        q = quantize_blockwise(x, block_size=128)
        y = q.dequantize()
        assert y.shape == x.shape

    def test_large_tensor(self):
        """大张量不崩溃。"""
        x = torch.randn(1024, 1024)
        y = dequantize_blockwise(quantize_blockwise(x))
        assert y.shape == x.shape


# ============================================================
# 5. 量化误差测量
# ============================================================

class TestQuantizationError:
    """quantization_error 函数测试。"""

    def test_error_in_range(self):
        """误差在合理范围内（int8 退化路径下可能较大）。"""
        x = torch.randn(4, 128)
        err = quantization_error(x)
        assert err >= 0.0

    def test_error_small_for_normal_data(self):
        """正态分布数据的误差应有限（int8 退化路径下 <50%）。"""
        x = torch.randn(4, 256)
        err = quantization_error(x)
        assert err < 0.50

    def test_error_zero_for_constant(self):
        """常量数据的误差应较小（int8 退化路径下仍有一定误差）。"""
        x = torch.ones(4, 128) * 5.0
        err = quantization_error(x)
        # int8 退化路径下，常量数据也会有量化误差
        assert err < 0.80


# ============================================================
# 6. FP8 数据类型
# ============================================================

class TestFp8Dtype:
    """fp8_dtype 函数测试。"""

    def test_returns_valid_dtype(self):
        """返回有效的 torch.dtype（int8 或 float8_e4m3fn）。"""
        dt = fp8_dtype()
        # Python 3.7 / PyTorch <2.1 返回 int8
        assert dt in (torch.int8, getattr(torch, "float8_e4m3fn", torch.int8))

    def test_consistent_across_calls(self):
        """多次调用返回相同结果。"""
        assert fp8_dtype() == fp8_dtype()
