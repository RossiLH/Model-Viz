"""Newton-Schulz 正交化测试。"""
import pytest
import torch

from models.llm.kimi_k3.optim.newton_schulz import match_update_rms, newton_schulz


class TestNewtonSchulz:
    """Newton-Schulz 正交化基本属性测试。"""

    def test_output_shape_matches_input(self):
        """输出 shape 与输入一致。"""
        g = torch.randn(4, 8)
        out = newton_schulz(g)
        assert out.shape == g.shape

    def test_output_dtype_matches_input(self):
        """输出 dtype 与输入一致。"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            g = torch.randn(4, 4).to(dtype)
            out = newton_schulz(g)
            assert out.dtype == dtype

    def test_output_is_finite(self):
        """输出所有元素有限（无 NaN/Inf）。"""
        g = torch.randn(8, 8)
        out = newton_schulz(g)
        assert torch.isfinite(out).all()

    def test_singular_values_near_one(self):
        """正交化后奇异值应接近 1。"""
        g = torch.randn(16, 16)
        out = newton_schulz(g, steps=5)
        s = torch.linalg.svdvals(out)
        # 所有奇异值应在 [0.5, 2.0] 范围内（宽松检查）
        assert s.min() > 0.5
        assert s.max() < 2.0

    def test_condition_number_near_one(self):
        """正交化后条件数（max_svd / min_svd）接近1。"""
        g = torch.randn(32, 16)
        out = newton_schulz(g, steps=5)
        s = torch.linalg.svdvals(out)
        condition_number = s.max() / s.min()
        assert condition_number < 2.0

    def test_scale_invariance(self):
        """缩放输入不改变输出（NS 对缩放不变）。"""
        g = torch.randn(4, 6)
        out1 = newton_schulz(g)
        out2 = newton_schulz(g * 100.0)
        # 输出应接近（方向一致）
        cos_sim = torch.nn.functional.cosine_similarity(
            out1.reshape(1, -1), out2.reshape(1, -1)
        )
        assert cos_sim.item() > 0.99

    def test_batched_operation(self):
        """批处理：前导维度独立处理。"""
        g = torch.randn(3, 4, 8)  # 3 个 4×8 矩阵
        out = newton_schulz(g)
        assert out.shape == (3, 4, 8)
        # 每个批独立正交化
        for i in range(3):
            s = torch.linalg.svdvals(out[i])
            assert s.min() > 0.3

    def test_zero_input(self):
        """零输入返回零输出。"""
        g = torch.zeros(4, 4)
        out = newton_schulz(g)
        assert out.abs().max() < 1e-6

    def test_tall_matrix(self):
        """高矩阵 (m > n) 自动转置处理。"""
        g = torch.randn(16, 4)
        out = newton_schulz(g)
        assert out.shape == (16, 4)
        s = torch.linalg.svdvals(out)
        assert s.min() > 0.3

    def test_steps_parameter(self):
        """5步足够使条件数接近1（高矩阵）。"""
        g = torch.randn(32, 16)  # 高矩阵 (m > n)，典型场景
        out5 = newton_schulz(g, steps=5)
        s = torch.linalg.svdvals(out5)
        # 5步后条件数应 < 2（对高矩阵）
        assert s.max() / s.min() < 2.0


class TestMatchUpdateRMS:
    """Update RMS 缩放测试。"""

    def test_shape_mode_makes_rms_near_one(self):
        """shape 模式使 RMS 接近 1。"""
        g = torch.randn(8, 16)
        ortho = newton_schulz(g)
        scaled = match_update_rms(ortho, mode="shape")
        rms = float(scaled.float().square().mean().sqrt())
        assert abs(rms - 1.0) < 0.2

    def test_reference_rms_mode(self):
        """reference_rms 模式匹配参考 RMS。"""
        ref = torch.randn(4, 4) * 10.0
        ortho = torch.randn(4, 4) * 0.1
        scaled = match_update_rms(ortho, reference=ref, mode="reference_rms")
        ref_rms = float(ref.float().square().mean().sqrt())
        scaled_rms = float(scaled.float().square().mean().sqrt())
        assert abs(ref_rms - scaled_rms) < 0.1

    def test_invalid_mode_raises(self):
        """无效模式抛出异常。"""
        with pytest.raises(ValueError, match="未知模式"):
            match_update_rms(torch.randn(4, 4), mode="invalid")
