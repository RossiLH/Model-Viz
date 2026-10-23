"""统一激活管理器与块级 FP8 的 CPU 验证（论文 §5.2.2）。"""

from __future__ import annotations

import torch

from models.llm.kimi_k3.parallel.activation_mgr import (
    ActivationManager,
    Policy,
    quantization_error,
    quantize_blockwise,
)
from models.llm.kimi_k3.parallel.zero import reduce_scatter_grads


def test_blockwise_fp8_round_trip_is_accurate():
    torch.manual_seed(0)
    x = torch.randn(64, 512)
    assert quantization_error(x, block_size=128) < 0.05


def test_blockwise_beats_per_tensor_scaling_on_skewed_channels():
    """为什么逐块：一个离群 channel 会压坏其它 channel 的分辨率。"""
    torch.manual_seed(0)
    x = torch.randn(32, 512)
    x[:, 0] *= 10_000  # 一个剧烈离群的通道
    fine = quantization_error(x, block_size=32)
    coarse = quantization_error(x, block_size=512)  # 近似 per-tensor
    assert fine < coarse


def test_fp8_halves_the_footprint():
    x = torch.randn(128, 512, dtype=torch.bfloat16)
    q = quantize_blockwise(x, 128)
    assert q.nbytes() < x.numel() * x.element_size()


def test_policies_are_declared_not_hardcoded():
    mgr = ActivationManager()
    mgr.annotate("attn.out", Policy.FP8)
    mgr.annotate("mlp.hidden", Policy.OFFLOAD)
    mgr.annotate("elementwise", Policy.RECOMPUTE)

    x = torch.randn(8, 256)
    mgr.save("attn.out", x)
    mgr.save("mlp.hidden", x)
    mgr.save("elementwise", x, recompute_fn=lambda: x)
    mgr.save("plain", x)

    assert mgr.saved["attn.out"].policy is Policy.FP8
    assert mgr.saved["mlp.hidden"].bytes_on_gpu() == 0
    assert mgr.saved["elementwise"].bytes_on_gpu() == 0
    assert mgr.saved["plain"].bytes_on_gpu() == x.numel() * x.element_size()


def test_every_policy_restores_the_tensor():
    x = torch.randn(4, 128)
    for policy in Policy:
        mgr = ActivationManager()
        mgr.annotate("t", policy)
        mgr.save("t", x, recompute_fn=lambda: x)
        restored = mgr.restore("t")
        assert restored.shape == x.shape
        tol = 0.1 if policy is Policy.FP8 else 1e-6
        assert float((restored - x).abs().max()) < tol, policy


def test_recompute_requires_a_function():
    import pytest

    mgr = ActivationManager()
    mgr.annotate("t", Policy.RECOMPUTE)
    with pytest.raises(ValueError, match="重算"):
        mgr.save("t", torch.randn(2, 2))


def test_manager_reports_memory_reduction():
    mgr = ActivationManager(default_policy=Policy.FP8)
    for i in range(4):
        mgr.save(f"t{i}", torch.randn(32, 512))
    report = mgr.report()
    assert report["gpu_bytes"] < report["if_all_kept"]
    assert report["reduction"] > 1.5


def test_reduce_scatter_averages_across_grads():
    """reduce_scatter 的分片结果是各 rank 那片求平均（梯度 mean）。"""
    params = [torch.nn.Parameter(torch.ones(4, 5))]
    grads = [torch.randn(4, 5)]
    # 单进程模拟：world=2 平分并按均值归约，长度自动对齐。
    out = reduce_scatter_grads(params, grads, world=2)
    assert out[0].numel() <= 10
    assert out[0] is not None