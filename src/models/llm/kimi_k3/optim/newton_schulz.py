"""Newton-Schulz 正交化与 Update RMS 缩放。

Newton-Schulz 迭代用于近似计算矩阵的极分解正交因子，是 Muon 优化器的核心算子。
标准 SVD 分解在 GPU 上不高效，Newton-Schulz 仅需矩阵乘法即可在 5 步内收敛。

参考:
    - Jordan et al., 2024, "Muon: An optimizer for hidden layers"
    - Bernstein & Newhouse, 2024, "Old Optimizer, New Norm" (arXiv:2409.20325)
    - Liu et al., 2025, "Muon is Scalable for LLM Training" (arXiv:2502.16982)
"""
from __future__ import annotations

import torch
from torch import Tensor

# Quintic Newton-Schulz 系数 (Jordan et al.)
# 这组系数使迭代在 5 步内收敛到 ||X·X^T - I|| < 1e-6
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


@torch.no_grad()
def newton_schulz(
    g: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """近似计算矩阵 g 的正交极因子。

    给定矩阵 g，找到最近的半正交矩阵 O（满足 O^T O ≈ I），使得
    ||O - g||_F 最小。这是 g 的极分解 g = O · S 中的正交因子。

    支持批处理：前导维度被视为独立矩阵（用于 Per-Head 正交化）。

    Args:
        g: 输入矩阵，shape (..., m, n)，ndim >= 2
        steps: Newton-Schulz 迭代步数，默认 5
        eps: 数值稳定性常数，默认 1e-7

    Returns:
        近似正交矩阵，shape 与输入相同，dtype 与输入一致
    """
    assert g.ndim >= 2, f"输入必须至少 2 维，当前 {g.ndim} 维"
    a, b, c = _NS_COEFFS

    # 提升到 FP32 避免数值下溢（BF16/FP16 精度不够）
    work = g.detach().float().clone()

    # 对于 2D 矩阵，转置优化：高矩阵先转置
    # 对于批处理（ndim > 2），在最后两个维度操作
    if work.ndim == 2:
        transposed = work.shape[0] > work.shape[1]
        if transposed:
            work = work.T
        # 归一化使谱半径 ≤ 1
        work.div_(work.norm().clamp_min(eps))
        for _ in range(steps):
            gram = work @ work.T
            work = a * work + b * (gram @ work) + c * ((gram @ gram) @ work)
        if transposed:
            work = work.T
    else:
        # 批处理版本：对每个矩阵独立处理
        # 归一化使谱半径 ≤ 1
        norm = work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        work = work / norm
        for _ in range(steps):
            gram = work @ work.mT
            work = a * work + b * (gram @ work) + c * ((gram @ gram) @ work)

    return work.to(g.dtype)


@torch.no_grad()
def match_update_rms(
    orthogonal_update: Tensor,
    reference: Tensor | None = None,
    *,
    mode: str = "shape",
    eps: float = 1e-12,
) -> Tensor:
    """使正交化更新的 RMS 独立于矩阵纵横比。

    正交化后的矩阵 RMS 取决于形状：
    - 方阵 (m=n): RMS ≈ 1/sqrt(m)
    - 瘦高阵 (m>>n): RMS ≈ 1/sqrt(m) (更小)

    为使不同形状矩阵的更新幅度一致，需要缩放。

    Args:
        orthogonal_update: 正交化后的更新矩阵，shape (m, n)
        reference: 参考张量（用于 reference_rms 模式），shape 必须匹配
        mode: 缩放模式
            - "shape": 缩放因子 = sqrt(max(m, n))，使单位 RMS
            - "reference_rms": 匹配参考张量的 RMS
        eps: 数值稳定性常数

    Returns:
        缩放后的更新矩阵，shape 与输入相同
    """
    if orthogonal_update.ndim != 2:
        raise ValueError("orthogonal_update 必须是 2D")

    if mode == "shape":
        # 对于半正交矩阵 O (m×n, m≥n), ||O||_F = sqrt(n), RMS = sqrt(n/(m·n)) = 1/sqrt(m)
        # 乘以 sqrt(max(m,n)) 使 RMS ≈ 1（单位 RMS）
        return orthogonal_update * orthogonal_update.new_tensor(
            max(orthogonal_update.shape) ** 0.5
        )

    if mode != "reference_rms":
        raise ValueError(f"未知模式: {mode}")

    if reference is None or reference.shape != orthogonal_update.shape:
        raise ValueError("reference_rms 模式需要匹配的参考张量")

    target = reference.detach().float().square().mean().sqrt()
    current = orthogonal_update.detach().float().square().mean().sqrt()
    if float(current.item()) <= eps:
        return torch.zeros_like(orthogonal_update)
    return orthogonal_update * (target / current).to(orthogonal_update.dtype)
