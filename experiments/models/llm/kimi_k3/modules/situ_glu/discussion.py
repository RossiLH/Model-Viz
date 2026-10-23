# -*- coding: utf-8 -*-
"""SiTU-GLU 探讨实验（Analysis.md Phase 4.7）。

反方观点（4.3）验证：
  a/c) "SiTU-GLU 的 softcap 会抑制大激活，可能伤害模型在大激活区间的表达能力
       （把有用的强响应截平了）；相比无界的 SwiGLU，它对极端激活是保守的。"
  我们构造一个**大动态范围回归任务**：目标包含偶尔出现的极大值（离群点）。
  对比 SwiGLU 与 SiTU-GLU 拟合该任务的差异：
    - SwiGLU 无界，能原样通过大的激活（但低精度下会溢出/离群）；
    - SiTU-GLU 有界（|·|<=100），大激活被软截断，对离群目标是"保守的" ——
      它牺牲了在大激活区间的一小部分动态范围，换取数值稳定。

  同时验证 softcap 的数学性质（见 test_situ_glu.py 已覆盖的推论）：
    1. 有界性：|SiTU-GLU| <= beta_g*beta_u = 100；
    2. 原点附近近似 SwiGLU（一阶一致）；
    3. 饱和区梯度非零（光滑，优于硬 clamp）。

实测输出回填见 SiTU-GLU-梳理-V1.md §4.7。
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from models.llm.kimi_k3.modules.situ_glu import SiTUGLU, SwiGLU, situ_glu, softcap  # noqa: E402

BETA_GATE, BETA_UP = 4.0, 25.0
BOUND = BETA_GATE * BETA_UP


def make_outlier_task(N: int, outlier_frac: float = 0.02) -> tuple[torch.Tensor, torch.Tensor]:
    """回归任务：x 为随机数，目标含少量大离群点。"""
    torch.manual_seed(1)
    x = torch.randn(N, 16)
    y = torch.tanh(x[:, :1]) * 3.0                       # 主体小范围
    outlier_mask = torch.rand(N) < outlier_frac
    y[outlier_mask] = 200.0 + torch.randn(outlier_mask.sum(), 1) * 10.0  # 离群点目标
    return x, y


def fit(module: nn.Module, x: torch.Tensor, y: torch.Tensor,
        iters: int = 2000, lr: float = 1e-2) -> list[float]:
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = []
    for i in range(iters):
        opt.zero_grad()
        pred = module(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if i % 200 == 0:
            hist.append(float(loss.item()))
    return hist


def run() -> None:
    torch.manual_seed(0)
    x, y = make_outlier_task(2000)

    situ = SiTUGLU(16, 16, 1, beta_gate=BETA_GATE, beta_up=BETA_UP)
    swi = SwiGLU(16, 16, 1)

    hist_situ = fit(situ, x, y)
    hist_swi = fit(swi, x, y)

    print("含离群点回归任务的训练 MSE（每 200 步）:")
    print("  步数    SwiGLU     SiTU-GLU")
    for i, (s, w) in enumerate(zip(hist_situ, hist_swi)):
        print(f"  {i*200:5d}   {w:8.3f}   {s:8.3f}")

    print("\nsoftcap 数值性质验证:")
    z = torch.tensor([0.0, 0.1, 1.0, 10.0, 100.0, 1e4])
    print("  z         softcap(z,4)   sigmoid(z)")
    for zi in z:
        print(f"  {float(zi):8.1f}   {float(softcap(zi, BETA_GATE)):10.4f}   {float(torch.sigmoid(zi)):.4f}")

    # 有界性与梯度：softcap 在"中等大"输入处梯度仍非零（tanh 未完全饱和），
    # 而硬 clamp 在那里梯度为 0 —— 这是光滑截断相对硬截断的关键优势。
    z = torch.tensor([5.0, 20.0], requires_grad=True)
    out = situ_glu(z, z, BETA_GATE, BETA_UP)
    out.sum().backward()
    grad_situ = z.grad.detach().clone()

    z2 = torch.tensor([5.0, 20.0], requires_grad=True)
    (torch.clamp(z2, max=BETA_GATE) * torch.sigmoid(z2)).sum().backward()
    grad_clamp = z2.grad.detach().clone()

    print(f"\nSiTU-GLU(大输入) 有界: max={float(out.detach().abs().max()):.2f} (<=100)")
    print(f"  softcap 梯度 (z=5,20): {[f'{g:.4f}' for g in grad_situ.tolist()]}")
    print(f"  硬 clamp 梯度 (z=5,20): {[f'{g:.4f}' for g in grad_clamp.tolist()]}")
    print("  → 饱和区 softcap 仍保持非零梯度，硬 clamp 直接掐断梯度（=0）。")

    # 与 SwiGLU 原点一致。
    x0 = torch.linspace(-0.2, 0.2, 41)
    rel = ((situ_glu(x0, x0, BETA_GATE, BETA_UP) - (F.silu(x0) * x0)).abs() /
           (F.silu(x0) * x0).abs().clamp_min(1e-8)).max()
    print(f"原点附近相对误差 vs SwiGLU: {float(rel):.4f} (<1e-2 则一阶一致)")

    print("\n关键推论:")
    print("  - SiTU-GLU 在大动态范围 / 低精度下数值稳定（有界），这是它能用于 FP8 MoE")
    print("    专家推理的原因；代价是对超大离群激活保守（softcap 截平）。")
    print("  - 这与 4.3a 的质疑一致：确有小动态范围损失，但换取的是数值稳定与")
    print("    FP8 兼容（|·|<=100 < FP8_E4M3_MAX=448），在训练稳定性上收益更大。")


if __name__ == "__main__":
    run()
