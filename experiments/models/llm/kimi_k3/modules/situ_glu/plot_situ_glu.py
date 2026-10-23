# -*- coding: utf-8 -*-
"""SiTU-GLU 可视化 demo（Analysis.md Phase 4.8 / 5）。

生成三张图：
  1. plot_curve.png：SiTU-GLU vs SwiGLU 的激活曲线（含有界性）。
  2. plot_outlier_train.png：含离群点回归任务的训练曲线 —— 验证 4.3a
     （有界激活对离群目标保守）。
  3. plot_saturation_gradient.png：softcap 饱和区梯度 vs 硬 clamp —— 验证 4.3
     （光滑截断优于硬截断）。

数值回填见 SiTU-GLU-梳理-V1.md §4.8/5.3。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.modules.situ_glu import SiTUGLU, SwiGLU, situ_glu, softcap  # noqa: E402
from discussion import BETA_GATE, BETA_UP, BOUND, make_outlier_task, fit  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def plot_curve() -> None:
    x = torch.linspace(-20, 20, 400)
    situ = situ_glu(x, x, BETA_GATE, BETA_UP).detach()
    swi = (F.silu(x) * x).detach()
    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), swi.numpy(), color='#c62828', label='SwiGLU (无界)')
    plt.plot(x.numpy(), situ.numpy(), color='#238636', label='SiTU-GLU (有界 ≤100)')
    plt.axhline(BOUND, color='gray', linestyle='--', label=f'上界 {BOUND}')
    plt.axhline(-BOUND, color='gray', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('激活值')
    plt.title('SiTU-GLU vs SwiGLU：有界性')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_curve.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_curve.png")


def plot_outlier_train() -> None:
    torch.manual_seed(0)
    x, y = make_outlier_task(2000)
    situ = SiTUGLU(16, 16, 1, beta_gate=BETA_GATE, beta_up=BETA_UP)
    swi = SwiGLU(16, 16, 1)
    hist_situ = fit(situ, x, y)
    hist_swi = fit(swi, x, y)
    steps = list(range(0, 2000, 200))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, hist_swi, 'o-', color='#c62828', label='SwiGLU')
    plt.plot(steps, hist_situ, 's-', color='#238636', label='SiTU-GLU')
    plt.xlabel('迭代')
    plt.ylabel('MSE')
    plt.title('含离群目标回归：SiTU-GLU 对大离群激活保守（loss 略高）')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_outlier_train.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_outlier_train.png")
    print(f"  末步 MSE: SwiGLU={hist_swi[-1]:.3f}, SiTU-GLU={hist_situ[-1]:.3f}")


def plot_saturation_gradient() -> None:
    z = torch.linspace(-40, 40, 400, requires_grad=True)

    # softcap 的梯度。
    out = softcap(z, BETA_GATE).sum()
    out.backward()
    grad_softcap = z.grad.detach().clone()

    # 硬 clamp 的梯度（clamp 在饱和区梯度为 0）。
    z2 = torch.linspace(-40, 40, 400, requires_grad=True)
    (torch.clamp(z2, max=BETA_GATE).sum()).backward()
    grad_clamp = z2.grad.detach().clone()

    zv = z.detach().numpy()
    plt.figure(figsize=(8, 5))
    plt.plot(zv, grad_softcap.numpy(), color='#238636', label='softcap 梯度 (光滑, 饱和区仍>0)')
    plt.plot(zv, grad_clamp.numpy(), color='#c62828', label='硬 clamp 梯度 (饱和区=0)')
    plt.xlabel('z')
    plt.ylabel('d/dz')
    plt.title('softcap 光滑截断 vs 硬 clamp：饱和区梯度')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_saturation_gradient.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_saturation_gradient.png")


if __name__ == "__main__":
    plot_curve()
    plot_outlier_train()
    plot_saturation_gradient()
