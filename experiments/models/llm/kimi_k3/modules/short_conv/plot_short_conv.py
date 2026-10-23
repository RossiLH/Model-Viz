# -*- coding: utf-8 -*-
"""ShortConv 可视化 demo（Analysis.md Phase 4.8 / 5）。

生成三张图：
  1. plot_receptive_field.png：展示 kernel 宽度 vs 延迟重建误差 —— 验证 4.3c
     （局部感受野不足以捕获长程依赖）。
  2. plot_identity_init.png：恒等初始化 vs 随机初始化的收敛轨迹 —— 展示
     恒等初始化不扰动初始衰减动力学。
  3. plot_causality.png：因果性可视化 —— 修改未来 token 不影响过去输出。

数值回填见 ShortConv-梳理-V1.md §4.8/5.3。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.modules.short_conv import ShortConv  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
from discussion import make_delayed_task, fit_conv  # noqa: E402


def plot_receptive_field() -> None:
    torch.manual_seed(0)
    n_signals, T, delay = 3, 200, 8
    x, y = make_delayed_task(T, delay, n_signals)
    widths = [2, 4, 8, 12, 20]
    mses = []
    for w in widths:
        conv = ShortConv(n_signals, kernel_size=w)
        with torch.no_grad():
            conv.weight.normal_(std=0.5)
        mses.append(fit_conv(conv, x, y))

    plt.figure(figsize=(8, 5))
    plt.plot(widths, mses, 'o-', color='#c62828', linewidth=2, markersize=7)
    plt.axvline(delay, color='gray', linestyle='--', label=f'delay={delay}')
    plt.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    plt.xlabel('ShortConv kernel_size')
    plt.ylabel('重建 MSE')
    plt.title('ShortConv 局部感受野 vs 长程依赖重建：kernel 需 >= delay+1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_receptive_field.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_receptive_field.png")
    for w, m in zip(widths, mses):
        print(f"  kernel_size={w:2d}: MSE={m:.4f}")


def plot_identity_init() -> None:
    torch.manual_seed(0)
    n_signals, T, delay = 3, 200, 4  # delay=4，正好让卷积可学习
    x, y = make_delayed_task(T, delay, n_signals)
    dim = n_signals

    def trace(init_mode: str) -> list[float]:
        conv = ShortConv(dim, kernel_size=8)
        with torch.no_grad():
            if init_mode == "random":
                conv.weight.normal_(std=0.5)
            # identity 已由 reset_parameters 设置
        opt = torch.optim.Adam(conv.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        hist = []
        for step_i in range(1500):
            opt.zero_grad()
            pred, _ = conv(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            if step_i % 100 == 0:
                hist.append(float(loss.item()))
        return hist

    ident = trace("identity")
    rnd = trace("random")
    plt.figure(figsize=(8, 5))
    plt.plot(range(0, 1500, 100), ident, 'o-', color='#238636', label='恒等初始化')
    plt.plot(range(0, 1500, 100), rnd, 's-', color='#c62828', label='随机初始化')
    plt.xlabel('迭代')
    plt.ylabel('训练 MSE (log)')
    plt.yscale('log')
    plt.title('ShortConv 初始化：恒等不扰动、随机起步更抖')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_identity_init.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_identity_init.png")


def plot_causality() -> None:
    torch.manual_seed(0)
    conv = ShortConv(8, kernel_size=4)
    with torch.no_grad():
        conv.weight.normal_(std=0.3)
    x = torch.randn(1, 20, 8)
    with torch.no_grad():
        base = conv(x)[0][0, :, 0]  # 只看第 0 通道
        xp = x.clone()
        xp[:, 10:] += 100.0
        after = conv(xp)[0][0, :, 0]
    plt.figure(figsize=(8, 5))
    plt.plot(base.numpy(), 'o-', label='原始输出')
    plt.plot(after.numpy(), 's-', label='未来被扰动后的输出')
    plt.axvline(9, color='gray', linestyle='--', label='扰动起点(第10位)')
    plt.xlabel('时间步 t')
    plt.ylabel('通道 0 输出')
    plt.title('ShortConv 因果性：修改未来不影响过去输出')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_causality.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_causality.png")


if __name__ == "__main__":
    plot_receptive_field()
    plot_identity_init()
    plot_causality()
