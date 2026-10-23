# -*- coding: utf-8 -*-
"""PI0.5 可视化：Flow Matching 时间步采样分布 p(tau)=Beta((s-tau)/s; alpha, beta)。

【展示特色】论文时间步采样分布的两个性质：① 强调低时间步（概率质量压在 τ→0，
             即接近干净动作的一侧）；② 排除高时间步（τ ≥ s=0.999 完全不采样，
             因为推理 Euler 积分步长满足 dt > 1-s 时这些 τ 用不到）。
【展示思路】把「真实模块采样出的经验分布」与「理论密度」叠画在同一坐标轴上做
             before/after 对照：若二者重合，证明实现忠实还原了论文公式。
【如何展示】真实跑本项目的 sample_flow_time 采样 20000 个 τ，画直方图（密度归一）
             vs 理论密度 p(tau)=Beta((s-tau)/s; 1.5, 1)/s；再用半对数轴看清
             τ→s 处「被排除」的硬截止（落点严格小于 s=0.999）。

生成两张图：
  1. plot_beta_hist.png —— 直方图 + 理论密度 + s 截止线（线性纵轴）。
  2. plot_beta_ylog.png —— 同一数据的半对数纵轴，看清「排除 τ>s」的截止与尾部。

数据来源：真实跑本项目的 sample_flow_time（torch Beta(1.5,1) → τ=s(1-y)）。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX  # noqa: E402
from models.vla.pi05_faithful.modules.time_sampler import sample_flow_time  # noqa: E402

setup_fonts()

ALPHA, BETA, S = 1.5, 1.0, 0.999


def _sample(n: int = 20000) -> np.ndarray:
    torch.manual_seed(0)
    return sample_flow_time((n,), alpha=ALPHA, beta=BETA, s=S).numpy()


def _pdf(tau: np.ndarray) -> np.ndarray:
    """p(tau) = (1/s) * Beta_pdf((s-tau)/s; alpha, beta)（区间 [0, s) 外为 0）。"""
    y = (S - tau) / S
    pdf = np.zeros_like(tau)
    m = (tau >= 0) & (tau < S)
    y = np.clip((S - tau) / S, 1e-12, 1.0 - 1e-12)   # 数值保护：避免 log0 / log1p(-1)
    logb = math.lgamma(ALPHA) + math.lgamma(BETA) - math.lgamma(ALPHA + BETA)
    pdf[m] = np.exp((ALPHA - 1) * np.log(y[m]) + (BETA - 1) * np.log1p(-y[m]) - logb) / S
    return pdf


def _plot(hist_yscale: str, fname: str) -> None:
    data = _sample()
    grid = np.linspace(0.0, 1.0, 800)
    p = _pdf(grid)

    plt.figure(figsize=(9.5, 5.2))
    plt.hist(data, bins=120, range=(0.0, 1.0), density=True,
             color=C_AUX, alpha=0.55, label="sample_flow_time 采样（20000 次, 真实模块）")
    plt.plot(grid, p, "-", color=C_NEW, lw=2.2,
             label="理论密度 p(tau)=Beta((s-tau)/s; 1.5, 1)/s")
    plt.axvline(S, color=C_STD, lw=2, ls="--",
                label=f"s={S}：tau>=s 完全排除（容纳 ≤1000 积分步）")
    plt.axvspan(S, 1.0, color=C_STD, alpha=0.12)
    plt.xlabel("时间步 tau")
    plt.ylabel("概率密度" + ("（log 轴）" if hist_yscale == "log" else ""))
    plt.title("论文时间步采样：强调低时间步、排除高时间步（s=0.999 截止）")
    if hist_yscale == "log":
        plt.yscale("log")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=120)
    plt.close()
    print("已生成", OUT / fname)
    print(f"  采样 {len(data)} 个：min={data.min():.4f}  max={data.max():.4f}  "
          f"(应 < {S}, 排除 {float(np.sum(data >= S)) / len(data) * 100:.2f}% 的 tau>=s)")


def main() -> None:
    _plot("linear", "plot_beta_hist.png")
    _plot("log", "plot_beta_ylog.png")


if __name__ == "__main__":
    main()