# -*- coding: utf-8 -*-
"""有界 vs 无界衰减的数值稳定性图 —— 回答 Q4/Q5/Q6（对应 KDA 文档 §4.2.7 / §4.2.5 / §4.2.4）。

核心问题：Kimi Linear 用无界 softplus g = -exp(A)*softplus(z)，累积衰减乘积下溢、
其倒数（K/Gamma 的重缩放）上溢，迫使对角 tile 走显式 position-pair 路径；
K3 改用有界 g = g_min*sigmoid(exp(A)z)，把 log-decay 锁在 (g_min, 0)，从而
每个因果 tile（含对角）都能用稠密 Tensor-Core matmul 计算。

本脚本用真实 kda_chunkwise 跑四种门控，在同一张图里展示：
  1. 1/gamma 上界 vs dtype 上限（对数坐标）：有界版是"数据无关"的硬上界 exp(-g_min*tile)，
     无界版随 z 无限制增长。
  2. 实测：tile=16（论文 tile）下，无界 softplus 在 z 偏移大时 1/gamma -> inf、输出 NaN；
     有界 g_min=-5 永远安全。
  3. safe_chunk_size vs g_min：g_min 越负，允许的 chunk 越小。
  4. 4.2.7 的失败 case：unbounded, chunk=64 -> NaN。
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "src")
sys.path.insert(0, HERE)

import torch

# 必须在 pyplot 真正绘图前配置中文字体，否则中文渲染成方框。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from models.llm.kimi_k3.modules.kda_ref import kda_chunkwise, safe_chunk_size, max_reciprocal_decay


def run_cases(b=1, h=2, t=64, dk=16, dv=16, *, chunk=16, seed=0):
    """用真实 kda_chunkwise 跑有界/无界各档门控，返回 (名称, g.min, 1/gamma max, out finite)。"""
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk), dim=-1)
    v = torch.randn(b, h, t, dv)
    beta = torch.sigmoid(torch.randn(b, h, t))
    z = torch.randn(b, h, t, dk)

    results = []
    for name, g in [
        ("bounded g_min*sigmoid (K3)", -5.0 * torch.sigmoid(z)),
        ("unbounded softplus z~N(0,1)", -torch.nn.functional.softplus(z)),
        ("unbounded softplus z+3", -torch.nn.functional.softplus(z + 3.0)),
        ("unbounded softplus z+6", -torch.nn.functional.softplus(z + 6.0)),
        ("unbounded softplus z+9", -torch.nn.functional.softplus(z + 9.0)),
    ]:
        o, _ = kda_chunkwise(q, k, v, g, beta, chunk_size=chunk, dtype=torch.float32)
        gc = g.view(b, h, t // chunk, chunk, dk)
        inv = (-gc.cumsum(3)).exp()
        results.append((name, g.min().item(), inv.max().item(), bool(torch.isfinite(o).all()), int(torch.isnan(o).sum())))
    return results


def main():
    torch.set_num_threads(2)
    font = setup_cjk_font()
    print("cjk font:", font)

    log_fmt = FuncFormatter(lambda y, _p: "1e%d" % round(math.log10(y)) if y > 0 else "0")

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # ---- 左上：1/gamma 上界 vs dtype 上限（数据无关的硬上界）----
    g_min = -5.0
    tiles = [1, 2, 4, 8, 16, 32, 64]
    bf16_max = 3.3895e38
    ax[0, 0].loglog(tiles, [max_reciprocal_decay(g_min, t) for t in tiles], marker="o", label=f"有界上界 exp(-g_min*tile), g_min={g_min}")
    ax[0, 0].axhline(bf16_max, ls="--", color="red", lw=1, label="BF16 上限 3.39e38")
    ax[0, 0].axhline(65504, ls=":", color="orange", lw=1, label="FP16 上限 65504")
    ax[0, 0].axvline(16, ls="-.", color="gray", lw=1, label="论文 tile=16")
    ax[0, 0].set_xlabel("tile (chunk) 大小")
    ax[0, 0].set_ylabel("1/gamma 上界")
    ax[0, 0].xaxis.set_major_formatter(log_fmt)
    ax[0, 0].yaxis.set_major_formatter(log_fmt)
    ax[0, 0].set_title("有界版: 1/gamma 有数据无关硬上界，tile=16 在 BF16 内")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, which="both")

    # ---- 右上：实测 1/gamma max（tile=16），无界版溢出 ----
    results = run_cases(chunk=16)
    names = [r[0] for r in results]
    inv_max = [r[2] for r in results]
    finite = [r[3] for r in results]
    xpos = range(len(names))
    ax[0, 1].bar(xpos, inv_max, color=["C0" if f else "red" for f in finite])
    ax[0, 1].set_yscale("log")
    ax[0, 1].yaxis.set_major_formatter(log_fmt)
    ax[0, 1].set_xticks(list(xpos))
    ax[0, 1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[0, 1].axhline(3.3895e38, ls="--", color="red", lw=1)
    ax[0, 1].text(0, 3.3895e38, "BF16 max", color="red", fontsize=8, va="bottom")
    for i, (nm, mx, fn) in enumerate(zip(names, inv_max, finite)):
        # 溢出时 mx=inf，直接标 NaN 不放文本坐标（避免 posx/posy 非有限告警）。
        label = "NaN" if (not fn) else "%.1e" % mx
        posy = 1e38 if (not fn) else mx * 1.2
        ax[0, 1].text(i, posy, label, ha="center", va="bottom", fontsize=7, color=("black" if fn else "red"))
    ax[0, 1].set_title("实测 tile=16: 无界 softplus 溢出 -> NaN (红柱)")
    ax[0, 1].grid(alpha=0.3, which="both")

    # ---- 左下：safe_chunk_size vs g_min ----
    gms = [-3.0, -4.0, -5.0, -6.0, -8.0]
    for dt, nm in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        ax[1, 0].plot(gms, [safe_chunk_size(gm, dt) for gm in gms], marker="o", label=nm)
    ax[1, 0].set_xlabel("g_min")
    ax[1, 0].set_ylabel("safe_chunk_size")
    ax[1, 0].set_xticks(gms)
    ax[1, 0].set_title("safe_chunk_size vs g_min: g_min 越负, 允许的 chunk 越小")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # ---- 右下：4.2.7 失败 case —— 无界衰减在 chunk=64 溢出 ----
    res64 = run_cases(chunk=64)
    names64 = [r[0] for r in res64]
    inv64 = [r[2] for r in res64]
    fin64 = [r[3] for r in res64]
    ax[1, 1].bar(range(len(names64)), inv64, color=["C0" if f else "red" for f in fin64])
    ax[1, 1].set_yscale("log")
    ax[1, 1].yaxis.set_major_formatter(log_fmt)
    ax[1, 1].set_xticks(range(len(names64)))
    ax[1, 1].set_xticklabels(names64, rotation=20, ha="right", fontsize=8)
    for i, (nm, mx, fn) in enumerate(zip(names64, inv64, fin64)):
        label = "NaN" if (not fn) else "%.1e" % mx
        posy = 1e38 if (not fn) else mx * 1.2
        ax[1, 1].text(i, posy, label, ha="center", va="bottom", fontsize=7, color=("black" if fn else "red"))
    ax[1, 1].set_title("4.2.7 失败 case: chunk=64 无界/超界衰减 -> NaN")
    ax[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("有界 vs 无界衰减：为什么 K3 必须 g_min*sigmoid 才能稠密 Tensor-Core", fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "plot_decay_stability.png")
    plt.savefig(out, dpi=130)
    print("saved", out)
    for r in results:
        print(r)
    for r in res64:
        print(r)


if __name__ == "__main__":
    main()
