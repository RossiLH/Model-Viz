# -*- coding: utf-8 -*-
"""并行机制总览图（向后兼容入口）。

原先的 plot_parallel.py 把四个实验混在一个 main() 里。
现在重构为：每个模块一个独立的 plot_xxx.py，每个 fig_xxx() 返回 (fig, text)。
本文件作为总览入口保留，仍然输出 plot_parallel.png 这一张 2x2 总览图。

详细的各模块独立图请运行：
  - plot_groups.py       — PP x EP x CP x DP 进程网格
  - plot_kcp.py          — KDA Context Parallelism (KCP)
  - plot_moonep.py       — MoonEP 专家并行
  - plot_zero.py         — ZeRO 梯度分片
  - plot_pipeline.py     — 流水线 / 1F1B / AttnRes 缓存 / 激活均衡
  - plot_activation.py   — 激活管理器 / FP8 块量化

运行方式（与旧版一致）：
    python plot_parallel.py
"""
from __future__ import annotations

from plot_utils import add_src_to_path

add_src_to_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop, text as _text,
    ax_title as _at, ax_labels as _al, ax_legend as _leg,
)
from models.llm.kimi_k3.parallel.moonep import plan_redundant_experts
from models.llm.kimi_k3.parallel.pipeline import one_f_one_b_schedule

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))


def _moonep_data(seed=0, n_experts=64, n_ranks=8, tokens_per_rank=512, skew=2.0):
    gen = torch.Generator().manual_seed(seed)
    total = tokens_per_rank * n_ranks
    weights = torch.rand(n_experts, generator=gen).pow(skew) + 1e-6
    probs = weights / weights.sum()
    draw = torch.multinomial(probs, total, replacement=True, generator=gen)
    loads = torch.bincount(draw, minlength=n_experts).tolist()
    experts_per_rank = -(-n_experts // n_ranks)
    plan = plan_redundant_experts(loads, n_ranks, experts_per_rank=experts_per_rank)
    home_loads = [0] * n_ranks
    for e, load in enumerate(loads):
        home = min(e // experts_per_rank, n_ranks - 1)
        home_loads[home] += load
    return home_loads, plan.tokens_per_rank(), plan.max_redundant(), plan.capacity


def fig_overview():
    setup_cjk_font()
    fp = get_font_prop()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=120)

    # --- 左上：MoonEP 负载均衡 ---
    home, moon, mr, cap = _moonep_data()
    x = list(range(len(home)))
    w = 0.35
    axs[0, 0].bar([i - w/2 for i in x], home, width=w,
                  label="naive 主家", color="#e74c3c", alpha=0.8)
    axs[0, 0].bar([i + w/2 for i in x], moon, width=w,
                  label="MoonEP（完美均衡）", color="#2ecc71", alpha=0.8)
    axs[0, 0].axhline(moon[0], color="#f1c40f", linestyle="--",
                      linewidth=1.5, label=f"target = {moon[0]}")
    _at(axs[0, 0], f"MoonEP：复制专家消除热点  max_red={mr} <= ceil(E/R)={cap}")
    _al(axs[0, 0], "rank", "token 数")
    _leg(axs[0, 0])
    axs[0, 0].grid(True, alpha=0.3, axis="y")

    # --- 右上：ZeRO 显存阶梯 ---
    total = 1_000_000 * 4
    world = 8
    labels = ["ZeRO-0\n(无分片)", "ZeRO-1\n(优化器分片)",
              "ZeRO-2\n(梯度分片)", "ZeRO-2+offload\n(梯度移CPU)"]
    values = [total, total, total // world, 0]
    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    bars = axs[0, 1].bar(labels, [v / 1024 for v in values],
                          color=colors, width=0.6, edgecolor="white")
    for bar, val in zip(bars, values):
        axs[0, 1].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(values) / 1024 * 0.02,
                       f"{val/1024:.0f}KB", ha="center", fontsize=8,
                       fontweight="bold", fontproperties=fp)
    _at(axs[0, 1], "ZeRO 显存阶梯：梯度从全量降到 0")
    _al(axs[0, 1], "", "GPU 字节 (KB)")
    axs[0, 1].grid(True, alpha=0.3, axis="y")
    axs[0, 1].tick_params(axis="x", labelsize=8)

    # --- 左下：1F1B warmup 倾斜 ---
    n_stages, n_mb, vps = 4, 16, 2
    warmups = [one_f_one_b_schedule(s, n_stages, n_mb, vps).warmup_steps
               for s in range(n_stages)]
    axs[1, 0].bar(range(n_stages), warmups, color="#9b59b6", width=0.6)
    for i, w in enumerate(warmups):
        axs[1, 0].text(i, w + 0.2, str(w), ha="center",
                       fontsize=9, fontweight="bold", fontproperties=fp)
    _at(axs[1, 0], "1F1B：前段 stage 驻留更多 micro-batch（气泡）")
    _al(axs[1, 0], "stage", "warmup forward 数")
    axs[1, 0].grid(True, alpha=0.3, axis="y")

    # --- 右下：激活策略对比 ---
    from models.llm.kimi_k3.parallel.activation_mgr import Policy, ActivationManager
    x_test = torch.randn(16, 256)
    pol_bytes = []
    for pol in Policy:
        mgr = ActivationManager(default_policy=pol, fp8_block_size=128)
        mgr.save("test", x_test, recompute_fn=lambda: torch.zeros_like(x_test))
        pol_bytes.append(mgr.gpu_bytes())
    labels2 = [p.value for p in Policy]
    colors2 = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#95a5a6"]
    bars2 = axs[1, 1].bar(labels2, [b for b in pol_bytes],
                           color=colors2, width=0.6, edgecolor="white")
    for bar, b in zip(bars2, pol_bytes):
        axs[1, 1].text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() + max(pol_bytes) * 0.02,
                       f"{b}B", ha="center", fontsize=8,
                       fontweight="bold", fontproperties=fp)
    _at(axs[1, 1], "激活管理器：5 种存储策略的 GPU 占用")
    _al(axs[1, 1], "策略", "GPU 字节数")
    axs[1, 1].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "K3 并行机制总览：专家并行 / ZeRO / 流水线 / 激活管理",
        fontsize=14, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "四大并行机制总览图（详细子图见各 plot_xxx.py）",
        f"MoonEP: naive max={max(home)}, MoonEP all={moon[0]} (完美均衡)",
        f"ZeRO: {total} B → 0 B (offload 到 CPU)",
        f"1F1B warmup: stage0={warmups[0]} → stage{n_stages-1}={warmups[-1]} (倾斜)",
        f"激活策略: KEEP={pol_bytes[0]}B → FP8={pol_bytes[1]}B → OFFLOAD=0B",
    )


if __name__ == "__main__":
    import os
    from plot_utils import save
    fig, txt = fig_overview()
    out_path = os.path.join(HERE, "plot_parallel.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)
    print(txt)
