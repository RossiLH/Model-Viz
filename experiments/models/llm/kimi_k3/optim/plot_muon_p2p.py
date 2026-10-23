# -*- coding: utf-8 -*-
"""P2P Muon 可视化：所有权映射、分片、内存对比。

对应 tests/models/llm/kimi_k3/optim/test_muon_p2p.py。
单进程模式下 P2P 退化为本地 clone/copy，等价于 all-gather 语义。
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
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.optim.muon_p2p import (
    shard_ranges, OwnershipMap, P2PMuonOrthogonalizer,
)


def fig_shard_ranges(world=4, numel=100):
    setup_cjk_font()
    fp = get_font_prop()
    rngs = shard_ranges(numel, world)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    colors = plt.cm.Set2(np.linspace(0, 1, world))
    for i, (lo, hi) in enumerate(rngs):
        ax1.barh(world - 1 - i, hi - lo, left=lo, color=colors[i],
                 height=0.6, label=f"rank {i}: [{lo}, {hi})")
        ax1.text(lo + (hi - lo) / 2, world - 1 - i,
                 f"{hi - lo}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white",
                 fontproperties=fp)
    ax1.set_yticks(range(world))
    ax1.set_yticklabels([f"rank {world - 1 - i}" for i in range(world)],
                        fontproperties=fp)
    _at(ax1, f"分片分布 ({numel} 元素, {world} 个 rank)")
    _al(ax1, "元素索引", "")
    _leg(ax1, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="x")

    per_shard = [hi - lo for lo, hi in rngs]
    ax2.bar(range(world), per_shard, color=colors, width=0.6)
    ax2.axhline(y=numel / world, color="red", ls="--", alpha=0.6,
                label=f"平均 = {numel/world:.1f}")
    ax2.set_xticks(range(world))
    ax2.set_xticklabels([f"rank {i}" for i in range(world)],
                        fontproperties=fp, rotation=15, fontsize=8)
    _at(ax2, "各 rank 分片大小")
    _al(ax2, "", "元素数")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for i, s in enumerate(per_shard):
        ax2.text(i, s + 0.5, str(s), ha="center", va="bottom",
                 fontsize=9, fontweight="bold", fontproperties=fp)

    fig.tight_layout()

    max_diff = max(per_shard) - min(per_shard)
    return fig, _text(
        f"总元素数: {numel}, world size: {world}",
        f"每 rank 元素数: {per_shard}",
        f"最大/最小差异: {max_diff} (向上取整分块，末段可能更短)",
        "每个 rank 的分片是连续的，便于 coalesced memory access。",
    )


def fig_ownership_map(n_params=12, world=4):
    setup_cjk_font()
    fp = get_font_prop()
    om = OwnershipMap.round_robin(n_params, world, rank=0)
    mine = om.mine()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    colors = plt.cm.Set2(np.linspace(0, 1, world))
    for i in range(n_params):
        owner = om.owners[i]
        ax1.bar(i, 1, color=colors[owner], width=0.8)
        ax1.text(i, 0.5, f"P{i}", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white",
                 fontproperties=fp)
    ax1.set_yticks([])
    ax1.set_xticks(range(n_params))
    ax1.set_xticklabels([f"P{i}" for i in range(n_params)],
                        fontproperties=fp, fontsize=8)
    _at(ax1, f"Round-robin 所有权映射 (n_params={n_params}, world={world})")
    _al(ax1, "参数索引", "")
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)

    counts = [0] * world
    for o in om.owners:
        counts[o] += 1
    bars = ax2.bar(range(world), counts, color=colors, width=0.6)
    ax2.axhline(y=n_params / world, color="red", ls="--", alpha=0.6,
                label=f"平均 = {n_params/world:.1f}")
    ax2.set_xticks(range(world))
    ax2.set_xticklabels([f"rank {i}" for i in range(world)], fontproperties=fp)
    for bar, c in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(c), ha="center", va="bottom",
                 fontsize=10, fontweight="bold", fontproperties=fp)
    _at(ax2, "各 rank 拥有的参数数 (负载均衡)")
    _al(ax2, "", "参数数")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    return fig, _text(
        f"参数 i 归 rank (i % world) 所有",
        f"rank 0 拥有参数: {mine}",
        f"各 rank 参数数: {counts}",
        f"负载差: {max(counts) - min(counts)} (保证 <= 1)",
        "Round-robin 分配：确定性、均匀、无需通信协商。",
    )


def fig_p2p_memory(n_params=64, world_range=(1, 64), param_shape=(128, 64)):
    setup_cjk_font()
    fp = get_font_prop()

    worlds = list(range(world_range[0], world_range[1] + 1))
    p2p_memory = []
    ag_memory = []
    m, n = param_shape
    per_param = m * n
    total_params = n_params * per_param

    for w in worlds:
        ag_memory.append(n_params * per_param)
        owned = (n_params + w - 1) // w
        p2p_memory.append(owned * per_param)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")

    ax1.plot(worlds, ag_memory, "s-", color="#e74c3c", lw=2.5,
             ms=7, label="All-Gather (每个 rank 全量)")
    ax1.plot(worlds, p2p_memory, "o-", color="#2ecc71", lw=2.5,
             ms=7, label="P2P (只组装自己拥有的)")
    _at(ax1, "峰值内存对比 (元素数)")
    _al(ax1, "world size", "峰值缓冲元素数")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ratios = [a / p if p > 0 else 1 for a, p in zip(ag_memory, p2p_memory)]
    ax2.plot(worlds, ratios, "D-", color="#9b59b6", lw=2.5, ms=7)
    ax2.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="1:1 (无节省)")
    # 理想对角线：world <= n_params 时节省倍数 == world
    ideal_end = min(worlds[-1], n_params)
    ideal_x = list(range(worlds[0], ideal_end + 1))
    ideal_y = [float(w) for w in ideal_x]
    ax2.plot(ideal_x, ideal_y, ":", color="#7f8c8d", lw=1.5, alpha=0.8,
             label="理想线性 (倍数 = world)")
    _at(ax2, "内存节省倍数 (All-Gather / P2P)")
    _al(ax2, "world size", "节省倍数")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"P2P vs All-Gather 内存对比 ({n_params} 个 {m}x{n} 参数)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        f"总参数量: {total_params:,} 元素",
        f"world={worlds[-1]} 时，All-Gather 需要 {ag_memory[-1]:,} 元素/每个 rank",
        f"world={worlds[-1]} 时，P2P 仅需 {p2p_memory[-1]:,} 元素/每个 rank",
        f"理论节省约 {ag_memory[-1]/max(1,p2p_memory[-1]):.1f}x",
        "P2P 优势: 每个 rank 只组装自己拥有的 ~N/world 个参数",
        "round-robin 分配保证每个 rank 拥有参数数量之差 <= 1",
        "注意: 当 world > n_params 时会出现空 rank，节省倍数封顶于 n_params。",
    )


if __name__ == "__main__":
    fig, txt = fig_shard_ranges(world=4, numel=100)
    print("saved:", save(fig, "p2p_shard_ranges.png"))
    print(txt)
    print()

    fig, txt = fig_ownership_map(n_params=12, world=4)
    print("saved:", save(fig, "p2p_ownership.png"))

    fig, txt = fig_p2p_memory(n_params=64, world_range=(1, 16),
                               param_shape=(128, 64))
    print("saved:", save(fig, "p2p_memory.png"))
