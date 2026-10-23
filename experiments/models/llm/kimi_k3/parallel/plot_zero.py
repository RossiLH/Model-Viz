# -*- coding: utf-8 -*-
"""ZeRO / 梯度分片可视化。

对应源码 `models/llm/kimi_k3/parallel/zero.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_zero.py`。

三个图：
  1. fig_shard_layout      — shard_ranges 分片区间可视化（对齐 reduce_scatter）
  2. fig_memory_staircase  — 显存阶梯：ZeRO-0 / ZeRO-1 / ZeRO-2 / ZeRO-2+offload
  3. fig_roundtrip_world1  — world=1 下 materialise 完整重建梯度（往返等价性）
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
    ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.parallel.zero import (
    ShardedGradients, shard_ranges, reduce_scatter_grads,
)


def fig_shard_layout(numel=1000, world=8):
    setup_cjk_font()
    fp = get_font_prop()
    numel = int(numel); world = int(world)

    rngs = shard_ranges(numel, world)
    per = (numel + (-numel) % world) // world

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)

    colors = plt.cm.tab20(np.linspace(0, 1, world))
    for i, (lo, hi) in enumerate(rngs):
        ax.barh([0], [hi - lo], left=[lo], height=0.6,
                color=colors[i], edgecolor="white", linewidth=1.5)
        mid = (lo + hi) / 2
        if hi - lo > 0:
            ax.text(mid, 0, f"rank {i}\n[{lo},{hi})",
                    ha="center", va="center", fontsize=8, color="white",
                    fontweight="bold", fontproperties=fp)

    ax.set_yticks([])
    ax.set_xlim(0, numel)
    _at(ax, f"shard_ranges: {numel} 个元素切成 {world} 段（连续、近似相等）")
    _al(ax, "元素下标", "")
    ax.grid(True, alpha=0.3, axis="x")

    # 验证：所有段拼接覆盖 [0, numel)
    total = sum(hi - lo for lo, hi in rngs)
    contiguous = all(rngs[i][1] == rngs[i + 1][0] for i in range(world - 1))
    starts_at_zero = rngs[0][0] == 0
    ends_at_numel = rngs[-1][1] == numel

    fig.text(0.5, 0.01,
             f"总元素: {total} = numel={numel}  |  "
             f"连续: {contiguous}  |  "
             f"[0, numel) 全覆盖: {starts_at_zero and ends_at_numel}  |  "
             f"分块宽度 per = ceil({numel}/{world}) = {per}",
             ha="center", fontsize=10, fontproperties=fp,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig, _text(
        f"{numel} 个元素切成 {world} 段，分块宽度 = ceil(numel/world) = {per}",
        f"连续性: {contiguous}   全覆盖: {starts_at_zero and ends_at_numel}",
        f"最末段长度: {rngs[-1][1] - rngs[-1][0]}（numel 不能整除时稍短）",
        "每个 rank 的分片就是一个连续切片（view），reduce-scatter 和优化器 step 都操作 view",
        "连续性是重要性质：避免多次 gather，直接 view 即可操作分片",
    )


def fig_memory_staircase(total_params=1_000_000):
    setup_cjk_font()
    fp = get_font_prop()
    total_params = int(total_params)
    world = 8
    bytes_per_param = 4  # float32

    # 不同策略下的 GPU 显存占用（梯度部分）
    labels = []
    values = []

    # ZeRO-0: 不做分片，完整梯度都在 GPU
    gpu_0 = total_params * bytes_per_param
    labels.append("ZeRO-0\n(无分片)")
    values.append(gpu_0)

    # ZeRO-1: 优化器状态分片（本实现重点是梯度，ZeRO-1 这里仅参考）
    # 梯度仍在 GPU，但优化器状态被分片
    gpu_1 = total_params * bytes_per_param
    labels.append("ZeRO-1\n(优化器分片)")
    values.append(gpu_1)

    # ZeRO-2: 梯度也分片，每 rank 只存 1/world
    gpu_2 = total_params * bytes_per_param // world
    labels.append("ZeRO-2\n(梯度分片)")
    values.append(gpu_2)

    # ZeRO-2 + offload: 梯度分片 + offload 到 CPU，GPU 梯度为 0
    gpu_2_off = 0
    labels.append("ZeRO-2 + offload\n(梯度移 CPU)")
    values.append(gpu_2_off)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]
    bars = ax1.bar(labels, values, color=colors, width=0.6, edgecolor="white")
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + values[0] * 0.01,
                 f"{val/1024:.0f}KB" if val >= 1024 else f"{val}B",
                 ha="center", fontsize=9, fontweight="bold", fontproperties=fp)
    _at(ax1, "GPU 显存占用：梯度部分的阶梯式降低")
    _al(ax1, "", "GPU 字节数")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", labelsize=8)

    # 右：随 world size 变化的显存节省比例
    worlds = list(range(1, 17))
    saved_ratio = [(w - 1) / w * 100 for w in worlds]  # 1 - 1/world
    ax2.plot(worlds, saved_ratio, "o-", color="#3498db", linewidth=2, markersize=7)
    ax2.fill_between(worlds, 0, saved_ratio, alpha=0.2, color="#3498db")
    for w in [1, 2, 4, 8, 16]:
        idx = w - 1
        if idx < len(worlds):
            ax2.annotate(f"{saved_ratio[idx]:.0f}%",
                         xy=(w, saved_ratio[idx]),
                         xytext=(w + 0.3, saved_ratio[idx] - 3),
                         fontsize=9, fontproperties=fp)
    _at(ax2, "梯度显存节省比例 = (world - 1) / world")
    _al(ax2, "world size (数据并行度)", "节省比例 (%)")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle(
        f"ZeRO 显存阶梯：梯度从全量 ({gpu_0/1024:.0f}KB) 降到 0 (offload CPU)",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"总参数量: {total_params:,} (float32 = {total_params*4:,} B = {total_params*4/1024:.0f} KB)",
        f"ZeRO-0: {gpu_0/1024:.0f} KB（全部梯度在 GPU）",
        f"ZeRO-2 (world={world}): {gpu_2/1024:.0f} KB（每 rank 只存 1/{world} 片）",
        "ZeRO-2 + offload: 0 B（梯度移到 CPU，GPU 只剩双缓冲）",
        f"理论节省比例: world={world} 时 {(world-1)/world*100:.0f}%",
    )


def fig_roundtrip_world1(n_params=3, dim=64):
    setup_cjk_font()
    fp = get_font_prop()
    n_params = int(n_params); dim = int(dim)

    # 创建 n_params 个参数，设置梯度
    params = [torch.nn.Parameter(torch.randn(dim, dim)) for _ in range(n_params)]
    torch.manual_seed(0)
    for p in params:
        p.grad = torch.randn_like(p)
    original_grads = [p.grad.clone() for p in params]

    # world=1，offload 到 CPU
    sharded = ShardedGradients(params, world=1, rank=0, offload_to_cpu=True)
    sharded.accumulate()

    # 重建
    reconstructed = [sharded.materialise(i) for i in range(n_params)]

    # 计算误差
    max_diffs = []
    for i in range(n_params):
        max_diff = (reconstructed[i] - original_grads[i]).abs().max().item()
        max_diffs.append(max_diff)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5), dpi=120)

    # 左：往返前后的梯度数值对照（散点，y=x 对角线）
    # 说明：world=1 时误差恒为 0，画误差热图只会得到一整块纯色，信息量为零。
    # 改用「原始 vs 重建」散点：所有点严格落在 y=x 上才说明往返等价。
    orig_flat = original_grads[0].reshape(-1).numpy()
    rec_flat = reconstructed[0].reshape(-1).detach().numpy()
    n_show = min(orig_flat.size, 4000)
    idx = np.linspace(0, orig_flat.size - 1, n_show).astype(int)
    ax0.scatter(orig_flat[idx], rec_flat[idx], s=6, alpha=0.35,
                color="#3498db", edgecolors="none", label="每个梯度元素")
    lo = float(min(orig_flat.min(), rec_flat.min()))
    hi = float(max(orig_flat.max(), rec_flat.max()))
    ax0.plot([lo, hi], [lo, hi], "--", color="#e74c3c", lw=1.4, label="y = x（完全等价）")
    ax0.set_aspect("equal", adjustable="box")
    _at(ax0, f"第 0 个参数：原始 vs 重建（{n_show}/{orig_flat.size} 点）")
    _al(ax0, "原始梯度值", "重建梯度值")
    _leg(ax0)
    ax0.grid(True, alpha=0.3)

    # 中：逐元素误差沿扁平索引的分布（折线），直观看到「处处为 0」
    diff0 = (reconstructed[0] - original_grads[0]).reshape(-1).abs().detach().numpy()
    ax1.plot(np.arange(diff0.size), diff0, lw=0.8, color="#e67e22")
    ax1.axhline(0.0, color="#7f8c8d", lw=1.0, ls=":")
    if diff0.max() > 0:
        ax1.set_yscale("log")
    else:
        ax1.set_ylim(-1e-3, 1e-3)
        ax1.text(0.5, 0.5, "逐元素误差恒为 0\n（float32 下精确重建）",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="#27ae60", fontproperties=fp)
    _at(ax1, f"第 0 个参数逐元素 |误差| (max={max_diffs[0]:.2e})")
    _al(ax1, "扁平索引", "abs diff")
    ax1.grid(True, alpha=0.3)

    # 右：各参数的最大误差柱状图（log scale；若全为 0 则回退到 linear）
    param_labels = [f"param {i}" for i in range(n_params)]
    all_zero = max(max_diffs) == 0
    ax2.bar(param_labels, [max(d, 1e-12) for d in max_diffs],
            color="#9b59b6", width=0.5)
    if not all_zero:
        ax2.set_yscale("log")
        _al(ax2, "", "max abs diff (log)")
    else:
        # 全 0 时 log 轴会把柱子画到画布外，改用线性轴并给出显式提示
        ax2.set_ylim(0, 1.0)
        _al(ax2, "", "max abs diff (linear)")
        ax2.text(0.5, 0.55, "全部参数误差 = 0",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=13, fontweight="bold", color="#27ae60", fontproperties=fp)
    _at(ax2, f"各参数最大重建误差 (world=1, offload CPU)")
    ax2.grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(max_diffs):
        y = 0.05 if all_zero else max(d, 1e-12) * 2.0
        ax2.text(i, y, f"{d:.1e}", ha="center", fontsize=8, fontproperties=fp)

    all_none = all(p.grad is None for p in params)
    fig.suptitle(
        f"ZeRO 往返重建：accumulate -> CPU 分片 -> materialise   "
        f"GPU 梯度已释放: {all_none}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    max_all = max(max_diffs)
    return fig, _text(
        f"world=1 时往返重建最大误差: {max_all:.2e}"
        + ("（逐元素完全相等，float32 下精确重建）" if all_zero else "（float32 下几乎精确）"),
        f"GPU 梯度已释放: {all_none} (accumulate 后 p.grad = None)",
        "分片驻留 CPU: 节省 GPU 显存 = 全部梯度的字节数",
        "world>1 时真实场景需要 all-gather 才能重建完整梯度",
        "本实现的分片与 reduce_scatter_grads 的分块宽度严格对齐（见 test）",
    )


if __name__ == "__main__":
    fig, txt = fig_shard_layout()
    print("saved:", save(fig, "zero_shard_layout.png"))
    print(txt); print()
    fig, txt = fig_memory_staircase()
    print("saved:", save(fig, "zero_memory_staircase.png"))
    print(txt); print()
    fig, txt = fig_roundtrip_world1()
    print("saved:", save(fig, "zero_roundtrip.png"))
    print(txt)
