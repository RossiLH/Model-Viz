# -*- coding: utf-8 -*-
"""流水线并行 + AttnRes 缓存 + 激活均衡可视化。

对应源码 `models/llm/kimi_k3/parallel/pipeline.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_pipeline.py`。

四个图：
  1. fig_layer_partition     — 层切分（含 virtual pipeline stages）
  2. fig_1f1b_schedule       — interleaved 1F1B 调度甘特图 + warmup 倾斜
  3. fig_attnres_cache       — AttnRes 增量通信 vs naive 重发全集
  4. fig_activation_balance  — 激活均衡：远程 offload 拉平驻留
"""
from __future__ import annotations

from plot_utils import add_src_to_path

add_src_to_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop, text as _text,
    ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.parallel.pipeline import (
    partition_layers, one_f_one_b_schedule, AttnResCache, activation_balance,
)


def fig_layer_partition(n_layers=93, pp=16, virtual=1):
    setup_cjk_font()
    fp = get_font_prop()
    n_layers = int(n_layers); pp = int(pp); virtual = int(virtual)

    chunks = partition_layers(n_layers, pp=pp, virtual_pipeline_stages=virtual)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)

    colors = plt.cm.tab20(np.linspace(0, 1, len(chunks)))
    current = 0
    for i, chunk in enumerate(chunks):
        size = len(chunk)
        ax.barh([0], [size], left=[current], height=0.6,
                color=colors[i], edgecolor="white", linewidth=1.5)
        mid = current + size / 2
        if size >= 4:
            ax.text(mid, 0, f"chunk {i}\n{size} 层",
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold", fontproperties=fp)
        current += size

    # 标注 pp stage 边界
    stage_size = len(chunks) // pp
    for s in range(1, pp):
        boundary = sum(len(c) for c in chunks[:s * stage_size])
        ax.axvline(boundary, color="#e74c3c", linestyle="--", linewidth=2, alpha=0.7)
        ax.text(boundary, 0.7, f"stage {s-1} / {s}",
                ha="center", va="bottom", fontsize=9, color="#e74c3c",
                fontproperties=fp)

    ax.set_yticks([])
    ax.set_xlim(0, n_layers)
    _at(ax, f"层切分：{n_layers} 层 / {pp} stage / virtual={virtual} → 共 {len(chunks)} 段")
    _al(ax, "层编号", "")
    ax.grid(True, alpha=0.3, axis="x")

    sizes = [len(c) for c in chunks]
    fig.text(0.5, 0.01,
             f"段长范围: [{min(sizes)}, {max(sizes)}] 层  |  "
             f"最大-最小差: {max(sizes) - min(sizes)} 层  |  "
             f"总层数: {sum(sizes)} = {n_layers}",
             ha="center", fontsize=10, fontproperties=fp,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig, _text(
        f"{n_layers} 层切成 {len(chunks)} 段（pp={pp} × virtual={virtual}）",
        f"段长范围: [{min(sizes)}, {max(sizes)}] 层   差值: {max(sizes) - min(sizes)}",
        f"总层数: {sum(sizes)} = {n_layers} （不丢层）",
        "不要求整除：多出来的层均匀分给前面的段（每段至多差 1 层）",
        "段是连续的：跨段边界只有 pp-1 处，通信开销最小",
    )


def fig_1f1b_schedule(n_stages=4, n_mb=16, virtual=2):
    setup_cjk_font()
    fp = get_font_prop()
    n_stages = int(n_stages); n_mb = int(n_mb); virtual = int(virtual)

    # 收集每个 stage 的 step 序列
    all_scheds = []
    for st in range(n_stages):
        s = one_f_one_b_schedule(st, n_stages, n_mb, virtual)
        all_scheds.append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

    # 左：甘特图——每个 stage 的 forward/backward 时间线
    max_steps = max(len(s.steps) for s in all_scheds)
    fwd_color = "#3498db"
    bwd_color = "#e74c3c"
    for st in range(n_stages):
        steps = all_scheds[st].steps
        for t, (kind, m, _) in enumerate(steps):
            color = fwd_color if kind == "forward" else bwd_color
            ax1.barh(n_stages - 1 - st, 1, left=t, height=0.7,
                     color=color, edgecolor="white", linewidth=0.5, alpha=0.85)

    ax1.set_yticks(range(n_stages))
    ax1.set_yticklabels([f"stage {n_stages - 1 - s}" for s in range(n_stages)],
                        fontproperties=fp)
    _at(ax1, f"1F1B 调度甘特图 (stages={n_stages}, mb={n_mb}, virtual={virtual})")
    _al(ax1, "时间步", "")
    ax1.grid(True, alpha=0.2, axis="x")

    legend_patches = [
        mpatches.Patch(color=fwd_color, label="forward"),
        mpatches.Patch(color=bwd_color, label="backward"),
    ]
    ax1.legend(handles=legend_patches, prop=fp, fontsize=9, loc="upper right")

    # 右：每个 stage 的 warmup forward 数（即驻留倾斜）
    warmups = [s.warmup_steps for s in all_scheds]
    stages_x = list(range(n_stages))
    bars = ax2.bar(stages_x, warmups, color="#9b59b6", width=0.6, edgecolor="white")
    for bar, w in zip(bars, warmups):
        ax2.text(bar.get_x() + bar.get_width() / 2, w + 0.2, str(w),
                 ha="center", fontsize=10, fontweight="bold", fontproperties=fp)
    _at(ax2, "warmup 前向数：前段 stage 驻留更多 micro-batch")
    _al(ax2, "stage", "warmup forward 数")
    ax2.grid(True, alpha=0.3, axis="y")

    skew = warmups[0] / max(warmups[-1], 1)
    fig.suptitle(
        f"interleaved 1F1B：前段 stage 积压更多激活（这就是激活均衡要解决的 skew）",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"stages={n_stages}, micro-batches={n_mb}, virtual_pipeline_stages={virtual}",
        f"warmup 范围: [{min(warmups)}, {max(warmups)}] （stage 0 最大，stage {n_stages-1} 最小）",
        f"倾斜比: {skew:.1f}x（stage 0 vs stage {n_stages-1}）",
        "1F1B 调度：warmup 纯 forward → 稳态 前向/后向交替 → drain 纯 backward",
        "前段 stage prefetch 更多 → 激活驻留更多 → 激活均衡的必要性（远程 offload）",
    )


def fig_attnres_cache(n_stages=4, n_mb=8, blocks_per_stage=3):
    setup_cjk_font()
    fp = get_font_prop()
    n_stages = int(n_stages); n_mb = int(n_mb)
    blocks_per_stage = int(blocks_per_stage)

    # 模拟：每个 stage 完成后，向 AttnResCache 追加 blocks_per_stage 个 block
    cache = AttnResCache()
    block_shape = (2, 4, 32)  # 模拟的 block 形状

    transferred_per_stage = [0] * n_stages
    naive_per_stage = [0] * n_stages

    # 模拟一次 forward pass 中 AttnRes 的增量传递
    # 假设：每个 stage 收到上游的 n_already_sent 个已发送 block
    n_already_sent = 0
    total_blocks = 0
    for st in range(n_stages):
        cache.receive(st, [torch.randn(*block_shape) for _ in range(n_already_sent)])
        # 本 stage 生成 blocks_per_stage 个新 block
        for _ in range(blocks_per_stage):
            cache.append(st, torch.randn(*block_shape))
            total_blocks += 1
        # 发送新增部分到下游
        new_blocks = cache.outgoing(st, n_already_sent=n_already_sent)
        transferred_per_stage[st] = len(new_blocks)
        n_already_sent += len(new_blocks)
        naive_per_stage[st] = n_already_sent  # naive：每次都重发全集

    # 字节统计
    block_bytes = block_shape[0] * block_shape[1] * block_shape[2] * 4
    incremental_total = cache.bytes_transferred
    naive_total = sum(b * block_bytes for b in naive_per_stage)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    stages_x = list(range(n_stages))
    w = 0.35
    ax1.bar([i - w/2 for i in stages_x],
            [t * block_bytes / 1024 for t in transferred_per_stage],
            width=w, label="增量 (KCP AttnRes)", color="#2ecc71", alpha=0.85)
    ax1.bar([i + w/2 for i in stages_x],
            [n * block_bytes / 1024 for n in naive_per_stage],
            width=w, label="naive（重发全集）", color="#e74c3c", alpha=0.7)
    _at(ax1, "每 stage 传输量对比")
    _al(ax1, "stage", "传输字节 (KB)")
    _leg(ax1)
    ax1.grid(True, alpha=0.3, axis="y")

    # 右：累计传输量
    inc_cum = np.cumsum([t * block_bytes / 1024 for t in transferred_per_stage])
    naive_cum = np.cumsum([n * block_bytes / 1024 for n in naive_per_stage])
    ax2.plot(stages_x, inc_cum, "o-", color="#2ecc71", linewidth=2,
             markersize=7, label="增量 (AttnRes Cache)")
    ax2.plot(stages_x, naive_cum, "s--", color="#e74c3c", linewidth=2,
             markersize=7, label="naive（重发全集）")
    _at(ax2, "累计传输量：增量 vs naive")
    _al(ax2, "stage", "累计字节 (KB)")
    _leg(ax2)
    ax2.grid(True, alpha=0.3)
    ratio = naive_total / max(incremental_total, 1)
    ax2.text(0.02, 0.95,
             f"节省比: {ratio:.1f}x\n"
             f"增量: {incremental_total/1024:.0f} KB\n"
             f"naive: {naive_total/1024:.0f} KB",
             transform=ax2.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
             fontproperties=fp)

    fig.suptitle(
        f"AttnRes 增量通信：只传本 stage 新增的 block，用完即释放",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"stages={n_stages}, blocks_per_stage={blocks_per_stage}, micro-batches={n_mb}",
        f"增量传输总量: {incremental_total/1024:.1f} KB",
        f"naive 重发总量: {naive_total/1024:.1f} KB",
        f"节省比例: {ratio:.1f}x",
        "驻留内存 <= 在途 micro-batch 的活 block 数（micro-batch 结束即 retire 释放）",
    )


def fig_activation_balance(n_stages=4, skew=0.3):
    setup_cjk_font()
    fp = get_font_prop()
    n_stages = int(n_stages); skew = float(skew)

    # 模拟：前段 stage 驻留多，后段少
    loads = [int(10 * (1 - skew * i)) for i in range(n_stages)]
    # 取整保证总和
    total = sum(loads)
    capacity = total // n_stages + (1 if total % n_stages else 0)

    moves = activation_balance(loads, capacity)
    balanced = list(loads)
    for src, dst, c in moves:
        balanced[src] -= c
        balanced[dst] += c

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    stages_x = list(range(n_stages))
    w = 0.35
    ax1.bar([i - w/2 for i in stages_x], loads, width=w,
            label="均衡前（skew）", color="#e74c3c", alpha=0.8)
    ax1.bar([i + w/2 for i in stages_x], balanced, width=w,
            label="均衡后", color="#2ecc71", alpha=0.8)
    ax1.axhline(capacity, color="#f1c40f", linestyle="--", linewidth=2,
                label=f"capacity = {capacity}")
    _at(ax1, "各 stage 激活驻留：均衡前后对比")
    _al(ax1, "stage", "驻留 micro-batch 数")
    _leg(ax1)
    ax1.grid(True, alpha=0.3, axis="y")

    # 右：搬移计划瀑布图
    ax2.set_title("激活搬移计划 (src → dst)", fontsize=11, fontweight="bold",
                  pad=10, fontproperties=fp)
    if moves:
        y_pos = list(range(len(moves)))
        move_labels = [f"{src} → {dst}: {cnt}" for src, dst, cnt in moves]
        move_counts = [cnt for _, _, cnt in moves]
        ax2.barh(y_pos, move_counts, color="#3498db", height=0.6)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(move_labels, fontsize=9, fontproperties=fp)
        ax2.invert_yaxis()
        _al(ax2, "搬移数量", "搬移顺序（从上到下依次执行）")
        ax2.grid(True, alpha=0.3, axis="x")
    else:
        ax2.text(0.5, 0.5, "无搬移（负载已均衡）",
                 ha="center", va="center", fontsize=12, fontproperties=fp,
                 transform=ax2.transAxes)

    all_ok = all(x <= capacity for x in balanced)
    fig.suptitle(
        f"激活均衡：远程 offload 把富集 stage 的激活搬到贫 stage   "
        f"全部 <= capacity: {all_ok}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"均衡前负载: {loads}   最大-最小: {max(loads) - min(loads)}",
        f"capacity = total/stages = {capacity}",
        f"均衡后负载: {balanced}   全部 <= capacity: {all_ok}",
        f"搬移次数: {len(moves)}   总搬移量: {sum(c for _,_,c in moves)}",
        "原理：把超过 capacity 的 stage 的激活远程 offload 到 peer pipeline rank",
    )


if __name__ == "__main__":
    fig, txt = fig_layer_partition()
    print("saved:", save(fig, "pipeline_layer_partition.png"))
    print(txt); print()
    fig, txt = fig_1f1b_schedule()
    print("saved:", save(fig, "pipeline_1f1b_schedule.png"))
    print(txt); print()
    fig, txt = fig_attnres_cache()
    print("saved:", save(fig, "pipeline_attnres_cache.png"))
    print(txt); print()
    fig, txt = fig_activation_balance()
    print("saved:", save(fig, "pipeline_activation_balance.png"))
    print(txt)
