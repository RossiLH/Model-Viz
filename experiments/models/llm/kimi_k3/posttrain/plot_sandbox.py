# -*- coding: utf-8 -*-
"""可恢复沙箱可视化：生命周期 / 增量快照 / 池超分配。"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title, save
from models.llm.kimi_k3.posttrain.test_helpers import (
    sandbox_lifecycle_trace,
    incremental_snapshot_trace,
    pool_overcommit_curve,
)
from models.llm.kimi_k3.posttrain.agentenv.sandbox import SandboxState


def fig_sandbox_lifecycle(pause_fraction: float = 0.7,
                          total_ticks: int = 100) -> tuple:
    """沙箱生命周期：运行/暂停切换时间线 + 累积运行占比。"""
    setup_cjk_font()
    fp = get_font_prop()
    ticks, states, running_pct = sandbox_lifecycle_trace(
        pause_fraction=pause_fraction, total_ticks=total_ticks)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

    # 左：状态时间线
    colors = []
    for s in states:
        colors.append("#2ecc71" if s == "running" else "#e67e22"
                      if s == "paused" else "#95a5a6")
    ax1.barh([0] * len(ticks), [1] * len(ticks), left=ticks,
             color=colors, edgecolor="none", height=0.4)
    ax1.set_yticks([0])
    ax1.set_yticklabels(["沙箱状态"], fontproperties=fp)
    ax1.set_xlabel("时间 (tick)", fontproperties=fp)
    ax1.set_xlim(0, total_ticks)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#2ecc71"),
        plt.Rectangle((0, 0), 1, 1, fc="#e67e22"),
    ]
    ax1.legend(legend_handles, ["运行 RUNNING", "暂停 PAUSED"],
               loc="upper right", prop=fp)
    _ax_title(ax1, "沙箱状态时间线")

    # 右：累积运行占比
    ax2.plot(range(1, len(running_pct) + 1), running_pct,
             color="#3498db", lw=2)
    ax2.axhline(y=1 - pause_fraction, color="#e74c3c",
                linestyle="--", lw=1.5,
                label=f"理论值 {1-pause_fraction:.0%}")
    ax2.set_xlabel("时间 (tick)", fontproperties=fp)
    ax2.set_ylabel("累积运行占比", fontproperties=fp)
    ax2.set_ylim(0, 1)
    ax2.legend(prop=fp)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    _ax_title(ax2, "累积运行占比 (论文报告: 98% 时间在等待)")

    fig.suptitle("沙箱生命周期：暂停几乎零成本",
                 fontsize=13, fontweight="bold", y=0.98, fontproperties=fp)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    final_pct = running_pct[-1] if running_pct else 0.0
    return fig, _text(
        f"pause_fraction={pause_fraction:.0%}, total_ticks={total_ticks}",
        f"最终运行占比: {final_pct:.1%} (理论值 ~{1-pause_fraction:.0%})",
        "暂停态不消耗 CPU/内存，支持大规模超分配",
    )


def fig_incremental_snapshot(n_pages: int = 50, n_snapshots: int = 8,
                             dirty_ratio: float = 0.2,
                             seed: int = 42) -> tuple:
    """增量快照：每次只保存自上次以来的脏页。"""
    setup_cjk_font()
    fp = get_font_prop()
    indices, dirty_counts = incremental_snapshot_trace(
        n_pages=n_pages, n_snapshots=n_snapshots,
        dirty_ratio=dirty_ratio, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

    # 左：每次快照的脏页数
    ax1.bar(indices, dirty_counts, color="#9b59b6", alpha=0.85,
            edgecolor="white")
    ax1.axhline(y=n_pages, color="#e74c3c", linestyle="--",
                lw=1.5, label=f"全量快照 = {n_pages} 页")
    ax1.set_xlabel("快照序号", fontproperties=fp)
    ax1.set_ylabel("脏页数", fontproperties=fp)
    ax1.legend(prop=fp)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    _ax_title(ax1, "每次快照的脏页数")

    # 右：累积总开销 vs 全量对比
    cumul = []
    total = 0
    for d in dirty_counts:
        total += d
        cumul.append(total)
    full_total = [n_pages * (i + 1) for i in range(len(indices))]
    ax2.plot(indices, cumul, "o-", color="#1abc9c", lw=2,
             label="增量快照（累积）")
    ax2.plot(indices, full_total, "s--", color="#e74c3c", lw=1.5,
             label="全量快照（累积）")
    ax2.set_xlabel("快照序号", fontproperties=fp)
    ax2.set_ylabel("累积页数", fontproperties=fp)
    ax2.legend(prop=fp)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ratio = cumul[-1] / full_total[-1] if full_total else 0
    _ax_title(ax2, f"累积开销：增量 vs 全量 (节省 {1-ratio:.0%})")

    fig.suptitle("增量快照（论文报告 133ms/49ms）",
                 fontsize=13, fontweight="bold", y=0.98, fontproperties=fp)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, _text(
        f"总页数={n_pages}, 快照数={n_snapshots}, dirty_ratio≈{dirty_ratio:.0%}",
        f"增量累积: {cumul[-1]} 页, 全量累积: {full_total[-1]} 页",
        f"节省比例: {(1-ratio)*100:.1f}% (修改越少越省)",
    )


def fig_pool_overcommit(capacity: int = 8, n_sandboxes: int = 40,
                        pause_ratio: float = 0.8,
                        seed: int = 42) -> tuple:
    """沙箱池超分配：暂停沙箱不计入容量，总数量远超 capacity。"""
    setup_cjk_font()
    fp = get_font_prop()
    ns, ratios, running_list, paused_list = pool_overcommit_curve(
        capacity=capacity, n_sandboxes=n_sandboxes,
        pause_ratio=pause_ratio, seed=seed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=150)

    # 左：超分配比曲线
    ax1.plot(ns, ratios, "o-", color="#e67e22", lw=2)
    ax1.axhline(y=6.5, color="#e74c3c", linestyle="--", lw=1.5,
                label="论文报告最大值 6.5x")
    ax1.axhline(y=1.0, color="#7f8c8d", linestyle=":", lw=1,
                label="无超分配 1.0x")
    ax1.set_xlabel("已创建沙箱数", fontproperties=fp)
    ax1.set_ylabel("超分配比 (alive/running)", fontproperties=fp)
    ax1.legend(prop=fp)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    _ax_title(ax1, "超分配比随沙箱数增长")

    # 右：运行/暂停/总数堆叠条
    width = 0.6
    x = list(range(len(ns)))
    ax2.bar(x, running_list, width, label="运行中", color="#2ecc71", alpha=0.85)
    ax2.bar(x, paused_list, width, bottom=running_list,
            label="暂停中", color="#e67e22", alpha=0.85)
    ax2.axhline(y=capacity, color="#e74c3c", linestyle="--",
                lw=1.5, label=f"容量上限 {capacity}")
    ax2.set_xticks(x[::max(1, len(x)//6)])
    ax2.set_xticklabels([ns[i] for i in range(0, len(ns), max(1, len(ns)//6))],
                        fontproperties=fp)
    ax2.set_xlabel("批次", fontproperties=fp)
    ax2.set_ylabel("沙箱数", fontproperties=fp)
    ax2.legend(prop=fp)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    _ax_title(ax2, "运行 vs 暂停数量")

    fig.suptitle("沙箱池超分配（论文报告 5100 万沙箱 / 6.5x）",
                 fontsize=13, fontweight="bold", y=0.98, fontproperties=fp)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    max_ratio = max(ratios) if ratios else 0
    return fig, _text(
        f"容量={capacity}, 总沙箱={ns[-1]}, pause_ratio≈{pause_ratio:.0%}",
        f"最大超分配比: {max_ratio:.1f}x",
        "暂停沙箱不计入容量 → 大规模超分配成为可能",
    )


if __name__ == "__main__":
    fig, _ = fig_sandbox_lifecycle()
    save(fig, "sandbox_lifecycle.png")
    fig, _ = fig_incremental_snapshot()
    save(fig, "incremental_snapshot.png")
    fig, _ = fig_pool_overcommit()
    save(fig, "pool_overcommit.png")
    print("all sandbox plots saved to assets/")
