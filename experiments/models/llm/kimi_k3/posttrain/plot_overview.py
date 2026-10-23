# -*- coding: utf-8 -*-
"""后训练全景图：SFT → RL → MOPD → AgentENV 全链路架构示意。

Tab 1：展示 K3 后训练 7 大组件如何串成一条流水线，以及每个组件的
核心论文章节引用。作为整个 demo 的导航页。
"""
from __future__ import annotations

import sys, os
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_utils import setup_cjk_font, get_font_prop, text as _text, save


def fig_pipeline_overview() -> tuple:
    """返回 (fig, text)：后训练流水线架构图。"""
    setup_cjk_font()
    fp = get_font_prop()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 7 个阶段的盒子
    stages = [
        ("XTML\n对话模板",     "附录 F",  "#3498db"),
        ("SFT 冷启动",        "§4.1.1", "#2ecc71"),
        ("预算控制\n9 专家",   "§4.1.2", "#e67e22"),
        ("GRM\n生成式奖励",    "§4.1.2", "#9b59b6"),
        ("部分 Rollout",      "§4.1.2", "#e74c3c"),
        ("MOPD\n多教师蒸馏",   "§4.1.3", "#1abc9c"),
        ("AgentENV\n沙箱",     "§5.3",   "#f39c12"),
    ]
    n = len(stages)
    box_w, box_h = 1.6, 1.4
    gap = (14 - n * box_w) / (n + 1)

    for i, (name, section, color) in enumerate(stages):
        x = gap + i * (box_w + gap)
        y = 2.5
        rect = mpatches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color, alpha=0.85, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h * 0.65, name,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white", fontproperties=fp)
        ax.text(x + box_w / 2, y + box_h * 0.22, section,
                ha="center", va="center", fontsize=8,
                color="white", alpha=0.9, fontproperties=fp)

        if i < n - 1:
            ax.annotate("",
                        xy=(x + box_w + gap * 0.25, y + box_h / 2),
                        xytext=(x + box_w + gap * 0.05, y + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # 下方阶段分组
    groups = [
        ("数据格式", "#3498db", 0, 0),
        ("冷启动",   "#2ecc71", 1, 1),
        ("强化学习", "#e74c3c", 2, 4),
        ("蒸馏合并", "#1abc9c", 5, 5),
        ("执行环境", "#f39c12", 6, 6),
    ]
    for gname, gcolor, s, e in groups:
        x0 = gap + s * (box_w + gap)
        x1 = gap + e * (box_w + gap) + box_w
        ax.plot([x0, x1], [2.0, 2.0], color=gcolor, lw=3, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 1.55, gname,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=gcolor, fontproperties=fp)

    ax.set_title("K3 后训练流水线全景 (§4 Post-Training)",
                 fontsize=14, fontweight="bold", y=0.96, fontproperties=fp)

    fig.tight_layout()
    return fig, _text(
        "**7 大组件**：XTML → SFT → 预算控制 → GRM → 部分 Rollout → MOPD → AgentENV",
        "**3 阶段训练**：冷启动(SFT) → 强化学习(9 专家) → 蒸馏合并(MOPD)",
        "**核心创新**：预算控制 + 部分 Rollout + MOPD + 可恢复沙箱",
    )


if __name__ == "__main__":
    fig, _ = fig_pipeline_overview()
    path = save(fig, "plot_pipeline_overview.png")
    print("saved:", path)
