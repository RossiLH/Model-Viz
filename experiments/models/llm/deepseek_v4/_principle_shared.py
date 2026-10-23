# -*- coding: utf-8 -*-
"""原理示意图的共享绘图构件（V3 新增）。

与 `_shared.py` 的分工：
- `_shared.py`：提供「真实模型 + 真实配置」，服务于**实测数据图**（11 张旧图）。
- 本文件：提供「示意图画布 + 盒子/箭头/标注」原语，服务于**原理结构图**。

为什么要分开？因为两类图的取证等级不同：实测图的每个数字都来自 CPU 真跑，
原理图画的是**结构与数据流**（形状、流向、可见性），属于「论文明确写的」或
「本项目代码结构」的可视化，不承载实测数字。图注里会明确标注这一点，
避免读者把示意图当性能证据（对应文档 0x0FB.4 的两条金规）。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

OUT = pathlib.Path(__file__).resolve().parent

# 统一配色（与旧 11 张实测图同色系，保证文档内视觉连贯）
C_BASE = "#c62828"      # 红：标准/基线/朴素做法
C_V4 = "#238636"        # 绿：V4 创新做法
C_MID = "#1565c0"       # 蓝：中间态/过渡方案
C_WARN = "#ef8c00"      # 橙：代价/风险/丢失的信息
C_GRAY = "#8b949e"      # 灰：被裁掉/不可见/已丢弃
C_INK = "#24292f"        # 近黑：文字与主线条
C_PAPER = "#f6f8fa"     # 浅底：盒子填充


def setup_fonts() -> str:
    """启用中文字体（复用全仓 plot_utils），并修 mathtext 负号缺字形问题。"""
    import matplotlib.pyplot as plt
    from plot_utils import setup_cjk_font
    name = setup_cjk_font()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    return name


def box(ax, x, y, w, h, text, fc=C_PAPER, ec=C_INK, fontsize=9,
        lw=1.4, alpha=1.0, tc=None, weight="normal", zorder=3):
    """画一个带文字的圆角矩形（示意图的基本单元）。

    x/y 是**左下角**坐标，w/h 是宽高；text 支持 \\n 换行。
    """
    from matplotlib.patches import FancyBboxPatch
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha,
                       zorder=zorder)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc or C_INK, weight=weight,
            zorder=zorder + 1, linespacing=1.45)
    return p


def arrow(ax, x1, y1, x2, y2, color=C_INK, lw=1.6, style="-|>",
          ls="-", alpha=1.0, rad=0.0, zorder=2):
    """画箭头（rad != 0 时为弧线，用于跨层回连）。"""
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=13, color=color, linewidth=lw,
                        linestyle=ls, alpha=alpha, zorder=zorder,
                        connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def label(ax, x, y, text, fontsize=9, color=C_INK, ha="center", va="center",
          weight="normal", style="normal", zorder=6, bbox=None):
    """画一段说明文字。"""
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va=va,
            weight=weight, style=style, zorder=zorder, bbox=bbox,
            linespacing=1.45)


def blank_canvas(figsize, xlim, ylim):
    """建一张无坐标轴的纯示意画布。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> pathlib.Path:
    """存图到 experiments 目录（与旧图同一位置，再由文档流程拷入 Picture/）。"""
    path = OUT / name
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("已生成", path)
    return path
