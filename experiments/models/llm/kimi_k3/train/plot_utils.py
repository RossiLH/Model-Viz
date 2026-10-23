# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置 + 绘图共享工具，供 train 各绘图脚本复用。

功能:
  1. ``setup_cjk_font()`` — 探测系统 CJK 字体并启用
  2. ``apply_dark_theme()`` — 应用深色主题（匹配 Model-Viz 前端风格）
  3. ``get_font_prop()`` — 返回当前激活的 FontProperties 对象
  4. ``text()`` — 多行文本拼接，供 Gradio gr.Textbox 使用
  5. ``ax_title()`` / ``ax_labels()`` / ``ax_legend()`` — 带 fontproperties 的快捷函数
  6. ``assets_dir()`` — 返回(并按需创建) 本目录下的 assets/ 目录
  7. ``save()`` — 把 fig 保存到 assets/ 并关闭，供 plot 脚本 __main__ 使用
  8. ``add_src_to_path()`` — 把 Model-Viz/src 与本目录插入 sys.path

设计动机:
  train 目录下有多个 plot 脚本 + 1 个 gradio 入口，
  它们都需要「中文字体 + 存盘路径 + 文本拼接」这三件事。
  把它们收敛到一个模块，避免每个脚本各写一份 rcParams 配置。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

_CJK_CANDIDATES = [
    "SimHei", "Microsoft YaHei", "SimSun", "DengXian",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei",
]

_font_prop: FontProperties | None = None
_dark_applied: bool = False

HERE = os.path.dirname(os.path.abspath(__file__))


def add_src_to_path() -> None:
    """把 Model-Viz/src 和本目录插入 sys.path。

    本文件位于 experiments/models/llm/kimi_k3/train/，
    parents[5] 恰好是 Model-Viz/，所以 src 根为 parents[5]/"src"。
    插入本目录是为了让各 plot 脚本之间可以互相 `import plot_utils`。
    """
    src_root = str(Path(__file__).resolve().parents[5] / "src")
    for p in (src_root, HERE):
        if p not in sys.path:
            sys.path.insert(0, p)


def setup_cjk_font() -> str:
    """启用中文字体(若系统有), 返回实际使用的字体名。

    从 matplotlib 已扫描到的字体清单里逐个试候选字体，命中第一个就把它
    设为默认 sans-serif 并关闭 unicode 负号自动转 ASCII-，这样中文坐标轴/
    图例能正常显示。一个字体都没有时退回系统默认，只保证不报错。
    """
    global _font_prop
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            _font_prop = FontProperties(family=name)
            return name
    plt.rcParams["axes.unicode_minus"] = False
    _font_prop = None
    return "default"


def apply_dark_theme() -> None:
    """应用深色主题（InferFlux / Model-Viz 前端风格）。"""
    global _dark_applied
    if _dark_applied:
        return
    plt.rcParams["figure.facecolor"] = "#17181c"
    plt.rcParams["axes.facecolor"] = "#17181c"
    plt.rcParams["axes.edgecolor"] = "#888899"
    plt.rcParams["axes.labelcolor"] = "#dddddd"
    plt.rcParams["xtick.color"] = "#bbbbbb"
    plt.rcParams["ytick.color"] = "#bbbbbb"
    plt.rcParams["text.color"] = "#e8e8e8"
    plt.rcParams["axes.titlecolor"] = "#ffffff"
    plt.rcParams["legend.facecolor"] = "#22232a"
    plt.rcParams["legend.edgecolor"] = "#444455"
    plt.rcParams["legend.labelcolor"] = "#dddddd"
    _dark_applied = True


def get_font_prop() -> FontProperties | None:
    """返回 setup_cjk_font() 设置的 FontProperties。"""
    return _font_prop


def text(*lines: str) -> str:
    """多行文本拼接, 返回换行符分隔的字符串 (供 Gradio 文本框使用)。"""
    return "\n".join(lines)


def ax_title(ax, title: str, fontsize: int = 12) -> None:
    """带 fontproperties 的 ax.set_title(), 保证中文正常显示。"""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=10,
                 fontproperties=_font_prop)


def ax_labels(ax, xlabel: str = "", ylabel: str = "") -> None:
    """带 fontproperties 的坐标轴标签设置。"""
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=_font_prop)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=_font_prop)


def ax_legend(ax, **kwargs) -> None:
    """带 fontproperties 的图例设置。"""
    kwargs.setdefault("fontsize", 9)
    ax.legend(prop=_font_prop, **kwargs)


def assets_dir() -> str:
    """返回本目录下的 assets/ 目录路径, 不存在则创建。"""
    d = os.path.join(HERE, "assets")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name: str) -> str:
    """把 fig 存到 assets/<name>, 返回完整路径。供 plot 脚本 __main__ 使用。"""
    path = os.path.join(assets_dir(), name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def style_table(table, *, fontsize: int = 10, scale_x: float = 1.1, scale_y: float = 1.8,
                header_bg: str = "#2a2c33", cell_bg: str = "#1e1f25",
                header_text: str = "#ffffff", cell_text: str = "#dddddd",
                edge_color: str = "#444455") -> None:
    """让 matplotlib table 在深色主题下可读。

    默认 table 的 cell 背景是白色、文字继承 rcParams（浅色），
    在深色主题下会"白字白底"看不见。这个函数统一设置：
    - 表头：深色底 + 白色粗体字
    - 单元格：更深色底 + 浅灰字
    - 边框：中等灰色
    """
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(scale_x, scale_y)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color=header_text, fontweight="bold")
        else:
            cell.set_facecolor(cell_bg)
            cell.set_text_props(color=cell_text)
