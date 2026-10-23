# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置 + 绘图共享工具, 供 posttrain 各绘图脚本复用。

功能:
  1. ``setup_cjk_font()`` — 探测系统 CJK 字体并启用;
  2. ``text()`` — 多行文本拼接, 供 Gradio gr.Markdown / gr.Textbox 使用;
  3. ``ax_title()`` — 带 fontproperties 的标题设置;
  4. ``get_font_prop()`` — 返回当前激活的 FontProperties 对象;
  5. ``assets_dir()`` — 返回(并按需创建) 本目录下的 assets/ 目录。
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 候选字体按「常见程度」降序：中文 Windows 优先 SimHei/Microsoft YaHei，
# Linux 服务器常见 Noto Sans CJK SC / WenQuanYi Zen Hei。
_CJK_CANDIDATES = [
    "SimHei", "Microsoft YaHei", "SimSun", "DengXian",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei",
]

_font_prop: FontProperties | None = None

HERE = os.path.dirname(os.path.abspath(__file__))


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
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            _font_prop = FontProperties(family=name)
            return name
    plt.rcParams["axes.unicode_minus"] = False
    _font_prop = None
    return "default"


def get_font_prop() -> FontProperties | None:
    """返回 setup_cjk_font() 设置的 FontProperties, 供 ax.set_title() 等使用。"""
    return _font_prop


def text(*lines: str) -> str:
    """多行文本拼接, 返回换行符分隔的字符串 (供 Gradio 文本框使用)。"""
    return "\n".join(lines)


def ax_title(ax, title: str, fontsize: int = 11) -> None:
    """带 fontproperties 的 ax.set_title(), 保证中文正常显示。"""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=10,
                 fontproperties=_font_prop)


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
