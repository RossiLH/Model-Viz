# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置 + 绘图共享工具, 供 vision 模块各绘图脚本复用。

与 serve 目录下的 plot_utils.py 保持一致接口。
"""
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


def setup_cjk_font() -> str:
    """启用中文字体(若系统有), 返回实际使用的字体名。"""
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
    """多行文本拼接, 返回换行符分隔的字符串 (供 gr.Markdown / gr.Textbox 使用)."""
    return "\n".join(lines)


def ax_title(ax, title: str, fontsize: int = 11) -> None:
    """带 fontproperties 的 ax.set_title(), 保证中文正常显示。"""
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=10,
                 fontproperties=_font_prop)
