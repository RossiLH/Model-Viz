# -*- coding: utf-8 -*-
"""matplotlib 中文字体配置，供各绘图脚本复用（与 experiments/linear_attn/plot_utils.py 一致）。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

_CJK_CANDIDATES = [
    "SimHei", "Microsoft YaHei", "SimSun", "DengXian",
    "Noto Sans CJK SC", "WenQuanYi Zen Hei",
]


def setup_cjk_font() -> str:
    """启用中文字体（若系统有），返回实际使用的字体名。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "default"
