# -*- coding: utf-8 -*-
"""PI0.5 忠实还原 —— 可视化 demo 的共享构件。

集中放「小型 CPU 可运行配置」与模型构造，避免各绘图脚本各写一份。
所有绘图脚本的数据都来自**真实跑本项目的模块**（非手工编造曲线），
这样图上的数字可以直接和文档正文对照。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from models.vla.pi05_faithful.config import Pi05FaithfulConfig  # noqa: E402
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def setup_fonts() -> str:
    """启用中文字体，并修掉数学负号（U+2212）的缺字形告警。

    setup_cjk_font() 把 sans-serif 设为 SimHei，但 SimHei **没有** U+2212
    字形，而对数轴刻度 10^-1 走 mathtext 路径，会刷一串缺字形告警。
    把 mathtext 单独指回 dejavusans 即可（中文仍由 SimHei 渲染）。
    """
    import matplotlib.pyplot as plt
    from plot_utils import setup_cjk_font
    name = setup_cjk_font()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    return name


def make_config(**kwargs) -> Pi05FaithfulConfig:
    """小型可 CPU 运行的 PI0.5 配置（以 small_scale 为基，可覆盖任意字段）。"""
    cfg = Pi05FaithfulConfig.use_small_scale()
    if kwargs:
        base = {f: getattr(cfg, f) for f in (
            "paligemma_width", "paligemma_depth", "action_width", "action_depth",
            "head_dim", "paligemma_num_heads", "paligemma_num_kv_heads",
            "action_num_heads", "action_num_kv_heads", "vocab_size",
            "image_size", "patch_size", "action_horizon", "action_dim",
            "num_inference_steps", "proprio_enabled", "fast_enabled",
            "rtc_max_delay", "n_action_steps",
        )}
        base.update(kwargs)
        cfg = Pi05FaithfulConfig(**base)
    return cfg


def build_model(**kwargs) -> Pi05FaithfulPolicyModel:
    """构造并置为 eval 模式的小型 PI0.5 策略模型。"""
    return Pi05FaithfulPolicyModel(make_config(**kwargs)).eval()


# 统一配色（与项目其他可视化保持一致）
C_STD = "#c62828"      # 标准基线 / 劣势侧（红）
C_NEW = "#238636"      # PI0.5 创新 / 优势侧（绿）
C_AUX = "#1565c0"      # 辅助（蓝）
C_AUX2 = "#6a1b9a"     # 辅助（紫）
C_WARN = "#ef8c00"     # 警示（橙）
