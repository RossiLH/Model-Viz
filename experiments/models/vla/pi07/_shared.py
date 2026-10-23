# -*- coding: utf-8 -*-
"""PI0.7 忠实还原 —— 可视化 demo 的共享构件。

集中放「小型 CPU 可运行配置」与模型构造，避免各绘图脚本各写一份。
所有绘图脚本的数据都来自**真实跑本项目的模块**（非手工编造曲线），
这样图上的数字可以直接和文档正文对照。

与 pi05_faithful/_shared.py 的分工：PI0.5 与 PI0.7 的配置字段名不同
（PI0.7 是嵌套的 vision/action/rtc/metadata/subgoal/cfg_guidance dataclass），
故各自维护一份共享构件。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from models.vla.pi07.config import (  # noqa: E402
    Pi07Config, Pi07VisionConfig, Pi07ActionConfig, Pi07RTCConfig,
    Pi07MetadataConfig, Pi07SubgoalConfig, Pi07CFGConfig,
)
from models.vla.pi07.model import Pi07PolicyModel  # noqa: E402

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


def make_config(**kwargs) -> Pi07Config:
    """小型可 CPU 运行的 PI0.7 配置，可覆盖任意字段（按 dataclass 分段展开）。

    说明：为体现 π0.7 的四个新增块（video 时空压缩 / block-causal /
    RTC / metadata+CFG），这里特意把关键规模压小但保留所有机制路径：
    - 视频：28x28 图、patch 7 → 每帧 16 patch，3 帧历史；
    - 动作 chunk：H=8（论文 H=50 的 1/6 量级），便于快速跑出对照曲线；
    - 去噪步：4 步（论文 5 步）。
    """
    vision = dict(image_size=28, patch_size=7, dim=32, n_layers=1, n_heads=4,
                  n_obs_steps=3, temporal_layers=1, temporal_heads=2)
    action = dict(width=32, depth=1, n_heads=4, n_kv_heads=1, head_dim=8,
                  mlp_dim=64)
    rtc = dict(enabled=True, max_delay=4, n_action_steps=4,
               per_token_timestep=True)
    metadata = dict(drop_prob=0.15, per_field_drop_prob=0.05,
                    speed_bin_size=500, quality_min=1, quality_max=5)
    subgoal = dict(keep_prob=0.25, subtask_drop_prob_given_subgoal=0.30)
    cfg_guidance = dict(enabled=False, beta=1.7, beta_choices=(1.3, 1.7, 2.2))

    # 把嵌套段按前缀展开（"vision_image_size" -> vision["image_size"]），
    # 以便调用方用扁平关键字覆盖任意字段。先确保各段 dict 已初始化。
    for seg in ("vision", "action", "rtc", "metadata", "subgoal",
                "cfg_guidance"):
        kwargs.setdefault(seg, {})
    for k in list(kwargs):
        for seg in ("vision", "action", "rtc", "metadata", "subgoal",
                    "cfg_guidance"):
            if k.startswith(seg + "_"):
                kwargs[seg][k[len(seg) + 1:]] = kwargs.pop(k)

    cfg = Pi07Config(
        vision=Pi07VisionConfig(**{**vision, **kwargs.pop("vision", {})}),
        action=Pi07ActionConfig(**{**action, **kwargs.pop("action", {})}),
        rtc=Pi07RTCConfig(**{**rtc, **kwargs.pop("rtc", {})}),
        metadata=Pi07MetadataConfig(**{**metadata, **kwargs.pop("metadata", {})}),
        subgoal=Pi07SubgoalConfig(**{**subgoal, **kwargs.pop("subgoal", {})}),
        cfg_guidance=Pi07CFGConfig(**{**cfg_guidance, **kwargs.pop("cfg_guidance", {})}),
        action_horizon=kwargs.pop("action_horizon", 8),
        action_dim=kwargs.pop("action_dim", 8),
        state_dim=kwargs.pop("state_dim", 8),
        num_inference_steps=kwargs.pop("num_inference_steps", 4),
        prompt_max_length=kwargs.pop("prompt_max_length", 16),
        **kwargs,
    )
    return cfg


def build_model(**kwargs) -> Pi07PolicyModel:
    """构造并置为 eval 模式的小型 PI0.7 策略模型。"""
    return Pi07PolicyModel(make_config(**kwargs)).eval()


# 统一配色（与项目其他可视化保持一致）
C_STD = "#c62828"      # 标准基线 / 劣势侧（红）
C_NEW = "#238636"      # PI0.7 创新 / 优势侧（绿）
C_AUX = "#1565c0"      # 辅助（蓝）
C_AUX2 = "#6a1b9a"     # 辅助（紫）
C_WARN = "#ef8c00"     # 警示（橙）
