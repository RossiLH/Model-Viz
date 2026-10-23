# -*- coding: utf-8 -*-
"""DeepSeek-V4 可视化 demo 的共享构件（C9）。

集中放「小型 CPU 可运行配置」与模型构造，避免 4 个绘图脚本各写一份。
配置与 scripts/deepseek_v4/09_probe_cache_generate.py 保持一致，
这样图上的数字可以直接和探针输出对照。
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from models.llm.deepseek_v4.config import (  # noqa: E402
    DeepSeekV4ModelConfig, HCAConfig, CSAConfig, MHCConfig, MoEV4Config,
    MQAV4Config, MTPConfig,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def setup_fonts() -> str:
    """启用中文字体，并修掉对数坐标轴的负号告警。

    setup_cjk_font() 会把 sans-serif 设为 SimHei，但 SimHei **没有** U+2212
    （数学负号）字形，而对数轴刻度 10^-1 走的是 mathtext 路径，于是每张图都会
    刷一串 "Font 'default' does not have a glyph for '\\u2212'" 告警。
    把 mathtext 单独指回 dejavusans 即可（中文仍由 SimHei 渲染）。
    """
    import matplotlib.pyplot as plt
    from plot_utils import setup_cjk_font
    name = setup_cjk_font()
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["mathtext.default"] = "regular"
    return name


def make_config(**kwargs) -> DeepSeekV4ModelConfig:
    """小型可 CPU 运行的 V4 配置（与探针 09 一致）。"""
    base = dict(
        dim=64, n_layers=6, n_heads=4, head_dim=16, q_lora_rank=32,
        qk_rope_head_dim=8, window_size=8, vocab_size=1000, max_seq_len=512,
        compress_ratios=[0, 0, 4, 128, 4, 128],
        mqa=MQAV4Config(dim=64, n_heads=4, head_dim=16, q_lora_rank=32,
                        qk_rope_head_dim=8, window_size=8, max_seq_len=512),
        hca=HCAConfig(compression_factor=4),
        csa=CSAConfig(compression_factor=4),
        mhc=MHCConfig(n_hc=2, sinkhorn_iters=10),
        moe=MoEV4Config(n_routed_experts=8, n_shared_experts=1,
                        num_experts_per_tok=2, moe_intermediate_size=64,
                        num_hash_layers=2),
        mtp=MTPConfig(num_nextn_predict_layers=2),
    )
    base.update(kwargs)
    cfg = DeepSeekV4ModelConfig(**base)
    cfg.validate()
    return cfg


def build_model(**kwargs) -> DeepSeekV4LM:
    """构造并置为 eval 模式的小型 V4 模型。"""
    return DeepSeekV4LM(make_config(**kwargs)).eval()
