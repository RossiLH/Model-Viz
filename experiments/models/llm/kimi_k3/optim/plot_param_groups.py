# -*- coding: utf-8 -*-
"""参数分组可视化：build_muon_param_groups 的分组结果。

展示参数分组逻辑如何把模型参数拆分为 Muon 组 + Per-Head 组 + AdamW 组。
对应 tests/models/llm/kimi_k3/optim/test_param_groups.py。
"""
from __future__ import annotations

from plot_utils import add_src_to_path
add_src_to_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.optim.param_groups import build_muon_param_groups


class _DemoModel(nn.Module):
    """用于演示参数分组的小型 Transformer 模型。"""

    def __init__(self, d_model=32, nhead=4, n_layers=2, vocab_size=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.num_heads = nhead  # 与 K3 KDA/MLA 层一致，_head_count 会读取 owner.num_heads
            layer.q_proj = nn.Linear(d_model, d_model, bias=False)
            layer.k_proj = nn.Linear(d_model, d_model, bias=False)
            layer.v_proj = nn.Linear(d_model, d_model, bias=False)
            layer.o_proj = nn.Linear(d_model, d_model, bias=False)
            layer.gate_proj = nn.Linear(d_model, d_model * 4, bias=False)
            layer.up_proj = nn.Linear(d_model, d_model * 4, bias=False)
            layer.down_proj = nn.Linear(d_model * 4, d_model, bias=False)
            layer.input_layernorm = nn.LayerNorm(d_model)
            layer.post_attention_layernorm = nn.LayerNorm(d_model)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)


def fig_param_groups(per_head=True, d_model=32, nhead=4):
    setup_cjk_font()
    fp = get_font_prop()
    model = _DemoModel(d_model=d_model, nhead=nhead)
    groups = build_muon_param_groups(model, per_head=per_head)

    muon_total = 0
    adamw_total = 0
    per_head_total = 0
    per_head_by_heads = {}
    muon_names = []
    adamw_names = []
    per_head_names = []

    for g in groups:
        for p in g["params"]:
            numel = p.numel()
            if g.get("use_muon", True):
                heads = g.get("num_heads")
                if heads and heads > 1:
                    per_head_total += numel
                    per_head_by_heads[heads] = per_head_by_heads.get(heads, 0) + numel
                    per_head_names.append(p.shape)
                else:
                    muon_total += numel
                    muon_names.append(p.shape)
            else:
                adamw_total += numel
                adamw_names.append(p.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
    ax_pie, ax_bar = axes

    sizes = [muon_total, per_head_total, adamw_total]
    labels = [
        f"普通 Muon\n({len(muon_names)} 个参数)",
        f"Per-Head Muon\n({len(per_head_names)} 个参数)",
        f"AdamW (1D/embedding/router)\n({len(adamw_names)} 个参数)",
    ]
    colors = ["#3498db", "#9b59b6", "#95a5a6"]
    if per_head_total == 0:
        sizes = [muon_total, adamw_total]
        labels = labels[0:1] + labels[2:3]
        colors = [colors[0], colors[2]]

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"fontproperties": fp, "fontsize": 9},
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    _at(ax_pie, "参数量分布 (按分组)")

    all_params = list(model.named_parameters())
    types_count = {
        "2D 矩阵 (Muon 候选)": 0,
        "Embedding (AdamW)": 0,
        "1D/LayerNorm (AdamW)": 0,
        "Bias (AdamW)": 0,
        "LM Head (AdamW)": 0,
    }
    for name, p in all_params:
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            types_count["2D 矩阵 (Muon 候选)"] += 1
        elif "embed" in name:
            types_count["Embedding (AdamW)"] += 1
        elif name.endswith("bias"):
            types_count["Bias (AdamW)"] += 1
        elif "norm" in name or p.ndim == 1:
            types_count["1D/LayerNorm (AdamW)"] += 1
        elif "lm_head" in name:
            types_count["LM Head (AdamW)"] += 1

    keys = list(types_count.keys())
    vals = list(types_count.values())
    bar_colors = ["#3498db", "#e67e22", "#95a5a6", "#f1c40f", "#2ecc71"]
    bars = ax_bar.barh(range(len(keys)), vals, color=bar_colors[:len(keys)], height=0.6)
    ax_bar.set_yticks(range(len(keys)))
    ax_bar.set_yticklabels(keys, fontproperties=fp, fontsize=9)
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=10, fontweight="bold",
                    fontproperties=fp)
    _at(ax_bar, "参数类型统计 (数量)")
    _al(ax_bar, "参数个数", "")
    ax_bar.grid(True, alpha=0.3, axis="x")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fig.suptitle(
        f"参数分组可视化 (总参数量: {total_params:,})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        f"总参数量: {total_params:,}",
        f"普通 Muon 组: {len(muon_names)} 个参数, {muon_total:,} 元素",
        f"Per-Head Muon 组: {len(per_head_names)} 个参数, {per_head_total:,} 元素",
        f"AdamW 组: {len(adamw_names)} 个参数, {adamw_total:,} 元素",
        "规则: 2D 矩阵 → Muon; q/k/v_proj → Per-Head; embedding/bias/norm/router → AdamW",
    )


if __name__ == "__main__":
    fig, txt = fig_param_groups(per_head=True)
    print("saved:", save(fig, "param_groups.png"))
    print(txt)
