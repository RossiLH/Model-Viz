# -*- coding: utf-8 -*-
"""SFT 可视化：mask 分布、loss 计算。"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _ax_title, save,
)

from models.llm.kimi_k3.posttrain.test_helpers import (
    sample_conversation, sft_mask_stats,
)
from models.llm.kimi_k3.posttrain.sft import sft_loss


def fig_sft_mask(kind: str = "multi", thinking: bool = True,
                 mask_prompt: bool = True, max_length: int = 1000) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    conv = sample_conversation(kind)
    stats = sft_mask_stats(conv, thinking=thinking,
                           mask_prompt=mask_prompt, max_length=max_length)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120,
                                    gridspec_kw={"width_ratios": [1, 2]})

    # 左：饼图
    sizes = [stats.supervised, stats.masked]
    labels = ["supervised", "masked (-100)"]
    colors = ["#2ecc71", "#bdc3c7"]
    explode = (0.03, 0.03)
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, explode=explode, textprops={"fontproperties": fp},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    _ax_title(ax1, "Mask 比例")

    # 右：labels 向量彩色条
    labels_arr = stats.example.labels.numpy()
    n_tokens = len(labels_arr)
    display_n = min(n_tokens, 400)
    step = max(1, n_tokens // display_n)
    display_labels = labels_arr[::step]
    n_display = len(display_labels)

    ax2.set_xlim(0, n_display)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("token 位置（等比例采样）", fontproperties=fp)

    colors_arr = []
    for val in display_labels:
        if val == -100:
            colors_arr.append("#bdc3c7")
        else:
            hue = (val % 100) / 100.0
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue * 0.7, 0.6, 0.9)
            colors_arr.append((r, g, b))
    ax2.bar(range(n_display), [1] * n_display, color=colors_arr,
            edgecolor="none", linewidth=0, width=1)

    # 标注比例分割线位置
    first_supervised = -1
    for i, val in enumerate(labels_arr):
        if val != -100:
            first_supervised = i
            break
    if first_supervised > 0 and n_tokens > 0:
        frac = first_supervised / n_tokens
        ax2.axvline(x=frac * n_display, color="#e74c3c", linestyle="--",
                    linewidth=1.5, label=f"监督起始 @ {first_supervised}")
        ax2.legend(prop=fp, fontsize=9, loc="upper right")

    for spine in ["top", "right", "left"]:
        ax2.spines[spine].set_visible(False)
    _ax_title(ax2, f"Labels 向量可视化（共 {n_tokens} token）")

    fig.tight_layout()

    return fig, _text(
        f"总 token 数：{stats.total}",
        f"监督 token 数：{stats.supervised}",
        f"Masked token 数：{stats.masked}",
        f"监督比例：{stats.ratio:.2%}",
        f"mask_prompt：{mask_prompt}，thinking：{thinking}",
    )


def fig_sft_loss(batch_size: int = 4, seq_len: int = 32,
                 vocab_size: int = 128, seed: int = 42,
                 mask_ratio: float = 0.4) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    rng = torch.Generator().manual_seed(seed)
    np_rng = np.random.RandomState(seed)

    logits = torch.randn(batch_size, seq_len, vocab_size, generator=rng)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng)

    n_mask = max(1, int(seq_len * mask_ratio))
    valid_counts = []
    for i in range(batch_size):
        mask_pos = np_rng.choice(seq_len, size=n_mask, replace=False)
        for p in mask_pos:
            labels[i, p] = -100
        valid_counts.append(int((labels[i] != -100).sum().item()))

    loss = sft_loss(logits, labels).item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # 左：各样本有效 token 数柱状图
    sample_ids = [f"样本 {i + 1}" for i in range(batch_size)]
    colors = [f"C{i}" for i in range(batch_size)]
    bars = ax1.bar(range(batch_size), valid_counts, color=colors,
                   edgecolor="white", linewidth=1)
    for bar, v in zip(bars, valid_counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(v), ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ax1.set_ylabel("有效 token 数", fontproperties=fp)
    ax1.set_xticks(range(batch_size))
    ax1.set_xticklabels(sample_ids, fontproperties=fp)
    ax1.axhline(y=seq_len, color="#95a5a6", linestyle=":", linewidth=1,
                label=f"序列总长 {seq_len}")
    ax1.legend(prop=fp, fontsize=9)
    _ax_title(ax1, "各样本有效 token 数")

    # 右：logits 分布 + 总 loss 标注
    logits_flat = logits[:, :-1].reshape(-1).numpy()
    ax2.hist(logits_flat, bins=50, density=True, color="#3498db",
             alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(x=logits_flat.mean(), color="#e74c3c", linestyle="--",
                linewidth=2, label=f"均值 {logits_flat.mean():.3f}")
    ax2.set_xlabel("logit 值", fontproperties=fp)
    ax2.set_ylabel("密度", fontproperties=fp)
    ax2.legend(prop=fp, fontsize=9)

    total_valid = sum(valid_counts)
    actual_mask_ratio = 1 - total_valid / (batch_size * seq_len)
    info_text = (
        f"总 loss: {loss:.4f}\n"
        f"有效 token: {total_valid}\n"
        f"mask 比例: {actual_mask_ratio:.2%}"
    )
    ax2.text(0.97, 0.97, info_text, transform=ax2.transAxes,
             ha="right", va="top", fontsize=10, fontweight="bold",
             color="#2c3e50", fontproperties=fp,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa",
                       edgecolor="#bdc3c7"))
    _ax_title(ax2, "Logits 分布")

    fig.tight_layout()

    return fig, _text(
        f"loss 值：{loss:.6f}",
        f"有效 token 总数：{total_valid}",
        f"mask 比例：{actual_mask_ratio:.2%}",
        f"batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}",
    )


if __name__ == "__main__":
    fig, _ = fig_sft_mask(kind="multi", thinking=True, mask_prompt=True, max_length=1000)
    p1 = save(fig, "plot_sft_mask.png")
    print("saved:", p1)

    fig, _ = fig_sft_loss(batch_size=4, seq_len=32, vocab_size=128, seed=42, mask_ratio=0.4)
    p2 = save(fig, "plot_sft_loss.png")
    print("saved:", p2)
