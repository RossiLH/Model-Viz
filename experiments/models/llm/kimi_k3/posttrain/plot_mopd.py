# -*- coding: utf-8 -*-
"""MOPD 相关可视化脚本。

包含：
  - OPD 裁剪奖励曲线
  - 9 教师（3 domain × 3 effort）热力图 + 选中教师 per-token 奖励分布
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title, save

from models.llm.kimi_k3.posttrain.test_helpers import opd_reward_curve
from models.llm.kimi_k3.posttrain.rl.mopd import multi_teacher_reward


DOMAINS = ("general", "general_agent", "coding_agent")
EFFORTS = ("low", "high", "max")

setup_cjk_font()


def fig_opd_reward(r_max: float = 4.0, min_gap: float = -8.0,
                   max_gap: float = 8.0) -> tuple:
    """绘制 OPD 裁剪奖励曲线。

    x = teacher-student log-prob gap, y = reward。
    |gap| < r_max 时线性（y=x），两侧被裁剪到 ±r_max。
    """
    font_prop = get_font_prop()
    gaps, rewards = opd_reward_curve(r_max=r_max, min_gap=min_gap, max_gap=max_gap)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(gaps, rewards, lw=2, color="#1f77b4", label="OPD reward")
    ax.axvline(r_max, color="red", linestyle="--", lw=1.2, label=f"±r_max = {r_max}")
    ax.axvline(-r_max, color="red", linestyle="--", lw=1.2)
    ax.axhline(r_max, color="red", linestyle=":", lw=0.8, alpha=0.5)
    ax.axhline(-r_max, color="red", linestyle=":", lw=0.8, alpha=0.5)
    ax.axline((0, 0), slope=1, color="gray", linestyle=":", lw=0.8, alpha=0.5, label="y=x (linear)")
    ax.set_xlabel("Teacher-Student log-prob gap", fontproperties=font_prop)
    ax.set_ylabel("Reward", fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(prop=font_prop, fontsize=9)
    _ax_title(ax, "OPD 裁剪奖励曲线 (Clipped Reward)")

    clip_range = (-r_max, r_max)
    linear_slope = 1.0

    info = _text(
        f"裁剪范围: [{-r_max:.2f}, {r_max:.2f}]",
        f"线性区间斜率: {linear_slope:.2f}",
        f"gap 范围: [{min_gap:.1f}, {max_gap:.1f}]",
    )
    return fig, info


def fig_multi_teacher(seed: int = 42, seq_len: int = 20,
                      selected_domain: str = "general",
                      selected_effort: str = "high") -> tuple:
    """9 教师（3 domain × 3 effort）热力图 + 选中教师 per-token 奖励分布。"""
    font_prop = get_font_prop()
    rng = np.random.default_rng(seed)

    student_logprobs = torch.tensor(rng.normal(0, 1, (1, seq_len)), dtype=torch.float32)

    teacher_logprobs: dict[tuple[str, str], torch.Tensor] = {}
    reward_means = np.zeros((len(DOMAINS), len(EFFORTS)))
    for i, d in enumerate(DOMAINS):
        for j, e in enumerate(EFFORTS):
            base_gap = (i - 1) * 1.5 + (j - 1) * 2.0
            gap = rng.normal(base_gap, 0.8, (1, seq_len))
            teacher_logprobs[(d, e)] = student_logprobs + torch.tensor(gap, dtype=torch.float32)
            reward = multi_teacher_reward(teacher_logprobs, student_logprobs, d, e)
            reward_means[i, j] = reward.mean().item()

    sel_reward = multi_teacher_reward(teacher_logprobs, student_logprobs,
                                      selected_domain, selected_effort)
    sel_reward_np = sel_reward[0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    im = ax1.imshow(reward_means, cmap="RdYlGn", aspect="auto", vmin=reward_means.min(), vmax=reward_means.max())
    ax1.set_xticks(range(len(EFFORTS)))
    ax1.set_xticklabels(EFFORTS, fontproperties=font_prop)
    ax1.set_yticks(range(len(DOMAINS)))
    ax1.set_yticklabels(DOMAINS, fontproperties=font_prop, fontsize=8)
    for i in range(len(DOMAINS)):
        for j in range(len(EFFORTS)):
            ax1.text(j, i, f"{reward_means[i, j]:.2f}", ha="center", va="center",
                     fontsize=9, fontproperties=font_prop,
                     color="black" if abs(reward_means[i, j]) < 2 else "white")
    di = DOMAINS.index(selected_domain)
    ei = EFFORTS.index(selected_effort)
    rect = plt.Rectangle((ei - 0.5, di - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5)
    ax1.add_patch(rect)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    _ax_title(ax1, "9 教师平均奖励热力图 (3 domain × 3 effort)")

    tokens = np.arange(seq_len)
    ax2.bar(tokens, sel_reward_np, color="#2ca02c", alpha=0.7, label="per-token reward")
    ax2.axhline(sel_reward_np.mean(), color="red", linestyle="--", lw=1.2,
                label=f"mean = {sel_reward_np.mean():.3f}")
    ax2.set_xlabel("Token position", fontproperties=font_prop)
    ax2.set_ylabel("Reward", fontproperties=font_prop)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(prop=font_prop, fontsize=9)
    _ax_title(ax2, f"选中教师 per-token 奖励分布 ({selected_domain}/{selected_effort})")

    fig.tight_layout()

    sel_mean = float(sel_reward_np.mean())
    r_min = float(reward_means.min())
    r_max = float(reward_means.max())

    info = _text(
        f"选中教师奖励均值: {sel_mean:.4f}",
        f"9 教师奖励范围: [{r_min:.4f}, {r_max:.4f}]",
        f"选中 domain/effort: {selected_domain} / {selected_effort}",
    )
    return fig, info


if __name__ == "__main__":
    fig1, _ = fig_opd_reward()
    p1 = save(fig1, "mopd_opd_reward.png")
    print(f"saved: {p1}")

    fig2, _ = fig_multi_teacher()
    p2 = save(fig2, "mopd_multi_teacher.png")
    print(f"saved: {p2}")
