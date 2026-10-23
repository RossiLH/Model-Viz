# -*- coding: utf-8 -*-
"""Partial Rollout 相关可视化脚本。

包含：
  - Rollout 调度进度（active/paused/completed 堆叠面积图 + 新增完成数柱状图）
  - Staleness regularizer & PPO clipped surrogate loss 对比
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

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title, save

from models.llm.kimi_k3.posttrain.test_helpers import (
    simulate_rollout,
    regularizer_curves,
    ppo_clipped_curve,
)

setup_cjk_font()


def fig_rollout_schedule(n_prompts: int = 8, k_samples: int = 2,
                         lam: float = 0.5, max_iter: int = 15,
                         long_tail_ratio: float = 0.3,
                         seed: int = 42) -> tuple:
    """Rollout 调度进度可视化。

    左图：每轮 active / paused / completed 三条堆叠面积图
    右图：每轮新增完成数柱状图
    """
    font_prop = get_font_prop()
    result = simulate_rollout(n_prompts, k_samples, lam,
                              max_iter=max_iter,
                              long_tail_ratio=long_tail_ratio,
                              seed=seed)

    iters = np.arange(1, result.iterations + 1)
    active = [s["active"] for s in result.per_iter]
    paused = [s["paused"] for s in result.per_iter]
    completed = [s["completed"] for s in result.per_iter]
    new_done = [s["new_done"] for s in result.per_iter]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.stackplot(iters, active, paused, completed,
                  labels=["active", "paused", "completed"],
                  colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
                  alpha=0.7)
    ax1.set_xlabel("Iteration", fontproperties=font_prop)
    ax1.set_ylabel("Trajectory count", fontproperties=font_prop)
    ax1.legend(prop=font_prop, fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)
    _ax_title(ax1, "Rollout 调度进度 (堆叠面积图)")

    ax2.bar(iters, new_done, color="#9467bd", alpha=0.8, label="new completed")
    ax2.set_xlabel("Iteration", fontproperties=font_prop)
    ax2.set_ylabel("Newly completed", fontproperties=font_prop)
    ax2.legend(prop=font_prop, fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    _ax_title(ax2, "每轮新增完成轨迹数")

    fig.tight_layout()

    total_iters = result.iterations
    total_completed = result.completed
    final_paused = result.paused

    info = _text(
        f"总迭代数: {total_iters}",
        f"完成轨迹数: {total_completed}",
        f"最终暂停数: {final_paused}",
        f"总轨迹数: {n_prompts * k_samples}",
    )
    return fig, info


def fig_regularizer(epsilon: float = 0.2, advantage: float = 1.0,
                    max_ratio: float = 2.5) -> tuple:
    """Staleness regularizer & PPO clipped surrogate loss 对比。

    左图：staleness_regularizer raw vs clamped 曲线
    右图：PPO clipped surrogate loss (unclipped vs clipped)
    """
    font_prop = get_font_prop()
    ratios, raw, clamped = regularizer_curves(epsilon=epsilon, max_ratio=max_ratio)
    _, unclip, clip = ppo_clipped_curve(epsilon=epsilon, advantage=advantage,
                                        max_ratio=max_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(ratios, raw, lw=2, color="#1f77b4", label="raw |ratio - 1|")
    ax1.plot(ratios, clamped, lw=2, color="#d62728", linestyle="--",
             label=f"clamped (ε={epsilon})")
    ax1.axvline(1.0 + epsilon, color="red", linestyle=":", lw=1, alpha=0.6)
    ax1.axvline(1.0 - epsilon, color="red", linestyle=":", lw=1, alpha=0.6)
    ax1.axhline(epsilon, color="red", linestyle=":", lw=1, alpha=0.6)
    ax1.set_xlabel("Importance ratio", fontproperties=font_prop)
    ax1.set_ylabel("Staleness penalty", fontproperties=font_prop)
    ax1.legend(prop=font_prop, fontsize=9)
    ax1.grid(True, alpha=0.3)
    _ax_title(ax1, "Staleness Regularizer (raw vs clamped)")

    ax2.plot(ratios, unclip, lw=2, color="#2ca02c", label="unclipped loss")
    ax2.plot(ratios, clip, lw=2, color="#d62728", linestyle="--",
             label=f"clipped loss (ε={epsilon})")
    ax2.axvline(1.0 + epsilon, color="red", linestyle=":", lw=1, alpha=0.6)
    ax2.axvline(1.0 - epsilon, color="red", linestyle=":", lw=1, alpha=0.6)
    ax2.set_xlabel("Importance ratio", fontproperties=font_prop)
    ax2.set_ylabel("Surrogate loss", fontproperties=font_prop)
    ax2.legend(prop=font_prop, fontsize=9)
    ax2.grid(True, alpha=0.3)
    _ax_title(ax2, "PPO Clipped Surrogate Loss")

    fig.tight_layout()

    clamp_threshold = epsilon
    saturation_size = 2.0 * epsilon

    info = _text(
        f"clamp 阈值: {clamp_threshold:.3f}",
        f"饱和区间大小: {saturation_size:.3f} (±epsilon)",
        f"advantage: {advantage:.2f}",
        f"max_ratio: {max_ratio:.2f}",
    )
    return fig, info


if __name__ == "__main__":
    fig1, _ = fig_rollout_schedule()
    p1 = save(fig1, "rollout_schedule.png")
    print(f"saved: {p1}")

    fig2, _ = fig_regularizer()
    p2 = save(fig2, "rollout_regularizer.png")
    print(f"saved: {p2}")
