# -*- coding: utf-8 -*-
"""预算控制可视化：悬崖曲线 + 努力退火。"""
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

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _ax_title, save,
)

from models.llm.kimi_k3.posttrain.test_helpers import (
    reward_vs_tokens_curve, budget_thresholds,
)


def fig_budget_cliff(
    problem_id: str = "math_easy",
    base_budget: int = 500,
    tau: float = 3.0,
    task_reward: float = 1.0,
    hard_cap: int = 0,
    count_mode: str = "thinking",
) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    _hard_cap = hard_cap if hard_cap > 0 else None
    tokens, rewards, exceeded = reward_vs_tokens_curve(
        problem_id, base_budget, tau,
        task_reward=task_reward, hard_cap=_hard_cap,
        count_mode=count_mode,
    )

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.plot(tokens, rewards, "b-o", markersize=4, linewidth=2, label="reward")

    threshold = tau * base_budget
    if _hard_cap is not None:
        threshold = min(threshold, _hard_cap)
    ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"阈值 = {threshold:.0f}")

    ax.set_xlabel("Token 数", fontproperties=fp)
    ax.set_ylabel("Reward", fontproperties=fp)
    _ax_title(ax, f"预算悬崖曲线 (tau={tau}, base_budget={base_budget})")
    ax.legend(prop=fp)
    ax.grid(True, alpha=0.3)

    cliff_idx = None
    for i, ex in enumerate(exceeded):
        if ex:
            cliff_idx = i
            break
    cliff_point = tokens[cliff_idx] if cliff_idx is not None else None

    fig.tight_layout()
    return fig, _text(
        f"**阈值位置**：{threshold:.0f} tokens",
        f"**悬崖跳变点**：tokens={cliff_point}（reward 从 {task_reward} 跳变至 -1）",
        f"**hard_cap**：{'无' if _hard_cap is None else _hard_cap}",
        f"**count_mode**：{count_mode}",
    )


def fig_effort_annealing(
    tau_max: float = 8.0,
    tau_high: float = 3.0,
    tau_low: float = 1.0,
    n_problems: int = 5,
) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    problem_ids = [f"prob_{i}" for i in range(n_problems)]
    base_budgets_list = np.logspace(2, 4, n_problems).astype(int).tolist()
    base_budgets = dict(zip(problem_ids, base_budgets_list))

    tau_values = [tau_max, tau_high, tau_low]
    tau_labels = ["max", "high", "low"]
    thresholds = budget_thresholds(base_budgets, tau_values)

    matrix = np.array([thresholds[pid] for pid in problem_ids]).T  # (3, n_problems)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    for i, (label, color) in enumerate(zip(tau_labels, colors)):
        ax1.plot(base_budgets_list, matrix[i], "o-", color=color,
                 linewidth=2, markersize=6, label=f"tau={tau_values[i]:.1f} ({label})")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("问题难度 (base_budget)", fontproperties=fp)
    ax1.set_ylabel("阈值 (tokens)", fontproperties=fp)
    _ax_title(ax1, "三阶段阈值随问题难度变化")
    ax1.legend(prop=fp, fontsize=9)
    ax1.grid(True, alpha=0.3, which="both")

    im = ax2.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax2.set_xticks(range(n_problems))
    ax2.set_xticklabels([f"P{i+1}\n({b})" for i, b in enumerate(base_budgets_list)],
                        fontproperties=fp, fontsize=8)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([f"{l}\n(tau={t:.1f})" for l, t in zip(tau_labels, tau_values)],
                        fontproperties=fp, fontsize=9)
    _ax_title(ax2, "阈值热力图 (行=effort阶段, 列=问题)")
    for i in range(3):
        for j in range(n_problems):
            ax2.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                     fontsize=8, color="black" if matrix[i, j] < matrix.max() * 0.6 else "white")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    monotonic = bool(np.all(np.diff(matrix, axis=1) >= 0))
    range_per_stage = [
        f"{tau_labels[i]}: [{matrix[i].min():.0f}, {matrix[i].max():.0f}]"
        for i in range(3)
    ]

    fig.tight_layout()
    return fig, _text(
        "**三阶段阈值范围**：",
        *range_per_stage,
        f"**单调性验证**：{'通过（阈值随难度递增）' if monotonic else '未通过'}",
        f"**问题数量**：{n_problems}，base_budget 范围 [{base_budgets_list[0]}, {base_budgets_list[-1]}]",
    )


if __name__ == "__main__":
    fig1, _ = fig_budget_cliff()
    save(fig1, "budget_cliff.png")

    fig2, _ = fig_effort_annealing()
    save(fig2, "effort_annealing.png")
