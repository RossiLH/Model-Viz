# -*- coding: utf-8 -*-
"""训练指标可视化：RunningMean / MetricLog / expert_balance / decay_stats。

对应 tests/models/llm/kimi_k3/train/test_metrics.py。

每个 `fig_*()` 函数返回 `(fig, text)` 元组，供 Gradio 直接复用；
`if __name__ == "__main__"` 保存到同目录 assets/，保持原有脚本可独立运行。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_HERE = str(_THIS_FILE.parent)
_SRC = str(_THIS_FILE.parents[5] / "src")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, apply_dark_theme, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg,
    save, style_table,
)
setup_cjk_font()
apply_dark_theme()

from models.llm.kimi_k3.train.metrics import (
    RunningMean, MetricLog, expert_balance, decay_stats,
)
from demo_pretrain_e2e import build_nano_config, build_nano_dataset, run_training


# ============================================================
# 1. RunningMean 滑动窗口均值
# ============================================================

def fig_running_mean(window: int = 10, noise_scale: float = 0.3,
                     n_points: int = 100, seed: int = 42):
    """可视化 RunningMean 的滑动窗口平滑效果。

    生成一条带噪的递减曲线，比较原始值 vs 滑动均值。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.arange(n_points)
    raw = 2.0 * np.exp(-x / 30) + 0.5 + rng.normal(0, noise_scale, n_points)

    rm = RunningMean(window=window)
    smoothed = []
    for v in raw:
        rm.add(float(v))
        smoothed.append(rm.mean)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, raw, color="#8888aa", lw=0.8, alpha=0.6, label="原始值（带噪）")
    ax.plot(x, smoothed, color="#e74c3c", lw=2.0, label=f"滑动均值 (window={window})")
    ax.axvline(x=window, color="#3498db", ls="--", alpha=0.6, label=f"窗口填满点 (step={window})")
    _at(ax, f"RunningMean：滑动窗口均值（window={window}）")
    _al(ax, "step", "值")
    _leg(ax)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    first_full = smoothed[window]
    final = smoothed[-1]
    return fig, _text(
        f"窗口大小: {window}",
        f"窗口填满后(step={window})均值: {first_full:.4f}",
        f"最终均值: {final:.4f}",
        f"原始值方差: {raw.var():.4f} → 平滑后方差: {np.nanvar(smoothed):.4f}",
        "RunningMean 用于训练 loss 的实时平滑显示，抑制梯度噪声。",
    )


# ============================================================
# 2. MetricLog 记录与文件 I/O
# ============================================================

def fig_metric_log(total_steps: int = 50, log_every: int = 2, seed: int = 42):
    """可视化 MetricLog 的记录/系列提取/平滑能力，并验证 JSONL 写入。"""
    import json
    import tempfile

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ml = MetricLog()

    base_lr = 1e-3
    for i in range(total_steps):
        if i % log_every == 0:
            loss = 3.0 * np.exp(-i / max(1, total_steps // 3)) + 0.1 + rng.normal(0, 0.1)
            lr = base_lr * max(0.1, 1.0 - i / total_steps)
            ml.log(step=i, loss=float(loss), lr=lr, grad_norm=float(rng.uniform(0.5, 2.0)))

    steps = ml.series("step")
    losses = ml.series("loss")
    lrs = ml.series("lr")
    smoothed = ml.smoothed("loss", window=5)

    # 验证 JSONL 写入
    import json
    from pathlib import Path
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    ml2 = MetricLog(path=Path(tmp_path))
    for i in range(5):
        ml2.log(step=i, loss=float(i))
    ml2_lines = open(tmp_path).read().strip().split("\n")
    os.unlink(tmp_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(steps, losses, color="#8888aa", lw=0.9, alpha=0.5, label="loss (原始)")
    ax1.plot(steps, smoothed, color="#e74c3c", lw=1.8, label="loss (smoothed, w=5)")
    _at(ax1, f"MetricLog：loss 曲线与平滑 (log_every={log_every})")
    _al(ax1, "", "loss")
    _leg(ax1)
    ax1.grid(True, alpha=0.2)

    ax2.plot(steps, lrs, color="#3498db", lw=2.0, label="lr")
    ax2.set_ylabel("学习率", color="#3498db")
    ax2.tick_params(axis="y", labelcolor="#3498db")
    ax2_r = ax2.twinx()
    gn = [r["grad_norm"] for r in ml.history if "grad_norm" in r]
    ax2_r.plot(steps, gn, color="#f39c12", lw=1.2, alpha=0.7, label="grad_norm")
    ax2_r.set_ylabel("grad_norm", color="#f39c12")
    ax2_r.tick_params(axis="y", labelcolor="#f39c12")
    _at(ax2, "lr 与 grad_norm 双轴")
    _al(ax2, "step", "")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, prop=get_font_prop(), fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()

    return fig, _text(
        f"总步数: {total_steps}, 日志记录数: {len(ml.history)}",
        f"loss 范围: [{min(losses):.4f}, {max(losses):.4f}]",
        f"JSONL 写入验证: {len(ml2_lines)} 行（预期 5 行）",
        "MetricLog 支持内存历史 + JSONL 持久化，series()/smoothed() 方便后处理。",
    )


# ============================================================
# 3. MoE 专家负载均衡（训练中实时监测）
# ============================================================

def fig_expert_balance(steps: int = 30, seed: int = 42,
                       micro_batch_size: int = 4, top_k: int = 2):
    """可视化训练过程中 MoE 负载均衡指标的变化。

    验证 Quantile Balancing（router_biases 延迟更新）是否使负载趋向均匀。
    """
    cfg = build_nano_config(steps=steps, log_every=1,
                            micro_batch_size=micro_batch_size,
                            seed=seed)
    cfg.model.moe.top_k = top_k
    cfg.model.moe.qb_enabled = True
    dataset = build_nano_dataset(cfg, seed=seed)

    trainer = run_training(cfg, dataset, steps=steps, verbose=False)

    history = trainer.metrics.history
    steps_list = [r["step"] for r in history if "balance/max_over_mean" in r]
    max_over_mean = [r["balance/max_over_mean"] for r in history if "balance/max_over_mean" in r]
    cvs = [r["balance/mean_cv"] for r in history if "balance/mean_cv" in r]
    ratios = [r["balance/max_min_ratio"] for r in history if "balance/max_min_ratio" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(steps_list, max_over_mean, color="#e74c3c", lw=2.0, label="max/mean")
    ax1.axhline(y=1.0, color="#22c55e", ls="--", lw=1.8, alpha=0.8, label="理想值 = 1.0")
    _at(ax1, "MoE 负载均衡：max/mean（越小越均匀）")
    _al(ax1, "step", "max / mean")
    _leg(ax1, fontsize=10)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylim(bottom=0.95)

    ax2.plot(steps_list, cvs, color="#3498db", lw=2.0, label="变异系数 CV")
    ax2.plot(steps_list, ratios, color="#f39c12", lw=1.8, ls="--", label="max/min ratio")
    ax2.axhline(y=0, color="#22c55e", ls="--", alpha=0.5, lw=1.5)
    _at(ax2, "负载均衡：CV 与 max/min ratio")
    _al(ax2, "step", "值")
    ax2.legend(prop=get_font_prop(), fontsize=10)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()

    final_balance = max_over_mean[-1] if max_over_mean else float("nan")
    final_cv = cvs[-1] if cvs else float("nan")
    return fig, _text(
        f"训练步数: {steps}",
        f"Top-K: {top_k}, 专家数: {cfg.model.moe.n_routed}",
        f"最终 max/mean: {final_balance:.4f} （理想 ≈ 1.0）",
        f"最终 CV: {final_cv:.4f} （理想 → 0）",
        "Quantile Balancing 通过延迟一步更新 router bias，逐步把负载拉平。",
    )


# ============================================================
# 4. KDA alpha 衰减统计（训练中实时监测）
# ============================================================

def fig_decay_stats(steps: int = 30, seed: int = 42,
                    micro_batch_size: int = 4):
    """可视化训练过程中 KDA alpha 衰减因子的分布变化。

    验证 Eq. 5 的约束：alpha 始终在 (exp(g_min), 1) 范围内。
    """
    cfg = build_nano_config(steps=steps, log_every=1,
                            micro_batch_size=micro_batch_size,
                            seed=seed)
    dataset = build_nano_dataset(cfg, seed=seed)

    trainer = run_training(cfg, dataset, steps=steps, verbose=False)

    history = trainer.metrics.history
    steps_list = [r["step"] for r in history if "kda/alpha_min" in r]
    alpha_min = [r["kda/alpha_min"] for r in history if "kda/alpha_min" in r]
    alpha_max = [r["kda/alpha_max"] for r in history if "kda/alpha_max" in r]
    alpha_mean = [r["kda/alpha_mean"] for r in history if "kda/alpha_mean" in r]

    g_min = cfg.model.kda.g_min
    lo = float(np.exp(g_min))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.fill_between(steps_list, alpha_min, alpha_max, alpha=0.25,
                    color="#9b59b6", label="alpha 范围 (min ~ max)")
    ax.plot(steps_list, alpha_mean, color="#9b59b6", lw=2.0, label="alpha_mean")
    ax.axhline(y=lo, color="#ef4444", ls="--", alpha=0.7,
               label=f"下界 exp(g_min) = {lo:.4f}")
    ax.axhline(y=1.0, color="#22c55e", ls="--", alpha=0.7, label="上界 1.0")
    _at(ax, f"KDA alpha 衰减因子（g_min={g_min}）")
    _al(ax, "step", "alpha = exp(g)")
    _leg(ax)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    final_min = alpha_min[-1] if alpha_min else float("nan")
    final_max = alpha_max[-1] if alpha_max else float("nan")
    final_mean = alpha_mean[-1] if alpha_mean else float("nan")
    in_range = lo < final_min < final_max < 1.0
    return fig, _text(
        f"训练步数: {steps}",
        f"alpha 理论范围: [{lo:.4f}, 1.0]",
        f"最终 alpha_min: {final_min:.4f} (> {lo:.4f})",
        f"最终 alpha_max: {final_max:.4f} (< 1.0)",
        f"最终 alpha_mean: {final_mean:.4f}",
        "Eq. 5 保证 alpha 有界，避免训练中衰减因子爆炸或归零。",
        f"约束验证: {'通过' if in_range else '失败'}",
    )


# ============================================================
# 5. expert_balance 与 decay_stats 的无模型回退行为
# ============================================================

def fig_metrics_no_model():
    """验证 expert_balance / decay_stats 在无模型时返回空 dict 且不报错。"""
    # expert_balance 访问 model.model.routers()，需要两层结构
    dummy = torch.nn.Linear(4, 4)

    class FakeInner:
        def routers(self):
            return iter([])

    class FakeModel:
        def __init__(self):
            self.model = FakeInner()
            self._dummy = dummy
        def modules(self):
            return [dummy]

    bal = expert_balance(FakeModel())
    dec = decay_stats(FakeModel())

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    rows = [
        ["expert_balance(无router)", "{} (空 dict)", "无 MoE 层时优雅降级"],
        ["decay_stats(无KDA层)", "{} (空 dict)", "无 KDA 层时优雅降级"],
    ]
    table = ax.table(cellText=rows, colLabels=["函数", "返回值", "说明"],
                     loc="center", cellLoc="center")
    style_table(table, fontsize=10, scale_x=1.1, scale_y=1.8)
    _at(ax, "无模型 / 无 KDA/MoE 层时的回退行为")
    fig.tight_layout()

    ok = bal == {} and dec == {}
    return fig, _text(
        f"expert_balance 返回空 dict: {bal == {}}",
        f"decay_stats 返回空 dict: {dec == {}}",
        "两个函数都能优雅处理「模型不含对应模块」的情况，训练日志不会因缺少模块而报错。",
    )


# ============================================================
# 独立运行时保存图片
# ============================================================

if __name__ == "__main__":
    for name, fn in [
        ("running_mean.png", lambda: fig_running_mean()),
        ("metric_log.png", lambda: fig_metric_log()),
        ("expert_balance.png", lambda: fig_expert_balance()),
        ("decay_stats.png", lambda: fig_decay_stats()),
        ("metrics_no_model.png", lambda: fig_metrics_no_model()),
    ]:
        fig, txt = fn()
        print(f"saved: {save(fig, name)}")
        print(txt)
        print()
