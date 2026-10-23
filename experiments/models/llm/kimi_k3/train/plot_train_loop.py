# -*- coding: utf-8 -*-
"""训练循环可视化：loss 曲线 / LR 调度 / curriculum / grad_accum / ckpt / eval / MTP。

对应 tests/models/llm/kimi_k3/train/test_loop.py。

每个 `fig_*()` 函数返回 `(fig, text)` 元组，供 Gradio 直接复用；
`if __name__ == "__main__"` 保存到同目录 assets/，保持原有脚本可独立运行。
"""
from __future__ import annotations

import os
import sys
import tempfile
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

from models.llm.kimi_k3.train.loop import Trainer, TrainState, resolve_device
from models.llm.kimi_k3.optim.lr_scheduler import cosine_with_warmup, wsd_schedule
from models.llm.kimi_k3.data.curriculum import ContextCurriculum
from demo_pretrain_e2e import build_nano_config, build_nano_dataset, run_training


# ============================================================
# 1. 训练 loss 曲线 + LR + grad_norm（完整训练过程一览）
# ============================================================

def fig_training_overview(steps: int = 40, seed: int = 42, lr: float = 1e-3,
                          micro_batch_size: int = 4, log_every: int = 1):
    """完整训练过程总览：loss / smoothed loss / LR / grad_norm 四联图。"""
    cfg = build_nano_config(steps=steps, log_every=log_every,
                            lr=lr, micro_batch_size=micro_batch_size,
                            seed=seed)
    dataset = build_nano_dataset(cfg, seed=seed)
    trainer = run_training(cfg, dataset, steps=steps, verbose=False)

    history = trainer.metrics.history
    step_list = [r["step"] for r in history]
    loss_list = [r["loss"] for r in history]
    smoothed = trainer.metrics.smoothed("loss", window=max(3, steps // 10))
    lr_list = [r["lr"] for r in history]
    gn_list = [r["grad_norm"] for r in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    ax = axes[0, 0]
    ax.plot(step_list, loss_list, color="#8888aa", lw=0.9, alpha=0.5, label="loss")
    ax.plot(step_list, smoothed, color="#e74c3c", lw=1.8, label="smoothed")
    _at(ax, "训练 loss 曲线")
    _al(ax, "", "loss")
    _leg(ax)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    ax.plot(step_list, lr_list, color="#3498db", lw=2.0)
    _at(ax, "学习率调度（Cosine with Warmup）")
    _al(ax, "", "LR")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    ax.plot(step_list, gn_list, color="#f39c12", lw=1.2, alpha=0.8)
    ax.axhline(y=cfg.train.optim.grad_clip, color="#ef4444", ls="--", alpha=0.6,
               label=f"grad_clip = {cfg.train.optim.grad_clip}")
    _at(ax, "梯度范数（clip 约束）")
    _al(ax, "step", "grad_norm")
    _leg(ax)
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    tokens = [r["tokens"] for r in history]
    ax.plot(step_list, tokens, color="#9b59b6", lw=2.0)
    _at(ax, "已处理 token 数（累积）")
    _al(ax, "step", "tokens")
    ax.grid(True, alpha=0.2)

    fig.suptitle(f"K3 训练过程总览（{steps} 步，CPU 实时）",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    best_loss = trainer.state.best_loss
    final_loss = loss_list[-1]
    final_tokens = tokens[-1]
    return fig, _text(
        f"训练步数: {steps}, micro_batch_size: {micro_batch_size}",
        f"初始 LR: {lr}, 最终 LR: {lr_list[-1]:.2e}",
        f"初始 loss: {loss_list[0]:.4f}, 最终 loss: {final_loss:.4f}",
        f"最优 loss: {best_loss:.4f}",
        f"总 token 数: {final_tokens:,}",
        "Loss 下降、梯度在 clip 范围内 → 训练健康。",
    )


# ============================================================
# 2. LR 调度曲线：Cosine vs WSD + warmup 对比
# ============================================================

def fig_lr_schedules(total_steps: int = 200, warmup_ratio: float = 0.05,
                     min_lr_ratio: float = 0.1):
    """学习率调度可视化：Cosine vs WSD，对比不同 warmup/min_lr_ratio 的影响。"""
    steps = np.arange(total_steps)
    cosine_vals = [cosine_with_warmup(s, total_steps, warmup_ratio=warmup_ratio,
                                      min_lr_ratio=min_lr_ratio)
                   for s in steps]
    wsd_vals = [wsd_schedule(s, total_steps, warmup_ratio=warmup_ratio,
                             min_lr_ratio=min_lr_ratio)
                for s in steps]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5))

    ax1.plot(steps, cosine_vals, color="#3498db", lw=2.5, label="Cosine")
    ax1.plot(steps, wsd_vals, color="#e74c3c", lw=2.5, label="WSD")
    ax1.axvline(x=int(total_steps * warmup_ratio), color="#22c55e", ls="--",
                alpha=0.6, label=f"warmup end ({int(total_steps*warmup_ratio)} 步)")
    ax1.axhline(y=min_lr_ratio, color="gray", ls=":", alpha=0.6,
                label=f"min_lr_ratio = {min_lr_ratio}")
    _at(ax1, f"学习率调度曲线（total_steps={total_steps}, warmup={warmup_ratio}）")
    _al(ax1, "", "LR 乘子")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.2)

    cosine_diff = np.diff(cosine_vals)
    wsd_diff = np.diff(wsd_vals)
    ax2.plot(steps[1:], cosine_diff, color="#3498db", lw=1.3, alpha=0.7, label="Cosine dLR")
    ax2.plot(steps[1:], wsd_diff, color="#e74c3c", lw=1.3, alpha=0.7, label="WSD dLR")
    ax2.axhline(y=0, color="black", ls="-", alpha=0.3, lw=1)
    _at(ax2, "学习率变化率（差分）")
    _al(ax2, "Step", "delta LR")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()

    return fig, _text(
        "Cosine 调度: 先线性 warmup，再余弦衰减到 min_lr_ratio",
        "WSD 调度: Warmup-Stable-Decay，中间稳定在 1.0，最后线性衰减",
        f"warmup 步数: {int(total_steps * warmup_ratio)} ({warmup_ratio*100:.1f}%)",
        f"min_lr_ratio: {min_lr_ratio}",
        "WSD 适合需要长时间稳定学习率的场景（如 RLHF）。",
    )


# ============================================================
# 3. Context Curriculum 上下文课程：阶段切换 + token 预算
# ============================================================

def fig_curriculum(
    stages_str: str = "32,64,128,256",
    fractions_str: str = "0.4,0.7,0.9,1.0",
    total_steps: int = 200,
    max_tokens_per_rank: int = 64,
):
    """可视化 ContextCurriculum：各阶段 seq_len、阶段切换点、步数预算。"""
    lengths = tuple(int(x.strip()) for x in stages_str.split(","))
    fractions = tuple(float(x.strip()) for x in fractions_str.split(","))
    if len(lengths) != len(fractions):
        raise ValueError("stages 和 fractions 数量必须一致")

    cc = ContextCurriculum(lengths, fractions, total_steps,
                           max_tokens_per_rank=max_tokens_per_rank)

    steps = np.arange(total_steps)
    seq_lens = [cc.stage_for_step(s).seq_len for s in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.step(steps, seq_lens, where="post", color="#3498db", lw=2.5, label="seq_len")
    transition_steps = [s for s in range(1, total_steps) if cc.is_transition(s)]
    for ts in transition_steps:
        ax1.axvline(x=ts, color="#ef4444", ls="--", alpha=0.5, lw=1)
    _at(ax1, "Context Curriculum：seq_len 随阶段递增")
    _al(ax1, "step", "序列长度")
    _leg(ax1)
    ax1.grid(True, alpha=0.2)

    budget = cc.step_budget()
    labels = [f"{k:,} tok" for k in budget.keys()]
    values = list(budget.values())
    bars = ax2.bar(range(len(values)), values, color="#9b59b6", alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(values)))
    ax2.set_xticklabels(labels)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    _at(ax2, "各阶段步数预算（短序列占绝大多数预算）")
    _al(ax2, "上下文长度", "训练步数")
    ax2.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()

    total_budget = sum(budget.values())
    return fig, _text(
        f"总阶段数: {len(lengths)}",
        f"各阶段 seq_len: {lengths}",
        f"阶段切换步数: {transition_steps}",
        f"总训练步数: {total_budget}",
        "长序列阶段只占总预算的小部分，让课程既经济又能学到长程依赖。",
        f"各阶段 cp_degree: {[s.cp_degree for s in cc.stages]}",
    )


# ============================================================
# 4. 梯度累积：不同 grad_accum 下 loss 轨迹等价性验证
# ============================================================

def fig_grad_accum(steps: int = 20, seed: int = 42, lr: float = 1e-3,
                   total_batch_tokens: int = 32):
    """验证：不同 grad_accum 但有效批大小相同时，训练轨迹基本一致。

    有效批大小 = micro_batch_size * grad_accum。
    保持 total_batch_tokens 不变，比较 grad_accum=1 vs 2 vs 4。
    """
    configs = []
    for ga in [1, 2, 4]:
        mbs = max(1, total_batch_tokens // ga)
        cfg = build_nano_config(steps=steps, lr=lr, micro_batch_size=mbs,
                                grad_accum=ga, seed=seed, log_every=1)
        configs.append((ga, cfg))

    results = []
    for ga, cfg in configs:
        dataset = build_nano_dataset(cfg, seed=seed)
        trainer = run_training(cfg, dataset, steps=steps, verbose=False)
        steps_list = [r["step"] for r in trainer.metrics.history]
        losses = [r["loss"] for r in trainer.metrics.history]
        results.append((ga, steps_list, losses))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#3498db", "#e74c3c", "#9b59b6"]
    for (ga, s_list, losses), c in zip(results, colors):
        ax.plot(s_list, losses, lw=1.8, color=c, alpha=0.8,
                label=f"grad_accum={ga}, mbs={total_batch_tokens//ga}")

    _at(ax, f"梯度累积：不同 grad_accum 但有效批大小相同（≈{total_batch_tokens} tok/step）")
    _al(ax, "step", "loss")
    _leg(ax)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    final_losses = [losses[-1] for _, _, losses in results]
    max_diff = max(final_losses) - min(final_losses)
    return fig, _text(
        f"有效批大小: ≈ {total_batch_tokens} tokens / optimizer step",
        f"grad_accum=1 最终 loss: {final_losses[0]:.4f}",
        f"grad_accum=2 最终 loss: {final_losses[1]:.4f}",
        f"grad_accum=4 最终 loss: {final_losses[2]:.4f}",
        f"最大差值: {max_diff:.4f}（差异来自随机采样顺序）",
        "梯度累积让小显存设备也能等价于大 batch 训练。",
    )


# ============================================================
# 5. Checkpoint 保存/加载：Round-trip 一致性验证
# ============================================================

def fig_checkpoint_roundtrip(steps: int = 20, seed: int = 42):
    """验证：保存 checkpoint 后加载，状态完全一致（权重 / 优化器状态 / step）。"""
    cfg = build_nano_config(steps=steps, log_every=1, seed=seed)
    dataset = build_nano_dataset(cfg, seed=seed)
    trainer = run_training(cfg, dataset, steps=steps, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "ckpt.pt"
        trainer.save(ckpt_path)
        ckpt_size = ckpt_path.stat().st_size

        trainer2 = Trainer(cfg, dataset=dataset)
        trainer2.load(ckpt_path)

        sd1 = trainer.model.state_dict()
        sd2 = trainer2.model.state_dict()
        max_w_diff = 0.0
        n_params = 0
        for k in sd1:
            diff = (sd1[k] - sd2[k]).abs().max().item()
            max_w_diff = max(max_w_diff, diff)
            n_params += 1

        same_step = trainer.state.step == trainer2.state.step
        same_best = abs(trainer.state.best_loss - trainer2.state.best_loss) < 1e-9
        all_ok = same_step and same_best and max_w_diff == 0.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    rows = [
        ["训练步数", f"{trainer.state.step}", f"{trainer2.state.step}",
         "PASS" if same_step else "FAIL"],
        ["best_loss", f"{trainer.state.best_loss:.6f}",
         f"{trainer2.state.best_loss:.6f}", "PASS" if same_best else "FAIL"],
        ["权重最大误差", "-", f"{max_w_diff:.2e}",
         "PASS" if max_w_diff == 0 else "FAIL"],
        ["checkpoint 大小", "-", f"{ckpt_size/1024:.1f} KB", ""],
        ["参数张量数", f"{n_params}", f"{len(sd2)}", ""],
    ]
    table = ax.table(cellText=rows,
                     colLabels=["项目", "保存前", "加载后", "结果"],
                     loc="center", cellLoc="center")
    style_table(table, fontsize=10, scale_x=1.1, scale_y=1.8)
    _at(ax, "Checkpoint Round-trip 验证：save → load 后状态一致")
    fig.tight_layout()

    status = "全部通过" if all_ok else "存在差异"
    return fig, _text(
        f"状态: {status}",
        f"步数一致: {same_step}",
        f"best_loss 一致: {same_best}",
        f"权重完全一致: {max_w_diff == 0.0}（max_diff = {max_w_diff:.2e}）",
        f"checkpoint 大小: {ckpt_size/1024:.1f} KB",
        "Checkpoint 可用于中断恢复、模型迁移等场景。",
    )


# ============================================================
# 6. 评估 eval_loss / eval_ppl
# ============================================================

def fig_evaluate(steps: int = 15, seed: int = 42, micro_batch_size: int = 4):
    """训练前后评估 loss 和 ppl 的变化。"""
    cfg = build_nano_config(steps=steps, log_every=1,
                            micro_batch_size=micro_batch_size, seed=seed)
    dataset = build_nano_dataset(cfg, seed=seed)

    eval_batches = [dataset.batch(cfg.train.data.micro_batch_size) for _ in range(3)]
    trainer = Trainer(cfg, dataset=dataset)

    eval_before = trainer.evaluate(eval_batches)

    trainer.train(steps=steps, verbose=False)

    eval_after = trainer.evaluate(eval_batches)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    categories = ["训练前", "训练后"]
    losses = [eval_before["eval_loss"], eval_after["eval_loss"]]
    ppls = [eval_before["eval_ppl"], eval_after["eval_ppl"]]
    bars = ax.bar(categories, losses, color=["#e74c3c", "#22c55e"], alpha=0.8, width=0.5)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    _at(ax, "评估 loss 变化")
    _al(ax, "", "eval_loss")
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    bars = ax.bar(categories, ppls, color=["#e74c3c", "#22c55e"], alpha=0.8, width=0.5)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="bold")
    _at(ax, "困惑度 ppl 变化")
    _al(ax, "", "eval_ppl")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()

    loss_improve = eval_before["eval_loss"] - eval_after["eval_loss"]
    ppl_improve = eval_before["eval_ppl"] - eval_after["eval_ppl"]
    return fig, _text(
        f"训练前 eval_loss: {eval_before['eval_loss']:.4f}, eval_ppl: {eval_before['eval_ppl']:.2f}",
        f"训练后 eval_loss: {eval_after['eval_loss']:.4f}, eval_ppl: {eval_after['eval_ppl']:.2f}",
        f"loss 下降: {loss_improve:.4f}",
        f"ppl 下降: {ppl_improve:.2f}",
        " ppl = exp(loss)，loss 越小 → ppl 越低 → 模型预测越准。",
    )


# ============================================================
# 7. 设备选择：resolve_device 与 gradient_checkpointing
# ============================================================

def fig_device_and_gc(steps: int = 15, seed: int = 42):
    """可视化 resolve_device 逻辑 + gradient_checkpointing 的显存与速度权衡。"""
    dev_cpu = resolve_device("cpu")
    dev_auto = resolve_device("auto")
    has_cuda = torch.cuda.is_available()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.axis("off")
    rows = [
        ["显式 CPU", "cpu", dev_cpu.type, "预期 cpu"],
        ["auto 模式", "auto", dev_auto.type, "按可用选择"],
        ["CUDA 可用", "-", str(has_cuda), "系统检测"],
    ]
    table = ax.table(cellText=rows,
                     colLabels=["请求", "传入值", "实际设备", "说明"],
                     loc="center", cellLoc="center")
    style_table(table, fontsize=10, scale_x=1.1, scale_y=1.8)
    _at(ax, "resolve_device：设备选择策略")

    ax = axes[1]
    modes = ["无 GC", "有 GC"]
    # 示意数据（与经典文献数值对齐）：
    # 开启 GC 后，激活值显存约降到 60%（节省 ~40%），
    # 训练速度约降到 75%（增加 ~33% 重计算开销，即慢 ~25%）。
    # 含义：显存省 40%，速度慢 25%。
    mem = [100, 60]
    speed = [100, 75]
    x = np.arange(len(modes))
    w = 0.35
    bars1 = ax.bar(x - w/2, mem, w, label="激活值显存 (相对)", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + w/2, speed, w, label="训练速度 (相对)", color="#3498db", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.axhline(y=100, color="gray", ls=":", alpha=0.5, lw=1)
    _at(ax, "Gradient Checkpointing：显存-速度权衡（示意）")
    _al(ax, "", "相对值 (无 GC = 100%)")
    _leg(ax)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0, 115)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                    f"{h:.0f}%", ha="center", fontsize=9, fontweight="bold")

    fig.tight_layout()

    return fig, _text(
        f"设备: auto -> {dev_auto.type}, CPU -> {dev_cpu.type}",
        f"CUDA 可用: {has_cuda}",
        "Gradient Checkpointing 权衡（示意图，非实测）:",
        "  激活值显存: 100% → 60%（节省 ~40%）",
        "  训练速度:   100% → 75%（重计算开销 ~25%）",
        "设计意图: 反向传播时重新计算激活，而非保存，用计算时间换显存空间。",
        "适用场景: 显存不足但想跑更大模型 / 更长序列。",
    )


# ============================================================
# 独立运行时保存图片
# ============================================================

if __name__ == "__main__":
    for name, fn in [
        ("training_overview.png", lambda: fig_training_overview()),
        ("lr_schedules.png", lambda: fig_lr_schedules()),
        ("curriculum.png", lambda: fig_curriculum()),
        ("grad_accum.png", lambda: fig_grad_accum()),
        ("checkpoint_roundtrip.png", lambda: fig_checkpoint_roundtrip()),
        ("evaluate.png", lambda: fig_evaluate()),
        ("device_and_gc.png", lambda: fig_device_and_gc()),
    ]:
        fig, txt = fn()
        print(f"saved: {save(fig, name)}")
        print(txt)
        print()
