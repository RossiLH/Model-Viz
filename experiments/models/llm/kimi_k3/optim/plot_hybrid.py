# -*- coding: utf-8 -*-
"""混合优化器可视化：HybridMuonAdamW 的事务性更新 + NaN 跳过。

对应 tests/models/llm/kimi_k3/optim/test_hybrid.py。
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
from models.llm.kimi_k3.optim.muon import Muon
from models.llm.kimi_k3.optim.hybrid import HybridMuonAdamW


def fig_hybrid_nan_skip(seed=42):
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(seed)

    model = nn.Linear(8, 4, bias=True)
    init_w = model.weight.data.clone()
    init_b = model.bias.data.clone()

    muon_opt = Muon(
        [{"params": [model.weight], "use_muon": True, "num_heads": None}],
        lr=0.01,
    )
    adamw_opt = torch.optim.AdamW(
        [{"params": [model.bias]}], lr=0.001, weight_decay=0.0,
    )
    hybrid = HybridMuonAdamW(muon_opt, adamw_opt)

    losses = []
    w_norms = []
    b_norms = []
    executed_steps = []
    nan_steps = [10, 20, 30]
    total_steps = 50

    for step in range(total_steps):
        hybrid.zero_grad()
        x = torch.randn(4, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()

        if step in nan_steps:
            model.weight.grad[:] = float("nan")

        report = hybrid.step()
        losses.append(float(loss))
        w_norms.append(float(model.weight.norm()))
        b_norms.append(float(model.bias.norm()))
        executed_steps.append(1.0 if report.executed else 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    ax_loss, ax_w, ax_b, ax_exec = axes.flat

    ax_loss.plot(losses, color="#e74c3c", lw=1.5, alpha=0.5)
    ax_loss.plot(_smooth(losses, 5), color="#e74c3c", lw=2.5, label="loss")
    for ns in nan_steps:
        ax_loss.axvline(x=ns, color="red", ls="--", alpha=0.7, lw=1.5)
    _at(ax_loss, f"Loss 曲线 (NaN 注入在 step {nan_steps})")
    _al(ax_loss, "", "Loss")
    ax_loss.grid(True, alpha=0.3)

    ax_w.plot(w_norms, color="#3498db", lw=2, label="weight norm")
    ax_w.axhline(y=float(init_w.norm()), color="red", ls="--", alpha=0.5,
                 label="初始权重范数")
    for ns in nan_steps:
        ax_w.axvline(x=ns, color="red", ls="--", alpha=0.7, lw=1.5)
    _at(ax_w, "权重范数变化")
    _al(ax_w, "", "范数")
    _leg(ax_w, fontsize=9)
    ax_w.grid(True, alpha=0.3)

    ax_b.plot(b_norms, color="#2ecc71", lw=2, label="bias norm")
    ax_b.axhline(y=float(init_b.norm()), color="red", ls="--", alpha=0.5,
                 label="初始 bias 范数")
    for ns in nan_steps:
        ax_b.axvline(x=ns, color="red", ls="--", alpha=0.7, lw=1.5)
    _at(ax_b, "Bias 范数变化")
    _al(ax_b, "", "范数")
    _leg(ax_b, fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # 用颜色区分执行/跳过，高度统一为 1（否则 0 高度的跳过柱看不见）
    ax_exec.bar(range(total_steps), [1.0] * total_steps,
                color=["#2ecc71" if e else "#e74c3c" for e in executed_steps],
                width=1.0)
    # 在跳过的柱子上标注 "SKIP"
    for i, e in enumerate(executed_steps):
        if not e:
            ax_exec.text(i, 0.5, "SKIP", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", rotation=90)
    ax_exec.set_ylim(0, 1.1)
    ax_exec.set_yticks([0, 1])
    ax_exec.set_yticklabels(["跳过", "执行"])
    _at(ax_exec, "步骤执行状态 (绿=执行, 红=跳过)")
    _al(ax_exec, "Step", "状态")
    ax_exec.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"HybridMuonAdamW 事务性更新：NaN 梯度时整步跳过",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    n_exec = sum(executed_steps)
    n_skip = total_steps - n_exec
    return fig, _text(
        f"总步数: {total_steps}, 执行: {int(n_exec)}, 跳过: {int(n_skip)}",
        f"NaN 注入步数: {nan_steps}",
        "事务性保证: 任何一个参数有 NaN 梯度，整步全部跳过",
        "防止 NaN 污染整个模型状态，是分布式训练的重要安全机制。",
    )


def fig_hybrid_state_dict():
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(42)

    model = nn.Linear(8, 4, bias=True)
    muon_opt = Muon(
        [{"params": [model.weight], "use_muon": True, "num_heads": None}],
        lr=0.01,
    )
    adamw_opt = torch.optim.AdamW(
        [{"params": [model.bias]}], lr=0.001, weight_decay=0.0,
    )
    hybrid = HybridMuonAdamW(muon_opt, adamw_opt)

    w_before = model.weight.data.clone()
    b_before = model.bias.data.clone()

    # 使用可重现的随机数据
    torch.manual_seed(42)
    data = [torch.randn(4, 8) for _ in range(10)]

    # 训练5步
    for i in range(5):
        hybrid.zero_grad()
        loss = (model(data[i]) ** 2).mean()
        loss.backward()
        hybrid.step()

    w1 = model.weight.data.clone()
    b1 = model.bias.data.clone()

    # 保存状态（包括模型权重）
    state = {
        "model": model.state_dict(),
        "optimizer": hybrid.state_dict(),
    }

    # 创建第二个模型，加载状态
    model2 = nn.Linear(8, 4, bias=True)
    muon_opt2 = Muon(
        [{"params": [model2.weight], "use_muon": True, "num_heads": None}],
        lr=0.01,
    )
    adamw_opt2 = torch.optim.AdamW(
        [{"params": [model2.bias]}], lr=0.001, weight_decay=0.0,
    )
    hybrid2 = HybridMuonAdamW(muon_opt2, adamw_opt2)
    model2.load_state_dict(state["model"])
    hybrid2.load_state_dict(state["optimizer"])

    w2_before = model2.weight.data.clone()

    # 用相同数据训练第二个模型5步
    for i in range(5):
        hybrid2.zero_grad()
        loss2 = (model2(data[i]) ** 2).mean()
        loss2.backward()
        hybrid2.step()

    # 继续训练第一个模型5步（使用相同数据）
    for i in range(5, 10):
        hybrid.zero_grad()
        loss = (model(data[i]) ** 2).mean()
        loss.backward()
        hybrid.step()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    diff_before = (w1 - w2_before).abs().numpy()
    diff_after = (model.weight.data - model2.weight.data).abs().numpy()

    # 使用共享的颜色范围，使左右图可直接对比
    vmax = max(diff_before.max(), diff_after.max())
    im1 = ax1.imshow(diff_before, cmap="hot", aspect="auto", vmin=0, vmax=vmax)
    _at(ax1, "加载前权重差异 (5 步 Muon vs 初始)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(diff_after, cmap="hot", aspect="auto", vmin=0, vmax=vmax)
    _at(ax2, "加载后再训练 5 步差异")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"state_dict 往返验证: 加载后状态一致 (max_diff = {diff_after.max():.2e})",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    max_diff_before = float(diff_before.max())
    max_diff_after = float(diff_after.max())
    return fig, _text(
        f"加载前 max |W1 - W2_before| = {max_diff_before:.2e} (预期不同)",
        f"加载后各训练 5 步 max |W1 - W2| = {max_diff_after:.2e}",
        "state_dict 包含 muon + adamw 状态 + step_number",
        "load_state_dict 完整恢复后，训练轨迹应完全一致。",
    )


def _smooth(vals, w=5):
    return [sum(vals[max(0, i - w):i + 1]) / min(i + 1, w)
            for i in range(len(vals))]


if __name__ == "__main__":
    fig, txt = fig_hybrid_nan_skip()
    print("saved:", save(fig, "hybrid_nan_skip.png"))
    print(txt)
    print()

    fig, txt = fig_hybrid_state_dict()
    print("saved:", save(fig, "hybrid_state_dict.png"))
