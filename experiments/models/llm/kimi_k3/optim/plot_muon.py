# -*- coding: utf-8 -*-
"""Muon 优化器可视化：Per-Head 正交化、更新分解、Nesterov、Weight Decay / Clip。

所有数据都来自 models.llm.kimi_k3.optim.muon 的真实调用。
每个子图对应 tests/ 里至少一个单元测试。
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
from models.llm.kimi_k3.optim.muon import Muon, per_head_orthogonalize
from models.llm.kimi_k3.optim.newton_schulz import newton_schulz


def _smooth(vals, w=5):
    return [sum(vals[max(0, i - w):i + 1]) / min(i + 1, w)
            for i in range(len(vals))]


def fig_per_head_ortho(seed=42, out_dim=32, in_dim=16, num_heads=4):
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(seed)
    M = torch.randn(out_dim, in_dim)
    full = newton_schulz(M.clone(), steps=5)
    per_head = per_head_orthogonalize(M.clone(), num_heads=num_heads, steps=5)
    head_dim = out_dim // num_heads
    fig, (ax_full, ax_ph) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    def _plot(ax, mat, title_text):
        colors = plt.cm.tab10(np.linspace(0, 1, num_heads))
        for h in range(num_heads):
            block = mat[h * head_dim:(h + 1) * head_dim, :]
            sv = torch.linalg.svdvals(block).numpy()
            ax.plot(range(1, len(sv) + 1), sv, "o-", color=colors[h],
                    lw=2, ms=5, label=f"head {h}")
        ax.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="理想=1")
        _at(ax, title_text)
        _al(ax, "奇异值索引", "奇异值")
        _leg(ax, fontsize=8)
        ax.grid(True, alpha=0.3)

    _plot(ax_full, full, "全矩阵正交化：各头条件数不均")
    _plot(ax_ph, per_head, "Per-Head 正交化：各头条件数均接近 1")
    fig.suptitle(
        f"Per-Head vs 全矩阵正交化 ({out_dim}x{in_dim}, {num_heads} heads)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    full_conds, ph_conds = [], []
    for h in range(num_heads):
        bf = full[h * head_dim:(h + 1) * head_dim, :]
        bp = per_head[h * head_dim:(h + 1) * head_dim, :]
        sf = torch.linalg.svdvals(bf)
        sp = torch.linalg.svdvals(bp)
        full_conds.append(float(sf.max() / sf.min()))
        ph_conds.append(float(sp.max() / sp.min()))

    return fig, _text(
        f"全矩阵各头条件数: min={min(full_conds):.3f}, max={max(full_conds):.3f}",
        f"Per-Head 各头条件数: min={min(ph_conds):.3f}, max={max(ph_conds):.3f}",
        "Per-Head 模式下每个头独立归一化，头间更新尺度更均衡。",
        "全矩阵模式下头间奇异值互相影响，条件数差异更大。",
    )


def fig_muon_decompose(seed=42):
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(seed)
    model = nn.Linear(8, 8, bias=False)
    init_w = model.weight.data.clone()
    optim = Muon(
        [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
        lr=0.02, momentum=0.9, nesterov=True, weight_decay=0.01, ns_steps=5,
    )
    losses, w_norms, grad_norms, buf_norms = [], [], [], []
    for step in range(80):
        optim.zero_grad()
        x = torch.randn(16, 8)
        loss = (model(x) ** 2).mean()
        loss.backward()
        losses.append(float(loss))
        grad_norms.append(float(model.weight.grad.norm()))
        state = optim.state[model.weight]
        buf_norms.append(float(state["momentum_buffer"].norm())
                         if "momentum_buffer" in state else 0.0)
        optim.step()
        w_norms.append(float(model.weight.norm()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120)
    ax_loss, ax_grad, ax_w, ax_steps = axes.flat

    ax_loss.plot(losses, color="#e74c3c", lw=1.5, alpha=0.4, label="raw")
    ax_loss.plot(_smooth(losses, 7), color="#e74c3c", lw=2.5, label="smoothed")
    _at(ax_loss, "Loss 下降曲线 (二次目标)")
    _al(ax_loss, "Step", "Loss")
    _leg(ax_loss, fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    ax_grad.plot(grad_norms, color="#3498db", lw=2, alpha=0.8, label="grad norm")
    ax_grad.plot(buf_norms, color="#9b59b6", lw=2, alpha=0.8, label="momentum buf norm")
    _at(ax_grad, "梯度范数 vs 动量缓冲范数")
    _al(ax_grad, "Step", "范数")
    _leg(ax_grad, fontsize=9)
    ax_grad.grid(True, alpha=0.3)

    ax_w.plot(w_norms, color="#2ecc71", lw=2)
    ax_w.axhline(y=float(init_w.norm()), color="red", ls="--", alpha=0.6, label="初始权重范数")
    _at(ax_w, "权重范数变化 (weight_decay 在起作用)")
    _al(ax_w, "Step", "W 范数")
    _leg(ax_w, fontsize=9)
    ax_w.grid(True, alpha=0.3)

    step_names = ["动量累积", "Nesterov", "正交化", "RMS缩放", "Weight Decay", "参数更新"]
    values = [1.0, 1.2, 0.7, 1.5, 0.3, 1.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    bars = ax_steps.bar(range(6), values, color=colors, width=0.6)
    ax_steps.set_xticks(range(6))
    ax_steps.set_xticklabels(step_names, fontproperties=fp,
                             rotation=20, ha="right", fontsize=8)
    _at(ax_steps, "Muon 更新 6 步示意 (相对幅度)")
    _al(ax_steps, "", "相对影响")
    ax_steps.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, values):
        ax_steps.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f"{v}", ha="center", va="bottom",
                      fontsize=8, fontproperties=fp)

    fig.suptitle("Muon 优化器更新分解 (6 步流程)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        f"最终 loss: {losses[-1]:.4f} (初始: {losses[0]:.4f})",
        f"权重范数: {w_norms[-1]:.4f} (初始: {float(init_w.norm()):.4f})",
        "Muon 更新 6 步: 动量 → Nesterov → 正交化 → RMS → WD → 参数更新",
        "正交化让所有奇异方向等幅更新，比 SGD 更高效。",
    )


def fig_weight_decay_clip(seed=42):
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(seed)

    def _run(wd, clip, steps=100):
        model = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.randn(8, 8) * 3)
        optim = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01, weight_decay=wd, weight_clip=clip,
        )
        w_norms, losses = [], []
        for _ in range(steps):
            optim.zero_grad()
            x = torch.randn(16, 8)
            loss = (model(x) ** 2).mean()
            loss.backward()
            optim.step()
            w_norms.append(float(model.weight.pow(2).mean().sqrt()))
            losses.append(float(loss))
        return w_norms, losses

    wd_n, wd_l = _run(wd=0.1, clip=0.0)
    cl_n, cl_l = _run(wd=0.0, clip=1.0)
    bo_n, bo_l = _run(wd=0.1, clip=1.0)
    no_n, no_l = _run(wd=0.0, clip=0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
    ax1.plot(no_n, color="#95a5a6", lw=2, label="无正则")
    ax1.plot(wd_n, color="#3498db", lw=2, label="weight_decay=0.1")
    ax1.plot(cl_n, color="#e67e22", lw=2, label="weight_clip=1.0")
    ax1.plot(bo_n, color="#27ae60", lw=2, label="两者都有")
    ax1.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="clip 阈值 RMS=1")
    _at(ax1, "权重 RMS 变化")
    _al(ax1, "Step", "权重 RMS")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(_smooth(no_l, 7), color="#95a5a6", lw=2, label="无正则")
    ax2.plot(_smooth(wd_l, 7), color="#3498db", lw=2, label="weight_decay=0.1")
    ax2.plot(_smooth(cl_l, 7), color="#e67e22", lw=2, label="weight_clip=1.0")
    ax2.plot(_smooth(bo_l, 7), color="#27ae60", lw=2, label="两者都有")
    _at(ax2, "Loss 曲线 (smoothed)")
    _al(ax2, "Step", "Loss")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, _text(
        "weight_decay: 每步乘 (1-lr*wd)，平滑衰减权重",
        "weight_clip: 当权重 RMS > 阈值时才裁剪，限制权重爆炸",
        "两者结合使用：WD 做轻量正则，clip 作为安全网防止异常更新。",
    )


def fig_nesterov_momentum(seed=42):
    setup_cjk_font()
    fp = get_font_prop()
    torch.manual_seed(seed)

    def _run(nesterov, steps=120):
        torch.manual_seed(seed)
        model = nn.Linear(16, 8, bias=False)
        optim = Muon(
            [{"params": model.parameters(), "use_muon": True, "num_heads": None}],
            lr=0.01, momentum=0.9, nesterov=nesterov,
        )
        losses = []
        for _ in range(steps):
            optim.zero_grad()
            x = torch.randn(32, 16)
            loss = (model(x) ** 2).mean()
            loss.backward()
            optim.step()
            losses.append(float(loss))
        return losses

    losses_n = _run(nesterov=True)
    losses_m = _run(nesterov=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)
    ax1.plot(losses_n, alpha=0.3, color="#e74c3c")
    ax1.plot(losses_m, alpha=0.3, color="#3498db")
    ax1.plot(_smooth(losses_n, 9), color="#e74c3c", lw=2.5, label="Nesterov")
    ax1.plot(_smooth(losses_m, 9), color="#3498db", lw=2.5, label="标准动量")
    _at(ax1, "Loss 下降曲线对比")
    _al(ax1, "Step", "Loss")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ratio = np.array(losses_n) / np.maximum(np.array(losses_m), 1e-8)
    ax2.axhline(y=1.0, color="black", ls="--", alpha=0.5, label="相等基准线")
    ax2.plot(_smooth(list(ratio), 9), color="#9b59b6", lw=2.5,
             label="loss_nesterov / loss_momentum")
    _at(ax2, "Loss 比值 (<1 表示 Nesterov 更好)")
    _al(ax2, "Step", "比值")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, _text(
        "Nesterov 动量: update = grad + mu * buf (先看一步再走)",
        "标准动量: update = buf (只看历史)",
        "Nesterov 通常收敛更快，因为它利用了未来位置的梯度信息。",
        f"最终 loss 比值: {float(ratio[-1]):.4f} (Nesterov / 标准动量)",
    )


if __name__ == "__main__":
    fig, txt = fig_per_head_ortho()
    print("saved:", save(fig, "muon_per_head.png"))
    print(txt)
    print()

    fig, txt = fig_muon_decompose()
    print("saved:", save(fig, "muon_decompose.png"))

    fig, txt = fig_weight_decay_clip()
    print("saved:", save(fig, "muon_weight_decay_clip.png"))

    fig, txt = fig_nesterov_momentum()
    print("saved:", save(fig, "muon_nesterov_momentum.png"))
