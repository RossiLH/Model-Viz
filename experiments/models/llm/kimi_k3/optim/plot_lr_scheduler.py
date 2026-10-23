# -*- coding: utf-8 -*-
"""学习率调度可视化：Cosine vs WSD，双 LR 调度。

对应 tests/models/llm/kimi_k3/optim/test_lr_scheduler.py。
"""
from __future__ import annotations

from plot_utils import add_src_to_path
add_src_to_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.optim.lr_scheduler import (
    cosine_with_warmup, wsd_schedule, CosineWarmupLR,
)


def fig_lr_schedules(total_steps=1000, warmup_ratio=0.05, min_lr_ratio=0.1,
                     decay_ratio=0.1):
    setup_cjk_font()
    fp = get_font_prop()

    steps = np.arange(total_steps)
    cosine_vals = [cosine_with_warmup(s, total_steps,
                                      warmup_ratio=warmup_ratio,
                                      min_lr_ratio=min_lr_ratio)
                   for s in steps]
    wsd_vals = [wsd_schedule(s, total_steps,
                             warmup_ratio=warmup_ratio,
                             decay_ratio=decay_ratio,
                             min_lr_ratio=min_lr_ratio)
                for s in steps]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=120)
    ax1, ax2 = axes

    ax1.plot(steps, cosine_vals, color="#3498db", lw=2.5, label="Cosine")
    ax1.plot(steps, wsd_vals, color="#e74c3c", lw=2.5, label="WSD (Warmup-Stable-Decay)")
    ax1.axvline(x=int(total_steps * warmup_ratio), color="green", ls="--", alpha=0.6,
                label=f"warmup end ({int(total_steps*warmup_ratio)} 步)")
    ax1.axhline(y=min_lr_ratio, color="gray", ls=":", alpha=0.6,
                label=f"min_lr_ratio = {min_lr_ratio}")
    _at(ax1, "学习率调度曲线：Cosine vs WSD")
    _al(ax1, "", "LR 乘子")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3)

    cosine_diff = np.diff(cosine_vals)
    wsd_diff = np.diff(wsd_vals)
    ax2.plot(steps[1:], cosine_diff, color="#3498db", lw=1.5, alpha=0.7,
             label="Cosine dLR")
    ax2.plot(steps[1:], wsd_diff, color="#e74c3c", lw=1.5, alpha=0.7,
             label="WSD dLR")
    ax2.axhline(y=0, color="black", ls="-", alpha=0.3, lw=1)
    _at(ax2, "学习率变化率 (差分)")
    _al(ax2, "Step", "delta LR")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Cosine vs WSD 学习率调度 (total_steps={total_steps}, warmup={warmup_ratio})",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        "Cosine 调度: 先线性 warmup，再余弦衰减到 min_lr_ratio",
        "WSD 调度: Warmup-Stable-Decay，中间稳定在 1.0，最后线性衰减",
        f"warmup 步数: {int(total_steps * warmup_ratio)} ({warmup_ratio*100:.1f}%)",
        f"min_lr_ratio: {min_lr_ratio}",
        "WSD 适合需要长时间稳定学习率的场景（如 RLHF）。",
    )


def fig_dual_lr(total_steps=500, warmup_steps=50, base_lr=3e-4, base_muon_lr=2e-3,
                min_lr=1e-5, min_muon_lr=2e-4):
    setup_cjk_font()
    fp = get_font_prop()

    from models.llm.kimi_k3.optim.muon import Muon
    from models.llm.kimi_k3.optim.hybrid import HybridMuonAdamW

    model = torch.nn.Linear(4, 4, bias=True)
    muon_opt = Muon(
        [{"params": [model.weight], "use_muon": True, "num_heads": None}],
        lr=base_muon_lr, adamw_lr=base_lr,
    )
    adamw_opt = torch.optim.AdamW(
        [{"params": [model.bias]}], lr=base_lr,
    )
    hybrid = HybridMuonAdamW(muon_opt, adamw_opt)
    scheduler = CosineWarmupLR(
        hybrid, total_steps=total_steps, warmup_steps=warmup_steps,
        min_lr=min_lr, min_muon_lr=min_muon_lr,
    )

    muon_lrs, adamw_lrs = [], []
    for _ in range(total_steps):
        scheduler.step()
        d = scheduler.get_lr_dict()
        muon_lrs.append(d["muon_lr"])
        adamw_lrs.append(d["adamw_lr"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    ax1.plot(muon_lrs, color="#9b59b6", lw=2.5, label="Muon LR")
    ax1.plot(adamw_lrs, color="#2ecc71", lw=2.5, label="AdamW LR")
    ax1.axvline(x=warmup_steps, color="red", ls="--", alpha=0.6,
                label=f"warmup end ({warmup_steps} 步)")
    _at(ax1, "双 LR 调度：Muon 与 AdamW 独立学习率")
    _al(ax1, "Step", "学习率")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3)

    ratio = [m / a for m, a in zip(muon_lrs, adamw_lrs)]
    ax2.plot(ratio, color="#e67e22", lw=2.5, label="Muon LR / AdamW LR")
    ax2.axhline(y=base_muon_lr / base_lr, color="red", ls="--", alpha=0.6,
                label=f"峰值比 = {base_muon_lr/base_lr:.1f}x")
    _at(ax2, "学习率比值 (Muon / AdamW)")
    _al(ax2, "Step", "比值")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, _text(
        f"Muon 峰值 LR: {base_muon_lr} (比 AdamW 大 {base_muon_lr/base_lr:.1f}x)",
        f"AdamW 峰值 LR: {base_lr}",
        f"warmup 步数: {warmup_steps}",
        f"最终 Muon LR: {muon_lrs[-1]:.2e}, AdamW LR: {adamw_lrs[-1]:.2e}",
        "Muon 通常需要更大的学习率，因为正交化后的更新方向范数更小。",
    )


if __name__ == "__main__":
    fig, txt = fig_lr_schedules()
    print("saved:", save(fig, "lr_schedules.png"))
    print(txt)
    print()

    fig, txt = fig_dual_lr()
    print("saved:", save(fig, "dual_lr.png"))
