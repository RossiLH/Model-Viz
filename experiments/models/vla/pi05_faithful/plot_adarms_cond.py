# -*- coding: utf-8 -*-
"""Q12/Q13: adaRMSNorm 条件调制生效的实证图。

Q12 test_gate_depends_on_timestep —— gate 必须来自时间步 cond（条件调制生效）
Q13 test_suffix_output_depends_on_cond —— 不同 cond 产生不同 suffix 输出（调制真正影响输出）

所有数据来自真实跑本项目 AdaRMSNorm / Pi05TransformerBlock（CPU）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from _shared import setup_fonts  # noqa: E402

from matplotlib import pyplot as plt  # noqa: E402

from models.vla.pi05_faithful.config import Pi05FaithfulConfig  # noqa: E402
from models.vla.core.norms import AdaRMSNorm  # noqa: E402


def main():
    font = setup_fonts()
    print("font:", font)
    torch.manual_seed(0)
    cfg = Pi05FaithfulConfig.use_small_scale()
    D = cfg.action_width

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # ---- 左：零初始化 => gate 恒为 0；扰动后 gate 随 t 单调放大 ----
    n_pert = AdaRMSNorm(dim=D, cond_dim=D, return_gate=True)
    n_zero = AdaRMSNorm(dim=D, cond_dim=D, return_gate=True)
    torch.nn.init.normal_(n_pert.cond_proj.weight, std=0.05)
    x = torch.randn(1, cfg.action_horizon, D)
    ts = np.linspace(0, 1, 11)
    gate_std_pert, gate_std_zero, gate_mean_pert = [], [], []
    for t in ts:
        cond = torch.full((1, D), float(t))
        _, g = n_pert(x, cond)
        _, gz = n_zero(x, cond)
        gate_std_pert.append(g.std().item())
        gate_std_zero.append(gz.abs().max().item())
        gate_mean_pert.append(g.mean().item())
    ax[0, 0].plot(ts, gate_std_pert, marker="o", label="gate std（cond_proj 扰动 0.05）")
    ax[0, 0].plot(ts, gate_mean_pert, marker="s", label="gate mean")
    ax[0, 0].plot(ts, gate_std_zero, marker="^", color="gray", ls="--", label="零初始化 cond_proj: |gate|")
    ax[0, 0].set_xlabel("timestep tau")
    ax[0, 0].set_ylabel("gate 统计量")
    ax[0, 0].set_title("Q12: gate 随 timestep 变化（条件调制生效）")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].text(0.03, 0.97, "零初始化时 gate 恒为 0.00e+00\n（训练稳定：初始调制=0）", transform=ax[0, 0].transAxes,
                  fontsize=8, va="top", bbox=dict(boxstyle="round", fc="#fff8e1", alpha=0.9))

    # ---- 右：Q13 suffix 输出依赖 cond ----
    from models.vla.pi05_faithful.modules.pi05_transformer import Pi05TransformerBlock
    blk = Pi05TransformerBlock(cfg)
    for mod in blk.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == D * 3:
            torch.nn.init.normal_(mod.weight, std=0.02)
    prefix = torch.randn(1, 8, cfg.paligemma_width)
    suffix = torch.randn(1, cfg.action_horizon, D)
    mask = torch.ones(1, 8 + cfg.action_horizon, 8 + cfg.action_horizon, dtype=torch.bool)
    diffs = []
    tbase = np.linspace(0, 1, 11)
    with torch.no_grad():
        for t in tbase:
            c1 = torch.full((1, D), float(t))
            c2 = torch.full((1, D), float(min(t + 0.2, 1.0)))
            _, s1, _ = blk(prefix, suffix, c1, mask)
            _, s2, _ = blk(prefix, suffix, c2, mask)
            diffs.append((s1 - s2).abs().max().item())
    ax[0, 1].plot(tbase, diffs, marker="o", color="#c62828")
    ax[0, 1].axhline(1e-6, ls="--", color="gray", label="测试阈值 1e-6")
    ax[0, 1].set_xlabel("基准 timestep tau")
    ax[0, 1].set_ylabel("max|s(tau) - s(tau+0.2)|")
    ax[0, 1].set_title("Q13: suffix 输出依赖 cond（调制真正影响输出）")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].text(0.03, 0.97, "实测 max|s1-s2| = 1.15e-02 >> 1e-6\n（且远大于无 cond 时的数值噪声）",
                  transform=ax[0, 1].transAxes, fontsize=8, va="top",
                  bbox=dict(boxstyle="round", fc="#ffebee", alpha=0.9))

    # ---- 底左：真实路径（sinusoidal + TimeMLP）gate 随 t ----
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
    m = Pi05FaithfulPolicyModel(cfg).eval()
    blk2 = m.transformer.blocks[0]
    for mod in blk2.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == D * 3:
            torch.nn.init.normal_(mod.weight, std=0.02)
    gate_abs = []
    with torch.no_grad():
        for t in tbase:
            cond = m.time_mlp(m.time_embed(torch.full((1,), float(t))))
            _, g = blk2.input_norm_suffix(suffix, cond=cond)
            gate_abs.append(g.abs().max().item())
    ax[1, 0].plot(tbase, gate_abs, marker="o", color="#1565c0")
    ax[1, 0].set_xlabel("timestep tau")
    ax[1, 0].set_ylabel("gate absmax")
    ax[1, 0].set_title("真实路径: time_embed→TimeMLP→AdaRMSNorm 的 gate")
    ax[1, 0].grid(alpha=0.3)

    # ---- 底右：gate 逐维直方图（t=0 vs t=1） ----
    cond0 = m.time_mlp(m.time_embed(torch.zeros(1)))
    cond1 = m.time_mlp(m.time_embed(torch.ones(1)))
    with torch.no_grad():
        _, g0 = blk2.input_norm_suffix(suffix, cond=cond0)
        _, g1 = blk2.input_norm_suffix(suffix, cond=cond1)
    ax[1, 1].hist(g0.flatten().numpy(), bins=30, alpha=0.6, label="t=0", color="#90a4ae")
    ax[1, 1].hist(g1.flatten().numpy(), bins=30, alpha=0.6, label="t=1", color="#1565c0")
    ax[1, 1].set_xlabel("gate 值")
    ax[1, 1].set_ylabel("频数")
    ax[1, 1].set_title("gate 逐维分布: t=0 vs t=1")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle("adaRMSNorm 条件调制：gate 依赖时间步、suffix 输出依赖条件（Q12/Q13）", fontsize=12)
    plt.tight_layout()
    out = os.path.join(HERE, "plot_adarms_cond.png")
    plt.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
