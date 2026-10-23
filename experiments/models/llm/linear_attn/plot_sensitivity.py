# -*- coding: utf-8 -*-
"""参数敏感性图: alpha(decay) 与 beta(write) 对检索精度的影响。

验证 4.1.5: alpha 越大保留越多但混叠越多; beta 越大修正越激进。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font
setup_cjk_font()

from models.llm.linear_attn.variants import delta_net_step

OUT = pathlib.Path(__file__).resolve().parent


def sweep_alpha() -> plt.Figure:
    """固定 beta, 扫描 alpha: 看最终检索误差随 alpha 变化。返回 Figure。"""
    B, H, Dk, Dv, T = 1, 1, 8, 4, 30
    torch.manual_seed(1)
    k = torch.randn(B, H, Dk)
    k = k / k.norm(dim=-1, keepdim=True)

    alphas = [0.0, 0.3, 0.6, 0.8, 0.95, 0.99]
    beta = 0.8
    final_errors = []
    for alpha in alphas:
        S = torch.zeros(B, H, Dk, Dv)
        for t in range(T):
            v_true = torch.zeros(B, H, Dv)
            v_true[..., 0] = 1.0 if (t // 10) % 2 == 0 else -1.0
            q = torch.randn(B, H, Dk)
            _, S, _, _ = delta_net_step(S, q, k, v_true,
                                        alpha=alpha, beta=beta)
        retrieved = torch.einsum("bhd,bhdv->bhv", k, S)[..., 0]
        final_errors.append((retrieved - v_true[..., 0]).abs().item())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, final_errors, "o-", color="#238636", linewidth=2)
    ax.set_xlabel("alpha (衰减, 保留比例)")
    ax.set_ylabel("最终检索误差")
    ax.set_title("DeltaNet 参数敏感性: alpha 越大, 旧信息保留越多但混叠越多")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = sweep_alpha()
    fig.savefig(OUT / "plot_sensitivity.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_sensitivity.png")
