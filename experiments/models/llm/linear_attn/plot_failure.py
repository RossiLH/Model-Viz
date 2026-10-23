# -*- coding: utf-8 -*-
"""失败 case 图: KDA 固定状态的信息容量瓶颈。

验证反方 a (4.3.3): 同一 key 连续写不同 value, 旧 value 被覆写, 状态容量有限。
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

from models.llm.linear_attn.variants import kimi_linear_step

OUT = pathlib.Path(__file__).resolve().parent


def capacity_bottleneck() -> plt.Figure:
    """同一 key 写入不同 value, 观察状态覆写行为。返回 Figure。"""
    B, H, Dk, Dv = 1, 1, 8, 1
    S = torch.zeros(B, H, Dk, Dv)
    k = torch.zeros(B, H, Dk)
    k[..., 0] = 1.0
    values = [0.9, 0.5, 0.1, -0.3, 0.7, -0.8, 0.4, 0.0]
    retrievals = []
    for v in values:
        vs = torch.tensor([[[v]]])
        out, S, _, _ = kimi_linear_step(
            S, torch.zeros(B, H, Dk), k, vs,
            torch.zeros(B, H, Dk), torch.zeros(B, H), torch.ones(H),
        )
        retrievals.append(float(S[0, 0, 0, 0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(values)), retrievals, "o-", color="#a13a1a",
            linewidth=2, label="检索值 S[0,0]")
    ax.plot(range(len(values)), values, "s--", color="#1a60a1",
            linewidth=2, label="真实写入值 v")
    ax.set_xlabel("写入步数 (同一 key)")
    ax.set_ylabel("值")
    ax.set_title("KDA 信息容量瓶颈: 同一 key 旧 value 被覆写")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = capacity_bottleneck()
    fig.savefig(OUT / "plot_failure.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_failure.png")
