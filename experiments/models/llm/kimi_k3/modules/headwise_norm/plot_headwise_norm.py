# -*- coding: utf-8 -*-
"""HeadRMSNorm 可视化 demo（Analysis.md Phase 4.8 / 5）。

生成两张图：
  1. plot_head_vs_whole.png：head-wise 归一化 vs 整体归一化在"近零 head"上的对比，
     验证 4.3c（近零 head 被放大）。
  2. plot_scale_drift.png：模拟跨 head 尺度不平衡场景，展示 head-wise 归一化把
     各 head 拉到可比尺度（支撑 K3 用它做输出门第一步）。

数值回填见 HeadwiseRMSNorm-梳理-V1.md §4.8/5.3。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.modules.norm import HeadRMSNorm, rms_norm  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def plot_head_vs_whole() -> None:
    torch.manual_seed(0)
    head_dim = 8
    x = torch.zeros(1, 1, 3, head_dim)
    x[0, 0, 0] = torch.randn(head_dim) * 2.0
    x[0, 0, 1] = torch.randn(head_dim) * 1e-4
    x[0, 0, 2] = 0.0
    layer = HeadRMSNorm(head_dim, eps=1e-6)
    with torch.no_grad():
        out_head = layer(x)
        flat = x.reshape(1, 1, -1)
        out_whole = rms_norm(flat, weight=None, eps=1e-6).reshape_as(x)
    rms = lambda t: t.pow(2).mean(-1).sqrt()[0, 0]  # noqa: E731

    heads = ['正常 head', '近零 head (噪声)', '全零 head']
    head_rms = [float(rms(out_head)[h]) for h in range(3)]
    whole_rms = [float(rms(out_whole)[h]) for h in range(3)]

    import numpy as np
    xpos = np.arange(3)
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(xpos - width / 2, head_rms, width, label='HeadRMSNorm (head-wise)', color='#238636')
    plt.bar(xpos + width / 2, whole_rms, width, label='整体 RMSNorm', color='#c62828')
    plt.xticks(xpos, heads)
    plt.ylabel('归一化后输出 RMS')
    plt.title('近零/全零 head：head-wise 归一化放大噪声，整体归一化保持其小')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUT / "plot_head_vs_whole.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_head_vs_whole.png")
    print(f"  head-wise RMS: {[f'{v:.3f}' for v in head_rms]}")
    print(f"  整体 RMS:      {[f'{v:.2e}' for v in whole_rms]}")


def plot_scale_drift() -> None:
    """模拟跨 head 尺度漂移：各 head 输入幅度从 0.5 到 50。"""
    torch.manual_seed(0)
    head_dim, n_heads = 16, 6
    scales = [0.5, 1.0, 3.0, 10.0, 30.0, 50.0]
    x = torch.zeros(1, 1, n_heads, head_dim)
    for h, s in enumerate(scales):
        x[0, 0, h] = torch.randn(head_dim) * s
    layer = HeadRMSNorm(head_dim, eps=1e-6)
    with torch.no_grad():
        out = layer(x)
    rms = lambda t: t.pow(2).mean(-1).sqrt()[0, 0]  # noqa: E731

    in_rms = [float(rms(x)[h]) for h in range(n_heads)]
    out_rms = [float(rms(out)[h]) for h in range(n_heads)]
    plt.figure(figsize=(8, 5))
    plt.plot(in_rms, 'o-', label='输入 RMS（漂移明显）', color='#c62828')
    plt.plot(out_rms, 's-', label='HeadRMSNorm 后 RMS（被拉平）', color='#238636')
    plt.xlabel('head 索引')
    plt.ylabel('RMS')
    plt.title('HeadRMSNorm 把跨 head 尺度漂移拉平到可比尺度')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_scale_drift.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_scale_drift.png")
    print(f"  输入 RMS: {[f'{v:.2f}' for v in in_rms]}")
    print(f"  输出 RMS: {[f'{v:.3f}' for v in out_rms]}")


if __name__ == "__main__":
    plot_head_vs_whole()
    plot_scale_drift()
