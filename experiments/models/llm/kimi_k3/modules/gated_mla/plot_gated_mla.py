# -*- coding: utf-8 -*-
"""GatedMLA 可视化脚本（Analysis.md Phase 4.8 / 5）。

生成三张图：
  1. plot_gate_suppression.png：门强度 vs 输出幅度 —— 验证 4.3c（门可关断/强抑制）。
  2. plot_rank_scan.png：kv_lora_rank vs 检索丰富度 —— 验证 4.3a（latent 压缩
     有信息损失）。
  3. plot_cache_size.png：MLA latent cache vs 标准 KV cache 随序列长的增长 ——
     展示 MLA 的核心卖点（cache 不随 T 线性增长到 O(H·D)）。

每个 plot_xxx() 返回 (fig, text) 元组，供 Gradio demo 直接复用。
__main__ 块保存到 assets/ 并打印结论（向后兼容）。

数值回填见 GatedMLA-梳理-V1.md §4.8/5.3。
"""
from __future__ import annotations

import pathlib
import os
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── sys.path 设置（gated_mla 比 modules/ 多一层，需要 6 级 dirname）──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _THIS_DIR
for _ in range(6):
    _p = os.path.dirname(_p)
ROOT = pathlib.Path(_p)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_THIS_DIR))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.modules.mla import GatedMLA  # noqa: E402
from models.llm.kimi_k3.config import MLAConfig  # noqa: E402
from discussion import build  # noqa: E402

OUT = pathlib.Path(_THIS_DIR)


def plot_gate_suppression() -> Tuple[plt.Figure, str]:
    """输出门强度 vs MLA 输出幅度。

    返回 (fig, text)：fig 为折线图，text 为各偏置下的输出幅度数值。
    """
    layer = build().eval()
    x = torch.ones(1, 6, 32)
    offsets = [-1e4, -50.0, -10.0, 0.0, 10.0, 50.0]
    amps = []
    with torch.no_grad():
        for off in offsets:
            layer.g_proj.weight.fill_(off)
            amp = float(layer(x).abs().max())
            amps.append(amp)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(offsets, amps, 'o-', color='#1565c0', linewidth=2, markersize=7)
    ax.set_xlabel('g_proj 权重偏置 (sigmoid 输入)')
    ax.set_ylabel('MLA 输出 max|·|')
    ax.set_title('输出门 sigmoid(W_g x)：门越负输出越被抑制')
    ax.set_xscale('symlog', linthresh=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    lines = [f"gate_bias={o:>7}: 输出max|·|={a:.2e}" for o, a in zip(offsets, amps)]
    text = "输出门抑制能力验证:\n" + "\n".join(lines)
    return fig, text


def plot_rank_scan() -> Tuple[plt.Figure, str]:
    """kv_lora_rank vs 重构输出离散度（检索丰富度代理）。

    返回 (fig, text)：fig 为折线图，text 为各 rank 下的 std 数值。
    """
    torch.manual_seed(1)
    dim, T = 64, 24
    ranks = [8, 12, 16, 24, 48]
    stds = []
    for r in ranks:
        m = build(dim=dim, num_heads=4, qk_head_dim=dim // 4,
                  v_head_dim=dim // 4, q_lora_rank=dim, kv_lora_rank=r).eval()
        with torch.no_grad():
            x2 = torch.randn(1, T, dim)
            latent = m.kv_norm(m.kv_down(x2))
            k = m.k_up(latent).view(1, T, 4, dim // 4)
            v = m.v_up(latent).view(1, T, 4, dim // 4)
            qh = m._queries(x2).view(1, T, 4, dim // 4)
            attn = torch.softmax(
                (qh.transpose(1, 2) * m.scale) @ k.transpose(1, 2).transpose(-2, -1), dim=-1,
            )
            out = (attn @ v.transpose(1, 2)).transpose(1, 2)
            stds.append(float(out.reshape(1, -1).std().item()))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, stds, 's-', color='#6a1b9a', linewidth=2, markersize=7)
    ax.set_xlabel('kv_lora_rank（压缩维）')
    ax.set_ylabel('重构输出 std（检索丰富度）')
    ax.set_title('latent 压缩：rank 越低信息越被压平')
    ax.grid(alpha=0.3)
    plt.tight_layout()

    lines = [f"kv_lora_rank={r:2d}: std={s:.4f}" for r, s in zip(ranks, stds)]
    text = "latent 压缩信息损失:\n" + "\n".join(lines)
    return fig, text


def plot_cache_size() -> Tuple[plt.Figure, str]:
    """MLA latent cache vs 标准 KV cache 随 T 的增长。

    返回 (fig, text)：fig 为双对数折线图，text 为压缩比。
    """
    H, D, rank = 96, 128, 512  # 典型 K3 配置
    seqs = [512, 1024, 4096, 16384, 65536, 262144, 1048576]
    std_kv = [s * 2 * H * D for s in seqs]       # 每个 token 存 K 和 V
    mla_kv = [s * rank for s in seqs]            # 每个 token 只存 latent
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(seqs, [v / 1e6 for v in std_kv], 'o-', color='#c62828',
            label='标准 KV cache (O(T·H·D))')
    ax.plot(seqs, [v / 1e6 for v in mla_kv], 's-', color='#238636',
            label='MLA latent cache (O(T·rank))')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('序列长度 T')
    ax.set_ylabel('cache 元素数（百万）')
    ax.set_title('MLA latent 压缩：cache 从 O(H·D) 降到 O(kv_lora)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()

    ratio = 2 * H * D / rank
    text = (
        f"典型 K3 配置: H={H}, D={D}, rank={rank}\n"
        f"T=1M: 标准KV={std_kv[-1]/1e6:.0f}M 元素, MLA={mla_kv[-1]/1e6:.0f}M 元素\n"
        f"压缩比 ≈ {ratio:.0f}×\n"
        f"MLA 的核心卖点：cache 只随 kv_lora_rank 增长，不随 H·D 增长"
    )
    return fig, text


if __name__ == "__main__":
    fig1, txt1 = plot_gate_suppression()
    fig1.savefig(OUT / "plot_gate_suppression.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_gate_suppression.png")
    print(txt1)

    fig2, txt2 = plot_rank_scan()
    fig2.savefig(OUT / "plot_rank_scan.png", dpi=120)
    plt.close(fig2)
    print("\n已生成", OUT / "plot_rank_scan.png")
    print(txt2)

    fig3, txt3 = plot_cache_size()
    fig3.savefig(OUT / "plot_cache_size.png", dpi=120)
    plt.close(fig3)
    print("\n已生成", OUT / "plot_cache_size.png")
    print(txt3)
