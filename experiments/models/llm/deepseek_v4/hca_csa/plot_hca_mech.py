# -*- coding: utf-8 -*-
"""HCA 机制可视化：软池化权重 / 压缩率扫描 / 块完整性实验。

每个函数返回 ``(fig, text)``，供 Gradio demo 直接 import 复用。
"""
from __future__ import annotations

import pathlib
import sys

_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS
for _ in range(6):
    _ROOT = _ROOT.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "experiments"), str(_THIS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"

from models.llm.deepseek_v4.config import HCAConfig, MQAV4Config  # noqa: E402
from models.llm.deepseek_v4.modules.hca import HCATokenCompressor, HCAV4Attention  # noqa: E402

DIM, H, DH, QLR = 64, 4, 16, 32


def _small_mqa():
    return MQAV4Config(dim=DIM, n_heads=H, head_dim=DH, q_lora_rank=QLR,
                       qk_rope_head_dim=8, window_size=8, max_seq_len=512)


def plot_pooling_weights_figure(m: int = 8, head_dim: int = 16):
    """(fig, text)：HCA 软池化权重 heatmap — 门控 + position_bias -> softmax。

    demo 中 position_bias 用随机初始化（非零），让左图呈现「每个维度
    独立的位置偏好」结构，与生产训练后的期望形态一致。
    """
    torch.manual_seed(1)
    comp = HCATokenCompressor(m, head_dim)
    # 覆盖零初始化：给 position_bias 一个小的随机值，模拟「训练后」的
    # 非均匀位置先验（真实训练通过梯度下降习得，demo 用随机数代替以可视化）。
    # 幅度 0.15 让 position_bias 约为 gate（randn, std=1）的 1/7，
    # 右图主要由 gate 驱动呈现深绿斑块，同时左图保留可见的红蓝条纹。
    with torch.no_grad():
        comp.position_bias.copy_(0.15 * torch.randn(m, head_dim))
    kv = torch.randn(1, m, head_dim)
    gate = torch.randn(1, m, head_dim)
    scores = gate + comp.position_bias.unsqueeze(0)
    weights = F.softmax(scores.float(), dim=1).detach().numpy()
    bias = comp.position_bias.detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    vmax = float(abs(bias).max())
    im0 = axes[0].imshow(bias.T, aspect="auto", cmap="coolwarm",
                        vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel("块内位置 (0..m-1)")
    axes[0].set_ylabel("head_dim 维度")
    axes[0].set_title("position_bias：可学习的块内位置偏置 (论文 Eq.22 的 B)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(weights[0].T, aspect="auto", cmap="Greens",
                        vmin=0, vmax=1)
    axes[1].set_xlabel("块内位置 (0..m-1)")
    axes[1].set_ylabel("head_dim 维度")
    axes[1].set_title("softmax 池化权重：按维度独立归一化")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"HCA 软池化：w = softmax(gate + position_bias)  m={m}")
    fig.tight_layout()

    w = weights[0]
    text = (f"position_bias 形状 [{m}, {head_dim}]，每个维度独立 softmax，"
            f"沿位置维总和 = 1。\n"
            f"demo 用随机初始化代替训练后的 B（真实训练中通过梯度下降习得），"
            f"左图红蓝条纹展示「不同维度偏好不同位置」的结构。\n"
            f"偏置让模型学会「块内哪些位置更值得保留进摘要」"
            f"—— 对应论文 Eq.22 的 learnable positional biases B。\n"
            f"最大权重 {float(w.max()):.3f}（dim {int(w.argmax()//m)}, "
            f"pos {int(w.argmax()%m)}），均匀分布理论值 {1/m:.3f}。")
    return fig, text


def plot_completeness_figure(m: int = 4, head_dim: int = 16):
    """(fig, text)：块完整性实验 — 不完整的尾块不产压缩条目。"""
    torch.manual_seed(2)
    comp = HCATokenCompressor(m, head_dim)
    ts = list(range(0, 20))
    counts = []
    for T in ts:
        kv = torch.randn(1, T, head_dim)
        gate = torch.randn(1, T, head_dim)
        pos = torch.arange(T).unsqueeze(0)
        c, _ = comp(kv, gate, pos)
        counts.append(c.shape[1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.step(ts, counts, "o-", color="#238636", lw=2, where="post",
            markersize=6, label=f"压缩块数 S = Floor(T/m)")
    for i in range(0, 20, m):
        ax.axvline(i, color="#8b949e", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("输入序列长度 T")
    ax.set_ylabel("压缩块数 S")
    ax.set_title(f"块完整性：只有满 m={m} 个 token 才产压缩条目（尾块丢弃）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    text = (f"T = 0..{m-1}：0 块；T = {m}..{2*m-1}：1 块；依此类推。\n"
            f"尾块（不足 m 个 token）不产出压缩条目 —— 这是因果性的要求："
            f"如果产了半块，任何看得到它的 query 都会超前看到「未来 token」。\n"
            f"被扔掉的尾块 token 由滑窗分支兜住（论文 §2.3.3 的 SWA 动机）。")
    return fig, text


def plot_rate_sweep_figure(window_size: int = 128, head_dim: int = 128):
    """(fig, text)：压缩率扫描 — m 越大，KV cache 占比按 1/m 下降。"""
    seqlens = [1024, 4096, 16384, 65536, 262144, 1048576]
    rates = [2, 4, 8, 16, 32, 64, 128]

    fig, ax = plt.subplots(figsize=(10, 5))
    for r in rates:
        vals = [window_size + T // r for T in seqlens]
        ax.plot(seqlens, vals, "o-", markersize=5, lw=1.8,
                label=f"m = {r}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("序列长度 T")
    ax.set_ylabel("每 KV 头条目数（window + T/m）")
    ax.set_title("压缩率扫描：KV cache 斜率按 1/m 变缓")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    r128_1m = window_size + 1048576 // 128
    r4_1m = window_size + 1048576 // 4
    text = (f"T=1M 时：m=4 → {r4_1m:,} 条目，m=128 → {r128_1m:,} 条目，"
            f"差 {r4_1m / r128_1m:.0f} 倍。\n"
            f"window={window_size} 在短序列时主导，长序列时 T/m 占主导。\n"
            f"HCA 的 m=128 把「压缩腿」的斜率压到普通注意力的 1/128，"
            f"是百万上下文可行的数量级关键（论文 §2.3.2 / §2.3.4）。")
    return fig, text


def main():
    import pathlib
    out = pathlib.Path(__file__).parent / "assets"
    out.mkdir(exist_ok=True)

    for name, fn in [
        ("plot_hca_pooling", lambda: plot_pooling_weights_figure()),
        ("plot_hca_completeness", lambda: plot_completeness_figure()),
        ("plot_hca_rate_sweep", lambda: plot_rate_sweep_figure()),
    ]:
        fig, txt = fn()
        fig.savefig(out / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"已生成 {out / (name + '.png')}")
        print(txt)
        print("-" * 40)


if __name__ == "__main__":
    main()
