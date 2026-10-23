"""KDA Context Parallelism (KCP) 验证实验 —— 论文 §5.1.2 / Eq. 17。

核心问题：为什么 KDA 的状态跨 segment **不可加**？
- Vanilla 线性注意力 S = sum(k v^T)，矩阵加法可交换，S_total = S_seg1 + S_seg2。
- KDA 的更新 S = (I - beta k k^T) Diag(alpha) S + beta k v^T，中间的 (I-beta k k^T)
  是**依赖 token 的矩阵**作用于输入状态，变换不可交换 -> 各段贡献不能相加。

KCP 的解法：把每段的效果分解为两个局部可算的量
  - M_{[i]}  段内累积转移矩阵（homogeneous 响应，v=0 且 S0=I 跑一次）
  - Stilde_{[i]}  从 S=0 起算的段内自生成状态
然后按 S_{[i+1]} = Stilde_{[i+1]} + M_{[i+1]} S_{[i]} 前缀扫描重组。

本脚本在 CPU 上用 4 段测试三件事：
1. naive 相加（错误做法）结果 vs 精确递推的偏差 —— 验证"不可加"
2. KCP 前缀扫描（正确做法）vs 精确递推 —— 验证"可按 M/Stilde 重组"
3. 通信量对序列长度不敏感（与 softmax attention 的 KV 交换随长度增长对比）
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

# 必须在 pyplot 之前配置中文字体，否则标题/图例里的中文会渲染成方框（豆腐块）。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent

HERE = os.path.dirname(os.path.abspath(__file__))


def make_inputs(b=1, t=64, h=3, dk=16, dv=16, *, g_min=-5.0, dtype=torch.float64, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=dtype, generator=gen)
    g = g_min * torch.sigmoid(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=dtype, generator=gen))
    return q, k, v, g, beta


def zero_start_state(q, k, v, g, beta, seg):
    """段 [seg] 在 S=0 下跑出的状态 = Stilde。"""
    _, s = kda_recurrent(q[:, :, seg], k[:, :, seg], v[:, :, seg], g[:, :, seg], beta[:, :, seg])
    return s


def cumulative_transition(q, k, v, g, beta, seg):
    """M = homogeneous 响应：v=0、S0=I 下跑完该段的末状态。"""
    dk = k.shape[-1]
    eye = torch.eye(dk, dtype=k.dtype, device=k.device).expand(q.shape[0], q.shape[1], dk, dk).contiguous()
    zero = torch.zeros_like(k[:, :, seg])
    _, m = kda_recurrent(q[:, :, seg], k[:, :, seg], zero, g[:, :, seg], beta[:, :, seg], initial_state=eye)
    return m


def main():
    torch.set_num_threads(2)
    # 启用中文字体：不调用则所有中文标签变成方框（本脚本此前的乱码原因）。
    font = setup_cjk_font()
    print("cjk font:", font)
    q, k, v, g, beta = make_inputs(t=64)
    seg4 = [slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64)]

    # --- ground truth：一次性跑完整序列 ---
    _, exact = kda_recurrent(q, k, v, g, beta)  # (B,H,dk,dv)

    # --- 错误做法：各段 S=0 结果直接相加 ---
    add = sum(zero_start_state(q, k, v, g, beta, s) for s in seg4)
    add_err = (add - exact).abs().max().item()

    # --- 正确做法：KCP 前缀扫描 S <- M S + Stilde ---
    running = torch.zeros_like(exact)
    for s in seg4:
        running = cumulative_transition(q, k, v, g, beta, s) @ running + zero_start_state(q, k, v, g, beta, s)
    kcp_err = (running - exact).abs().max().item()

    print("exact  state rms      :", exact.square().mean().sqrt().item())
    print("[case1] 直接相加差    :", add_err)
    print("[case2] KCP 前缀扫描差:", kcp_err)

    # --- 通信量对比：KDA 固定 vs softmax KV 随长度增长 ---
    heads, dk, dv = 3, 16, 16
    lengths = [1024, 4096, 16384, 65536, 262144, 1048576]
    kda_bytes = [4 * (dk * dk + dk * dv) for _ in lengths]  # M + Stilde, fp32, 单层单 rank
    sm_bytes = [4 * lengths[i] * dk * dv for i in range(len(lengths))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    # log 轴的默认刻度用 mathtext 排版负号（U+2212），而 mathtext 走的是 'default'
    # 字体（不含该字形）→ 触发一堆 "does not have a glyph for '-'" 告警。
    # 这里统一改成 ASCII 连字符的 10^n 文本，彻底消除告警。
    log_fmt = FuncFormatter(lambda y, _p: "1e%d" % round(math.log10(y)) if y > 0 else "0")

    ax[0].bar(["直接相加", "KCP 前缀扫描"], [add_err, kcp_err])
    ax[0].set_ylabel("对精确递推的 max abs 误差")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_formatter(log_fmt)
    ax[0].set_title("跨段状态重组方式 vs 精确递推")
    ax[0].text(0, add_err * 1.3, f"{add_err:.2e}", ha="center")
    ax[0].text(1, kcp_err * 1.3, f"{kcp_err:.2e}", ha="center")

    ax[1].loglog(lengths, sm_bytes, marker="o", label="softmax attention KV 交换量")
    ax[1].loglog(lengths, kda_bytes, marker="s", label="KDA 固定 (M, Stilde) 交换量")
    ax[1].set_xlabel("序列长度")
    ax[1].set_ylabel("通信量 (字节 / rank / 层 / fp32)")
    ax[1].xaxis.set_major_formatter(log_fmt)
    ax[1].yaxis.set_major_formatter(log_fmt)
    ax[1].set_title("上下文并行通信量：KDA 与序列长度无关")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "plot_kcp.png"), dpi=130)
    print("saved plot_kcp.png")

    # 回填记录（Phase 5.3 的实测数值）
    print(f"直接相加误差={add_err:.3e}, KCP 误差={kcp_err:.3e}")
    print(kda_bytes, sm_bytes, sep="\n")


if __name__ == "__main__":
    main()