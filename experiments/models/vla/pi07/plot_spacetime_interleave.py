# -*- coding: utf-8 -*-
"""Q2 可视化：为什么"每隔 4 层"把空间注意力换成空间+时间交替？

三个子图回答三件事：
1. 左图：普通 SigLIP（全部 27 层纯空间）vs OpenTau SpaceTime（每 4 层一个
   时空层，索引 [3,7,11,15,19,23]）的层布局对比 —— 直观看出"稀疏插入"；
2. 中图：不同 stride 下"时空层占比"与"新增参数量"的关系 —— 说明为什么
   stride=4 是甜点（时间信息够用，且参数量恒为 0）；
3. 右图：真跑本项目 SpaceTimeSiglipEncoder，测"当前帧 token 对每一历史帧像素的
   敏感度" —— 证明时间轴信息真的被混进了当前帧表示。

第 3 个子图是真实数值：用有限差分（扰动第 t 帧像素，看当前帧输出变化量）。
实测结论（注意与直觉不同）：**各历史帧响应量相等**（本实现只有 1 层因果时间
注意力、且无时间位置编码时，softmax 对历史 key 给出近似均匀权重），当前帧
响应远大于历史帧（既有时间注意力的残差直连，也有它自身的空间通路）。
真实 OpenTau 加了正弦时间位置编码（当前帧那一行恒为 0），才会让不同时间距离
的帧被区分对待 —— 这正是本项目与官方实现的一处简化差异。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import C_STD, C_NEW, C_AUX, C_WARN, OUT, setup_fonts


def spatial_only_sensitivity(n_layers: int = 27, stride: int = 4) -> list[int]:
    """返回被替换为时空层的层索引（对齐 OpenTau：从 stride-1 开始每 stride 一个）。"""
    return list(range(stride - 1, n_layers, stride))


def measure_history_sensitivity(seed: int = 0):
    """真跑本项目编码器：测当前帧输出对每一历史帧的敏感度。

    做法（有限差分）：
    - 基线 video 全零，前向得到 base 输出（只保留当前帧 token）；
    - 逐个历史帧 t 加同样幅度的扰动，测输出的 L2 变化量。
    变化量 > 0 就说明"当前帧 token 里含有第 t 帧的信息"。
    """
    from models.vla.pi07.modules.space_time import SpaceTimeSiglipEncoder

    torch.manual_seed(seed)
    T = 6
    enc = SpaceTimeSiglipEncoder(image_size=28, patch_size=7, dim=32, n_layers=1,
                                 n_heads=4, temporal_layers=1, temporal_heads=2,
                                 max_num_frames=T).eval()
    video = torch.zeros(1, T, 3, 28, 28)
    with torch.no_grad():
        base = enc(video)                       # (1, S, D) 只含当前帧
    eps = 0.5
    sens = []
    for t in range(T):
        v = video.clone()
        v[:, t] += eps
        with torch.no_grad():
            out = enc(v)
        sens.append(float((out - base).norm()))
    return sens, T


def plot() -> tuple:
    """绘制时空交替注意力三子图，返回 (fig, text)。

    Returns:
        (fig, text): matplotlib Figure 对象和关键发现文本摘要。
    """
    setup_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # ---------------- 左：层布局对比 ----------------
    ax = axes[0]
    n_layers, stride = 27, 4
    st_idx = spatial_only_sensitivity(n_layers, stride)
    for i in range(n_layers):
        # 普通 SigLIP：全部纯空间
        ax.add_patch(plt.Rectangle((i, 1.15), 0.86, 0.62, facecolor=C_STD,
                                   edgecolor="white", linewidth=0.6))
        # SpaceTime：每 4 层换成时空层
        color = C_NEW if i in st_idx else C_AUX
        ax.add_patch(plt.Rectangle((i, 0.25), 0.86, 0.62, facecolor=color,
                                   edgecolor="white", linewidth=0.6))
        if i in st_idx:
            ax.text(i + 0.43, 0.56, "T", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(-0.6, n_layers + 0.4)
    ax.set_ylim(0.0, 2.15)
    ax.set_yticks([0.56, 1.46])
    ax.set_yticklabels(["π0.7 / MEM\nSpaceTime", "π0.5 / π0.6\n纯空间 SigLIP"])
    ax.set_xlabel("SigLIP 层索引（共 27 层）")
    ax.set_title("每隔 4 层插一个时空层\n被替换索引 = [3, 7, 11, 15, 19, 23]",
                 fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C_STD),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_AUX),
               plt.Rectangle((0, 0), 1, 1, facecolor=C_NEW)]
    ax.legend(handles, ["纯空间层（无时间轴）", "纯空间层（保留）", "时空层（空间+时间）"],
              loc="upper center", fontsize=8, ncol=1, framealpha=0.9)
    ax.set_xticks([0, 3, 7, 11, 15, 19, 23, 26])

    # ---------------- 中：stride 权衡 ----------------
    ax = axes[1]
    strides = [1, 2, 3, 4, 6, 9, 27]
    ratio = [len(spatial_only_sensitivity(27, s)) / 27 * 100 for s in strides]
    ax.plot(range(len(strides)), ratio, "o-", color=C_NEW, linewidth=2,
            markersize=8, label="时空层占比 (%)")
    ax.axvline(3, color=C_WARN, linestyle="--", linewidth=1.6)
    ax.text(3.1, 70, "stride=4\nMEM 论文取值", color=C_WARN, fontsize=9)
    for i, (s, r) in enumerate(zip(strides, ratio)):
        ax.annotate(f"{r:.0f}%", (i, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(range(len(strides)))
    ax.set_xticklabels([f"{s}" for s in strides])
    ax.set_xlabel("spacetime_layer_stride")
    ax.set_ylabel("27 层中时空层的占比 (%)")
    ax.set_title("为什么是 4：时间混合够用，\n且新增可训练参数恒为 0", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center right")
    ax.text(0.5, 12, "关键：时空层复用同一份 W_Q/W_K/W_V/W_O，\n"
                     "只是把 V 换成时间混合后的 V′\n→ 参数量 0 增长，可直接热启已有 SigLIP 权重",
            fontsize=8.5, bbox=dict(boxstyle="round", facecolor="#fff8e1",
                                    edgecolor=C_WARN, alpha=0.95))

    # ---------------- 右：真实历史帧敏感度（实测为均等，非衰减）----------------
    ax = axes[2]
    sens, T = measure_history_sensitivity()
    xs = np.arange(T)
    labels = [f"t-{T - 1 - i}" if i < T - 1 else "当前帧 t" for i in range(T)]
    # 实测：因果注意力 + 无时间位置编码 → 各历史帧响应均匀
    bars = ax.bar(xs, sens, color=[C_AUX] * (T - 1) + [C_NEW], alpha=0.9,
                  label="本项目实测（无时间位置编码 → 均匀）")
    for x, v in zip(xs, sens):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    # 理论对照：如果加上正弦时间位置编码（OpenTau 官方实现），应当随距离衰减
    # 这里用 e^{-k·d} 做示意（仅参考，不是真实数据）
    decay_ref = sens[-1] * np.exp(-0.7 * np.arange(T)[::-1])
    ax.plot(xs, decay_ref, "s--", color=C_WARN, lw=1.8, ms=7,
            label="理论参考（含时间位置编码时衰减）")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("当前帧输出的 L2 变化量")
    ax.set_xlabel("被扰动的帧")
    ax.set_title("真跑本项目编码器：扰动任一历史帧，\n当前帧 token 都会变（信息真被混进来）",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, loc="upper left")
    nonzero = sum(1 for v in sens if v > 1e-6)
    ax.text(0.98, 0.55, f"非零响应帧数 = {nonzero}/{T}\n"
                        f"→ 因果时间注意力让当前帧\n   聚合了全部历史\n"
                        f"→ 但各帧权重均匀（本项目简化）\n"
                        f"   官方加时间位置编码后才会衰减",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="#fff8e1",
                      edgecolor=C_WARN, alpha=0.95))

    fig.suptitle("Q2：为什么把空间注意力换成「空间+时间交替」，且每隔 4 层才换一次",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    text = (f"时空层索引: {st_idx}\n"
            f"历史帧敏感度: {[round(v, 4) for v in sens]}\n"
            f"非零响应帧数 = {nonzero}/{T} → 因果时间注意力聚合全部历史")
    return fig, text


def main() -> None:
    fig, text = plot()
    fig.savefig(OUT / "plot_spacetime_interleave.png", dpi=130)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
