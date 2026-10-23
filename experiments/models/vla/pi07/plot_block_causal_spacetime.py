# -*- coding: utf-8 -*-
"""PI0.7 可视化：block-causal 掩码 + SpaceTime 时空压缩（MEM 风格）

【展示特色】π0.7 相对 π0.5 的两个结构性升级：
   1. **block-causal attention**：视频 patch / 子目标图像各自成双向块、
      文本严格因果、状态与子目标聚成双向块 —— 一个 ar_mask 就描述全部。
   2. **MEM 风格时空压缩**：任意长度视频历史 → **固定数量 token**。
      无论喂 3 帧还是 12 帧，模型 prefix 里的视觉 token 数不变（只取
      当前帧的 patch 表示），token 数不随 T 增长 —— 这是"长历史可吞"的根。
【展示思路】左图：把一条真实 prefix 的 block-causal mask 画成热力图，
             并标注每个 token 的 block id（视频双向块 / 文本因果 / 状态块），
             让"块内双向、块间因果"一目了然。
             右图：真实 SpaceTimeSiglipEncoder 分别吃 1/3/6/12 帧，画
             "历史帧数 vs 输出 token 数" —— 曲线水平，证明 O(1) 时空压缩。
【如何展示】全部数据来自真实模块（embed_prefix / SpaceTimeSiglipEncoder），
             随机初始化 + 固定 seed 保证可复现。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_NEW, C_AUX  # noqa: E402
from models.vla.pi07.modules.attention import block_ids  # noqa: E402
from models.vla.pi07.modules.space_time import SpaceTimeSiglipEncoder  # noqa: E402

setup_fonts()


def _prefix_mask_example() -> tuple:
    """真实 embed_prefix 的 ar_mask / pad_mask 示例（带子目标图像）。

    构造一个小 batch（无响应、有子目标、有元数据），返回 ar_mask 和
    布局说明（每段的 token 数），用于画热力图。
    """
    torch.manual_seed(0)
    model = build_model()
    b = 1
    videos = torch.randn(b, 3, 3, 28, 28)          # 3 帧视频
    lang = torch.tensor([[5, 6, 7, 0, 0, 0]])      # 3 个有效文本 token
    lang_mask = torch.tensor([[True, True, True, False, False, False]])
    state = torch.randn(b, 3, 8)                   # 3 个状态 token
    meta = torch.tensor([[2000.0, 5.0, 0.0]])
    subgoal = torch.rand(b, 3, 28, 28)
    subgoal_mask = torch.ones(b, dtype=torch.bool)
    embs, pads, ars = model.embed_prefix(
        videos, lang, lang_mask, state,
        metadata=meta, subgoal_images=subgoal, subgoal_mask=subgoal_mask,
    )
    return ars[0].numpy(), pads[0].numpy(), embs.shape[1]


def _spacetime_token_count(num_frames: int, encoder: SpaceTimeSiglipEncoder) -> int:
    """跑真实 SpaceTimeSiglipEncoder，统计输出 token 数。"""
    vid = torch.randn(1, num_frames, 3, 28, 28)
    out = encoder(vid)
    return out.shape[1]


def plot_block_causal() -> tuple:
    """图1：block-causal 掩码热力图 + block id 分段。返回 (fig, text)。"""
    ar, pad, S = _prefix_mask_example()
    from models.vla.pi07.modules.attention import make_block_attn_mask
    ar_t = torch.tensor(ar, dtype=torch.bool)[None, :]
    pad_t = torch.tensor(pad, dtype=torch.bool)[None, :]
    ids = block_ids(ar_t)[0].numpy()
    mask = make_block_attn_mask(ar_t, pad_t)[0].numpy()
    reachable = mask > -1e6

    fig, ax = plt.subplots(figsize=(11, 7.5))
    im = ax.imshow(reachable, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("key 位置 j（被看的 token）")
    ax.set_ylabel("query 位置 i（正在看的 token）")
    ax.set_title("真实 embed_prefix 的 block-causal 掩码（S=%d）\n"
                 "深色 = 可 attend；块内双向（方块）、块间因果（阶梯）" % S,
                 fontsize=11)

    segments = _segment_legend(ar, pad)
    import matplotlib.patches as mpatches
    handles = []
    seg_colors = ["#1565c0", "#238636", "#ef8c00", "#6a1b9a", "#c62828"]
    for idx, (name, start, end, bid) in enumerate(segments):
        if start > 0:
            ax.axhline(start - 0.5, color="r", lw=1.1, alpha=0.55)
            ax.axvline(start - 0.5, color="r", lw=1.1, alpha=0.55)
        handles.append(mpatches.Patch(
            color=seg_colors[idx % len(seg_colors)],
            label="[%2d:%2d] %s" % (start, end, name)))
    ticks = [s for _, s, _, _ in segments] + [S]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=ax, label="可 attend", ticks=[0, 1], shrink=0.7)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
              fontsize=8, ncol=1, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    lines = ["block-causal 掩码 (S=%d)，5 个语义段:" % S]
    for name, start, end, bid in segments:
        lines.append("  tokens[%2d:%2d] block_id=%-3s %s" % (start, end, bid, name))
    text = "\n".join(lines)
    return fig, text


def _segment_legend(ar: np.ndarray, pad: np.ndarray) -> list:
    """把真实 ar_mask 切成连续段，并给每段一个**准确**的语义名。

    不能靠启发式猜名字（会把子目标图像块误标成"视频块"）。这里直接按
    embed_prefix 的构造顺序给出本配置（S=49）的段布局，并用 run 边界
    断言锁死 —— 模型若改版，断言先失败，图注不会静默变错。

    本配置的 5 个 run（实测，见 plot 输出）：
        run0 [0:17]  = 视频 patch 块（16 patch + cls，双向）
        run1 [17:25] = 语言(6) + "State: " 指示(1) + 状态首 token(1, ar=1)，因果
        run2 [25:27] = 状态剩余 token（双向块）
        run3 [27:33] = 分隔符 + 元数据(3) + 分隔符 + "Subgoal: " 指示，因果
        run4 [33:49] = 子目标图像 patch 块（双向）
    """
    runs = []
    i = 0
    while i < len(ar):
        j = i
        while j < len(ar) and ar[j] == ar[i]:
            j += 1
        runs.append((i, j, int(ar[i])))
        i = j
    # 断言 run 数与边界与 embed_prefix 构造一致（防图注脱节）
    assert len(runs) == 5, "ar_mask run 数=%d，与 embed_prefix 布局不符" % len(runs)
    assert runs[0][0] == 0 and runs[0][1] == 17
    assert runs[-1][1] == len(ar)

    names = [
        "视频 patch 块（含 cls，双向）",
        "语言 + \"State: \" 指示 + 状态首 token（因果）",
        "状态剩余 token（双向块）",
        "分隔符 + 元数据 + 分隔符 + \"Subgoal: \" 指示（因果）",
        "子目标图像 patch 块（双向）",
    ]
    out = []
    for (s0, s1, v), name in zip(runs, names):
        bid = block_ids(torch.tensor(ar[s0:s1], dtype=torch.bool))[0].item()
        out.append((name, s0, s1, bid))
    return out


def plot_spacetime_compression() -> tuple:
    """图2：历史帧数 → 输出 token 数（水平线 = O(1) 时空压缩）。返回 (fig, text)。"""
    torch.manual_seed(0)
    enc = SpaceTimeSiglipEncoder(
        image_size=28, patch_size=7, dim=32, n_layers=1, n_heads=4,
        temporal_layers=1, temporal_heads=2, max_num_frames=12,
    )
    frames = [1, 3, 6, 12]
    counts = [_spacetime_token_count(f, enc) for f in frames]
    n_patches = enc.num_patches
    counts_no_cls = [c - 1 for c in counts]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(frames, counts_no_cls, "o-", color=C_NEW, lw=2, ms=8)
    ax.set_xticks(frames)
    ax.set_xlabel("视频历史帧数 T")
    ax.set_ylabel("编码器输出 patch token 数（不含 cls）")
    ax.set_ylim(0, max(counts_no_cls) * 1.4)
    ax.axhline(counts_no_cls[-1], color=C_AUX, ls="--", lw=0.8)
    ax.annotate("patch token 数恒定 = %d（= 单帧 patch 数）\n不随 T 增长 → O(1) 状态" % counts_no_cls[-1],
                xy=(6, counts_no_cls[-1] * 1.15), fontsize=9, color=C_AUX)
    ax.set_title("SpaceTimeSiglip 时空压缩：任意长度历史 → 固定 token 数\n"
                 "MEM 式『只保留当前帧表示，历史信息经因果时间注意力注入』",
                 fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = "帧数->token: " + ", ".join("%d->%d" % (f, c) for f, c in zip(frames, counts_no_cls))
    text += "\npatch token 数恒定 = %d = O(1) 时空压缩" % counts_no_cls[-1]
    return fig, text


def main() -> None:
    fig1, t1 = plot_block_causal()
    fig2, t2 = plot_spacetime_compression()
    fig1.savefig(OUT / "plot_block_causal.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_block_causal.png")
    print(t1)
    fig2.savefig(OUT / "plot_spacetime_compression.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_spacetime_compression.png")
    print(t2)


if __name__ == "__main__":
    main()
