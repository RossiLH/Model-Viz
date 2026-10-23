# -*- coding: utf-8 -*-
"""Q5-图1：本项目实际运行的注意力掩码（串行两塔 + block-causal）。

直接调用本项目的掩码函数跑出来的真实张量：
- make_block_attn_mask（src/models/vla/pi07/modules/attention.py:34）
- make_suffix_cross_block_attn_mask（同文件:76，suffix 可 cross-attend prefix）

按 model.py embed_prefix 布局（model.py:120-258）构造典型样本的 ar_mask：
  video(4,一块双向) | lang(4,逐token因果) | "State:"(1)+state(3,一块双向)
  | response(2) | metadata(3) | sep(1) | "Subgoal:"(1)+subgoal(2,一块双向)
  | suffix 动作块 chunk=5（块内双向）

关键结论（Q5 核心）：
本项目是串行两塔：prefix_model 跑完 N 层（model.py:357）-> 拼接（:391）
-> 才喂 action_model（:396）。动作块能看全部 prefix = "二次前向"形态。
与官方 interleaved 的差异见 Q5-图2/图3。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import C_STD, OUT, setup_fonts
from models.vla.pi07.modules.attention import (
    block_ids,
    make_block_attn_mask,
    make_suffix_cross_block_attn_mask,
)

BLOCK_COLORS = {
    "video": "#4e342e", "lang": "#1565c0", "state": "#6a1b9a",
    "response": "#00838f", "metadata": "#ef8c00", "sep": "#455a64",
    "subgoal": "#2e7d32", "action": "#c62828",
}


def build_sample_ar():
    video_n, lang_n, state_n = 4, 4, 3
    resp_n, meta_n = 2, 3
    sg_n = 2
    chunk = 5
    ar = [False] * video_n                    # 视频：一块双向
    ar += [True] * lang_n                     # 语言：逐 token 因果
    ar += [True]                              # "State:" 指示（因果）
    ar += [True] + [False] * (state_n - 1)    # 状态块：双向
    ar += [True] * resp_n                     # response：逐 token 因果
    ar += [True] * meta_n                     # metadata：逐字段因果
    ar += [True]                              # 分隔符
    ar += [True]                              # "Subgoal:" 指示（因果）
    ar += [True] + [False] * (sg_n - 1)       # 子目标图像块：双向
    suffix_ar = [True] + [False] * (chunk - 1)  # 动作块内部双向
    prefix_ar = torch.tensor(ar, dtype=torch.bool)
    suffix_ar = torch.tensor(suffix_ar, dtype=torch.bool)
    return prefix_ar, suffix_ar, (video_n, lang_n, state_n, resp_n, meta_n, sg_n, chunk)


def block_label_of(idx, ns):
    video_n, lang_n, state_n, resp_n, meta_n, sg_n, chunk = ns
    segs = [(0, video_n, "video")]
    p = video_n
    segs.append((p, p + lang_n, "lang")); p += lang_n
    segs.append((p, p + 1, "state")); p += 1
    segs.append((p, p + state_n, "state")); p += state_n
    segs.append((p, p + resp_n, "response")); p += resp_n
    segs.append((p, p + meta_n, "metadata")); p += meta_n
    segs.append((p, p + 1, "sep")); p += 1
    segs.append((p, p + 1, "subgoal")); p += 1
    segs.append((p, p + sg_n, "subgoal")); p += sg_n
    for s, e, name in segs:
        if s <= idx < e:
            return name
    return "action"


def main():
    setup_fonts()
    prefix_ar, suffix_ar, ns = build_sample_ar()
    prefix_len = prefix_ar.shape[0]
    ids = block_ids(prefix_ar)[0]
    full = make_suffix_cross_block_attn_mask(prefix_len, suffix_ar, prefix_ar=prefix_ar)
    vis = (full == 0.0)[0].numpy()
    s_total = vis.shape[0]
    labels = ["V0", "V1", "V2", "V3", "L0", "L1", "L2", "L3", "St:",
              "S0", "S1", "S2", "R0", "R1", "M0", "M1", "M2", "sep",
              "Sg:", "G0", "G1", "A0", "A1", "A2", "A3", "A4"]
    block_of = [block_label_of(i, ns) for i in range(s_total)]
    action_see_prefix = vis[prefix_len:, :prefix_len].mean()
    print("s_total", s_total, "prefix_len", prefix_len)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.2),
                             gridspec_kw={"width_ratios": [2.1, 1.0]})
    # ---------------- 左：真实可见性矩阵 ----------------
    ax = axes[0]
    ax.imshow(vis, cmap="Greys", vmin=0, vmax=1)
    for name in ["video", "lang", "state", "response", "metadata", "sep",
                 "subgoal", "action"]:
        start, end = None, None
        for i, blk in enumerate(block_of):
            if blk == name:
                if start is None:
                    start = i
                end = i
        if start is not None:
            ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5),
                                       end - start + 1, end - start + 1,
                                       fill=False, edgecolor=BLOCK_COLORS[name],
                                       linewidth=2.2))
    ax.set_xticks(range(s_total))
    ax.set_yticks(range(s_total))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("key token")
    ax.set_ylabel("query token")
    ax.set_title("本项目真实掩码（真跑 make_suffix_cross_block_attn_mask）\n"
                 "白=可 attend，黑=-inf 屏蔽；红框=动作块，可见全部 prefix",
                 fontsize=11)
    # ---------------- 右：block id + 结论 ----------------
    ax = axes[1]
    ids_np = ids.numpy()
    xs = np.arange(prefix_len)
    ax.bar(xs, ids_np, color=[BLOCK_COLORS[block_of[i]] for i in range(prefix_len)],
           alpha=0.9, edgecolor="white", linewidth=0.4)
    ax.axhline(ids_np.max() + 1, color=C_STD, linestyle="--", linewidth=1.5)
    ax.text(0.02, ids_np.max() + 1.06, "suffix 动作块\n(block_id=max+1,\n恒>全部 prefix)",
            color=C_STD, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels[:prefix_len], rotation=90, fontsize=7)
    ax.set_ylabel("block id (cumsum(ar)-1)")
    ax.set_title("block-causal 的块 id\n视频共块(=-1)，语言/元数据逐token因果，\n"
                 "状态/子目标块内双向，动作块承接末尾",
                 fontsize=11)
    ax.text(0.03, 0.02,
            f"动作行可见 prefix 占比 = {action_see_prefix*100:.0f}%\n"
            "→ suffix 串行读走全部 prefix 表示\n（model.py:391-396 拼接后二次前向）",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="#ffebee", edgecolor=C_STD))
    fig.suptitle("Q5-图1：本项目真实注意力掩码 —— 串行两塔形态（block-causal + suffix 全览 prefix）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / "plot_q5_attn_mask.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")
    print(f"prefix_len={prefix_len}, s_total={s_total}, "
          f"action_see_prefix={action_see_prefix*100:.1f}%")
    print(f"block_ids={ids.numpy().tolist()}")


if __name__ == "__main__":
    main()
