# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：CSA 两层漏斗 + 重叠 a/b + 保守因果（知识点 3）。

生成 plot_principle_csa_funnel.png —— 把 CSA 一层的完整数据流画成立体结构图：

  输入 token 序列（T）→ 第一层漏斗（KV 压缩 m:1 + 重叠 a/b）→ 压缩块序列
  （n/m 条）→ 第二层漏斗（Lightning Indexer 打分 + 保守因果 + top-k 选择）
  → 命中 top-k 条 → 与滑窗 KV 一起进 Core Attention。

关键机制：
  1. 压缩 = softmax 池化：块内权重 w = softmax(gate + position_bias)
  2. 重叠 = 当前块 A 版本 + 前一块 B 版本，拼起来一起 softmax（跨块竞争）
  3. 保守因果 = query 只能看「完全在自己之前」的块 s < floor(t/m)，
     自己所在块不能看（哪怕块内是过去的 token）→ 这就是 SWA 存在的动机
  4. Indexer = 轻量打分（低秩 query + ReLU + 头加权），挑 top-k

【取证等级】结构示意 + 可见性规则；对应论文 §2.3 与文档 0x07。
"""
from __future__ import annotations

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

XL, XR, YB, YT = 0, 100, 0, 66


def plot_csa_funnel() -> None:
    fig, ax = blank_canvas((15.2, 9.8), (XL, XR), (YB, YT))

    # ============ 输入序列 ============
    label(ax, 4, 61.8, "输入 token 序列", fontsize=10, weight="bold", color="#1565c0", ha="left")
    for i in range(16):
        box(ax, 3 + i * 4.6, 56.5, 4.0, 4.5, f"t{i}", fontsize=6.5,
            fc="#e3f2fd", ec="#1565c0", lw=0.9)
    label(ax, 50, 54.0, "T 个 token，每 token 一条 KV（K 带部分 RoPE）", fontsize=8, color="#8b949e")

    # ============ 第一层漏斗：压缩 ============
    arrow(ax, 50, 56.2, 50, 50.5, color="#1565c0", lw=1.6)
    box(ax, 30, 43.0, 40, 6.5, "第一层漏斗：KV 压缩（m:1 + 重叠 a/b）\n"
                              "w = softmax(gate + position_bias)  →  c = Σ w·kv",
         fontsize=9, fc="#e8f5e9", ec="#238636", lw=1.8)

    # 重叠 a/b 示意
    label(ax, 8, 46.0, "重叠：", fontsize=8, ha="left", color="#8b949e")
    box(ax, 20, 45.0, 6, 3.0, "A块", fontsize=7, fc="#e8f5e9", ec="#238636", lw=1.0)
    box(ax, 27, 45.0, 6, 3.0, "B块", fontsize=7, fc="#e8f5e9", ec="#238636", lw=1.0)
    label(ax, 35, 46.5, "拼起来一起 softmax（跨块竞争）", fontsize=7.5, ha="left", color="#8b949e")

    # 压缩块序列
    arrow(ax, 50, 42.7, 50, 38.0, color="#238636", lw=1.6)
    label(ax, 30, 36.0, "压缩块序列：", fontsize=9, ha="right", color="#238636")
    for i in range(8):
        box(ax, 33 + i * 4.8, 34.0, 4.2, 3.0, f"c{i}", fontsize=7,
            fc="#e8f5e9", ec="#238636", lw=0.9)
    label(ax, 50, 31.0, "n/m 条压缩 KV（每 m 个 token 合成 1 条，块首位置做 RoPE）",
          fontsize=8, color="#8b949e")

    # ============ 第二层漏斗：Indexer + top-k ============
    arrow(ax, 50, 30.6, 50, 25.5, color="#ef8c00", lw=1.6)
    box(ax, 30, 17.0, 40, 8.0, "第二层漏斗：Lightning Indexer\n"
                              "score = Σ_h w_h·ReLU(q_h·c_b)\n"
                              "（低秩 query + ReLU 过滤 + 头加权）",
         fontsize=8.5, fc="#fff3e0", ec="#ef8c00", lw=1.8)
    # 保守因果
    label(ax, 78, 24.0, "保守因果：\ns < floor(t/m)\n(query 所在块不可见)", fontsize=7.5,
          ha="left", color="#c62828")
    arrow(ax, 70, 20.0, 76, 20.5, color="#c62828", lw=1.0)

    # ============ Core Attention ============
    arrow(ax, 50, 16.7, 50, 12.0, color="#238636", lw=1.6)
    box(ax, 24, 5.0, 52, 6.5, "Core Attention：对「top-k 压缩 KV + SWA 滑窗 KV」\n"
                              "做 MQA + attention sink + RMSNorm",
         fontsize=9, fc="#e8f5e9", ec="#238636", lw=1.8)
    label(ax, 50, 1.8, "每个 query 只读 k 条压缩 + W 条未压缩 —— 远低于 n", fontsize=8,
          color="#24292f", weight="bold")

    # 右侧注：为什么两层漏斗省这么多
    label(ax, 93.5, 40.5, "第一层省：", fontsize=7.5, ha="right", color="#238636")
    label(ax, 93.5, 39.2, "KV 条数 n → n/4", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 37.5, "第二层省：", fontsize=7.5, ha="right", color="#ef8c00")
    label(ax, 93.5, 36.2, "每 query 只读 k 条", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 34.0, "合计：", fontsize=7.5, ha="right", color="#c62828")
    label(ax, 93.5, 32.7, "点积数 ≈1000x 下降", fontsize=7.5, ha="right", color="#8b949e")

    fig.suptitle("CSA：压缩 + 稀疏 的两层漏斗（含重叠 a/b 与保守因果）", fontsize=14,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_csa_funnel.png")


def main() -> None:
    plot_csa_funnel()


if __name__ == "__main__":
    main()
