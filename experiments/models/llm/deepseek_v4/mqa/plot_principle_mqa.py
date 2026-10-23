# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：MQA 注意力基底（知识点 1）。

生成 plot_principle_mqa.png —— 一张「立体」的结构图，把 5 件事画在同一张
画布上，直接对应 V3 文档 0x05 章节的 5 个子题：

  1. 左上：MHA → GQA → MQA 的 KV 头收缩（KV head 数 8 → 4 → 1，KV 只有 1 份）
  2. 左下：q-LoRA 两段投影（dim → q_lora_rank → H·D_h，全宽 vs 低秩参数对比）
  3. 右中：K/V 共享低秩瓶颈（dim → head_dim 单次投影，K 带部分 RoPE、V 不旋转）
  4. 右下：attention sink 槽（每头一个标量 logit，拼在 softmax 末尾）
  5. 右上：分组输出投影 o_a/o_b（先降后升，省 2.6 倍参数）

【取证等级】本图是**结构示意**，不承载实测数字；所有形状/流向/可见性规则
均来自本项目源码（mqa.py）与论文 §2.3，详见 V3 文档 0x05.x。
"""
from __future__ import annotations

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

# 画布坐标（x: 0..100, y: 0..64）
XL, XR, YB, YT = 0, 100, 0, 64


def make_mqa_principle_figure():
    """返回 (fig, text)，供 Gradio 与文档复用。"""
    fig, ax = blank_canvas((15.5, 9.6), (XL, XR), (YB, YT))

    # ================= 顶部：KV 头收缩三兄弟 =================
    label(ax, 22, 61.5, "① MHA → GQA → MQA：KV 头逐级收缩", fontsize=11,
          weight="bold", color="#1565c0")
    label(ax, 2, 59.6, "MHA", fontsize=9, ha="left")
    label(ax, 2, 58.5, "Q/K/V 各 H 头", fontsize=8, ha="left", color="#8b949e")
    # MHA: 8 个 Q 头 + 8 个 KV 头（双排）
    for i in range(8):
        box(ax, 10 + i * 2.4, 51.5, 1.9, 3.0, f"H{i}", fontsize=6,
            fc="#ffebee", ec="#c62828", lw=0.9)
        box(ax, 10 + i * 2.4, 47.2, 1.9, 3.0, f"H{i}", fontsize=6,
            fc="#fff3e0", ec="#ef8c00", lw=0.9)
    label(ax, 31, 49.4, "KV: 8 头 / token", fontsize=8, color="#ef8c00")
    C_GRAY = "#8b949e"
    arrow(ax, 37, 50.5, 42, 50.5, color=C_GRAY, lw=1.4)

    # GQA: 4 个 KV 头，Q 8 头
    label(ax, 46, 59.6, "GQA", fontsize=9, ha="left")
    label(ax, 46, 58.5, "Q 8 头 / KV 4 头", fontsize=8, ha="left", color="#8b949e")
    for i in range(8):
        box(ax, 52 + i * 1.9, 51.5, 1.5, 3.0, f"H{i}", fontsize=6,
            fc="#ffebee", ec="#c62828", lw=0.9)
    for i in range(4):
        box(ax, 54 + i * 3.8, 47.2, 2.8, 3.0, f"K{i}", fontsize=6,
            fc="#fff3e0", ec="#ef8c00", lw=0.9)
    label(ax, 65.5, 49.4, "KV: 4 头 / token", fontsize=8, color="#ef8c00")
    arrow(ax, 68, 50.5, 72, 50.5, color=C_GRAY, lw=1.4)

    # MQA: KV 只有 1 头
    label(ax, 76, 59.6, "MQA", fontsize=9, ha="left")
    label(ax, 76, 58.5, "Q 8 头 / KV 1 头", fontsize=8, ha="left", color="#8b949e")
    for i in range(8):
        box(ax, 81 + i * 1.5, 51.5, 1.2, 3.0, f"H{i}", fontsize=5,
            fc="#e8f5e9", ec="#238636", lw=0.8)
    box(ax, 88, 47.2, 7.5, 3.0, "共享 K/V（1 头）", fontsize=8,
        fc="#e8f5e9", ec="#238636", lw=1.4)
    label(ax, 91.5, 49.4, "KV: 1 头 / token", fontsize=8, color="#238636")
    label(ax, 92, 45.4, "★ 本项目 KV 缓存缩小 H 倍", fontsize=8,
          color="#238636", weight="bold")

    # ================= 中部左：q-LoRA 两段投影 =================
    label(ax, 12, 40.8, "② q-LoRA 两段投影（Q 的重建）", fontsize=11,
          weight="bold", color="#1565c0")
    label(ax, 2, 38.8, "全宽：dim → H·D_h", fontsize=8, ha="left", color="#c62828")
    box(ax, 10, 35.2, 14, 3.0, "W_q: dim×(H·D_h)", fontsize=8, fc="#ffebee", ec="#c62828")
    label(ax, 19, 34.0, "参数 dim·H·D_h", fontsize=7.5, color="#c62828")
    arrow(ax, 27, 36.7, 31, 36.7, color="#8b949e", lw=1.3)
    label(ax, 33, 38.8, "低秩：dim → r → H·D_h", fontsize=8, ha="left", color="#238636")
    box(ax, 41, 35.2, 9, 3.0, "q_a: dim→r", fontsize=8, fc="#e8f5e9", ec="#238636")
    box(ax, 41, 30.8, 9, 3.0, "q_b: r→H·D_h", fontsize=8, fc="#e8f5e9", ec="#238636")
    arrow(ax, 50, 36.7, 50, 34.5, color="#238636", lw=1.2)  # a -> b
    label(ax, 55, 33.2, "参数 dim·r + r·(H·D_h)", fontsize=8, color="#238636")
    label(ax, 55, 31.6, "（探针 E-A：省 75%）", fontsize=7.5, color="#8b949e")

    # ================= 中部右：K/V 共享低秩瓶颈 =================
    label(ax, 62, 40.8, "③ K/V 共享低秩瓶颈", fontsize=11, weight="bold",
          color="#1565c0")
    box(ax, 58, 35.2, 13, 3.0, "kv_proj: dim → D_h", fontsize=8,
        fc="#e8f5e9", ec="#238636")
    box(ax, 58, 30.8, 13, 3.0, "kv_norm: RMSNorm(D_h)", fontsize=8,
        fc="#e8f5e9", ec="#238636")
    arrow(ax, 71, 36.7, 71, 34.5, color="#238636", lw=1.2)
    label(ax, 79.5, 33.0, "一份 K/V 基底", fontsize=8, color="#238636")
    label(ax, 79.5, 31.4, "K 带部分 RoPE；V 不旋转", fontsize=7.5, color="#8b949e")
    label(ax, 79.5, 29.8, "（无需逆 RoPE）", fontsize=7.5, color="#8b949e")

    # ================= 左下：attention sink 槽 =================
    label(ax, 14, 27.2, "④ Attention Sink：每头一个可学标量槽", fontsize=11,
          weight="bold", color="#1565c0")
    # 画一个软 attention 的 logit 行：K0..K3 + sink
    label(ax, 8, 24.2, "logits:", fontsize=8, ha="left")
    for i, t in enumerate(["K0", "K1", "K2", "K3"]):
        box(ax, 20 + i * 5.5, 20.6, 4.6, 3.0, t, fontsize=7,
            fc="#e3f2fd", ec="#1565c0", lw=0.9)
    box(ax, 43, 20.6, 5.0, 3.0, "sink", fontsize=7.5, fc="#fff8e1", ec="#ef8c00", lw=1.4)
    label(ax, 55, 23.6, "← 每头一个标量 logit（mqa.py:62）", fontsize=8, ha="left",
          color="#ef8c00")
    label(ax, 55, 21.4, "模型可把大量权重「倒进」槽里，", fontsize=7.5, ha="left")
    label(ax, 55, 20.0, "对无关 KV「合法摆烂」", fontsize=7.5, ha="left")

    # ================= 右上：分组输出投影 =================
    label(ax, 76, 27.2, "⑤ 分组输出投影（o_a/o_b）", fontsize=11, weight="bold",
          color="#1565c0")
    box(ax, 76, 20.6, 10, 3.0, "o_a: 分组+低秩", fontsize=7.5, fc="#e8f5e9", ec="#238636")
    box(ax, 76, 16.2, 10, 3.0, "o_b: 还原到 dim", fontsize=7.5, fc="#e8f5e9", ec="#238636")
    arrow(ax, 86, 23.6, 86, 19.8, color="#238636", lw=1.2)
    label(ax, 94, 19.6, "省 ~2.6 倍", fontsize=8, color="#238636")
    label(ax, 94, 18.0, "（论文 §2.3）", fontsize=7.5, color="#8b949e")

    # ================= 底部：V4 一句话总结 =================
    label(ax, 50, 7.5,
          "一句话：V4 把「MHA 的 H 份 KV」压成「1 份共享 KV」+「低秩重建 Q」+「sink 槽」，"
          "KV cache 缩小 H 倍 —— 相当于 MLA 的『极简版』（Q3 详解）",
          fontsize=9.5, weight="bold", color="#24292f")

    fig.suptitle("知识点 1：MQA 注意力基底 —— 五个机制一张图", fontsize=14,
                 weight="bold", y=0.99)
    text = (
        "MQA 注意力基底五件事：① KV 头从 MHA 的 H 份收缩到 MQA 的 1 份，KV cache 缩小 H 倍；"
        "② Q 用 q-LoRA 两段投影（dim→r→H·Dh）重建，参数量可省 75%；"
        "③ K/V 共享低秩瓶颈（kv_proj: dim→Dh），所有头共用一份窄 KV；"
        "④ attention sink 每头一个可学标量 logit，接在 softmax 末尾吸收多余权重；"
        "⑤ 输出投影 o_a/o_b 分组低秩，先组内降维再全局还原，省约 2.6 倍参数。"
    )
    return fig, text


def plot_mqa_principle() -> None:
    fig, _ = make_mqa_principle_figure()
    save(fig, "plot_principle_mqa.png")


def main() -> None:
    plot_mqa_principle()


if __name__ == "__main__":
    main()
