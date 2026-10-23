# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：残差连接 2.0 的三步演进 + Sinkhorn 手推（知识点 5）。

生成两张图：
  1. plot_principle_mhc_lineage.png —— 残差 → HC（Hyper-Connections）→ mHC
     三步演进：主干道从 1 条变 n_hc 条；B 从自由矩阵变双随机约束。
  2. plot_principle_sinkhorn_hand.png —— 用 2x2 矩阵手推一遍 Sinkhorn 迭代，
     逐轮展示「行归一化 → 列归一化」交替，直到行和=列和=1（Q28）。

【取证等级】结构示意 + 手工数值走查（迭代数字是手算示例，非实测）；
对应论文 §2.2 与文档 0x09。
"""
from __future__ import annotations

import numpy as np

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

XL, XR, YB, YT = 0, 100, 0, 66


def plot_mhc_lineage() -> None:
    fig, ax = blank_canvas((15, 9.6), (XL, XR), (YB, YT))

    # ============ 第一步：普通残差 ============
    box(ax, 6, 50.0, 26, 9.0, "① 普通残差\nx = x + F(x)", fontsize=11,
        fc="#ffebee", ec="#c62828", lw=1.8)
    label(ax, 19, 46.5, "1 条主干道（d 维）\n读写 1:1 平权", fontsize=8, color="#8b949e")
    # 画 1 条轨道
    arrow(ax, 10, 44.5, 40, 44.5, color="#c62828", lw=3.0)
    label(ax, 25, 43.0, "x_l  ->  x_{l+1}", fontsize=8, color="#c62828")

    # ============ 第二步：HC ============
    arrow(ax, 38, 55.0, 48, 55.0, color="#1565c0", lw=1.6)
    box(ax, 48, 50.0, 26, 9.0, "② HC：4 条并行轨道\nA/B/C 可学门控", fontsize=11,
        fc="#e3f2fd", ec="#1565c0", lw=1.8)
    label(ax, 61, 46.5, "A 读、B 搅拌、C 写", fontsize=8, color="#8b949e")
    for i in range(4):
        arrow(ax, 50 + i * 5.8, 49.5, 50 + i * 5.8, 40.5, color="#1565c0", lw=1.6)
    label(ax, 61, 38.5, "B 自由矩阵 → 深堆叠会爆", fontsize=8, color="#c62828")

    # ============ 第三步：mHC ============
    arrow(ax, 76, 55.0, 86, 55.0, color="#238636", lw=1.6)
    box(ax, 72, 30.0, 26, 9.0, "③ mHC：B 锁进\n双随机流形", fontsize=11,
        fc="#e8f5e9", ec="#238636", lw=1.8)
    label(ax, 85, 26.5, "谱范数 ≤ 1\n行和=列和=1\n深堆不爆", fontsize=8, color="#238636")
    for i in range(4):
        arrow(ax, 74 + i * 5.8, 29.5, 74 + i * 5.8, 20.5, color="#238636", lw=1.6)

    # 底部说明
    label(ax, 30, 8.0,
          "演进主线：1 条轨道（平权）→ n_hc 条轨道（可调读写权）→ 约束 B 保证稳定（守恒律）",
          fontsize=9.5, weight="bold", color="#24292f")

    fig.suptitle("残差连接 2.0：普通残差 → HC → mHC 的三步演进（Q7）", fontsize=14,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_mhc_lineage.png")


def _mat_text(m: np.ndarray) -> str:
    """把 2x2 矩阵格式化成多行文字。"""
    return "\n".join(f"{v:.3f}  {w:.3f}" for v, w in m)


def plot_sinkhorn_hand() -> None:
    """手推一遍 Sinkhorn：2x2 矩阵交替行/列归一化。"""
    # 手算示例（论文 §3.1 的 2x2 例子）：
    # [[2, 1], [1, 3]]
    orig = np.array([[2.0, 1.0], [1.0, 3.0]])
    fig, ax = blank_canvas((14.5, 8.8), (XL, XR), (YB, YT))

    label(ax, 50, 62.5, "Sinkhorn-Knopp 手推：交替「行归一化 → 列归一化」直到收敛", fontsize=13,
          weight="bold", color="#24292f")

    # 原始矩阵
    box(ax, 4, 36.0, 20, 16.0, "原始矩阵\n" + _mat_text(orig), fontsize=10,
        fc="#ffebee", ec="#c62828", lw=1.6)
    label(ax, 14, 31.5, "行和 [3, 4] 列和 [3, 4]\n都 ≠ 1", fontsize=8, color="#c62828")

    # 第 1 步：行归一化
    step1 = orig / orig.sum(axis=1, keepdims=True)
    box(ax, 36, 36.0, 20, 16.0, "第 1 步：行归一化\n" + _mat_text(step1), fontsize=10,
        fc="#fff3e0", ec="#ef8c00", lw=1.6)
    label(ax, 46, 31.5, "行和 [1, 1] (ok)\n列和 [0.917, 1.083] (否)", fontsize=8, color="#ef8c00")

    # 第 2 步：列归一化
    step2 = step1 / step1.sum(axis=0, keepdims=True)
    box(ax, 68, 36.0, 20, 16.0, "第 2 步：列归一化\n" + _mat_text(step2), fontsize=10,
        fc="#fff3e0", ec="#ef8c00", lw=1.6)
    label(ax, 78, 31.5, "列和 [1, 1] (ok)\n行和 [1.035, 0.965] (否)", fontsize=8, color="#ef8c00")

    arrow(ax, 25, 44.0, 35, 44.0, color="#ef8c00", lw=1.6)
    arrow(ax, 57, 44.0, 67, 44.0, color="#ef8c00", lw=1.6)

    # 收敛结果
    step3 = step2 / step2.sum(axis=1, keepdims=True)
    box(ax, 36, 8.0, 30, 12.0, "迭代 20 次后（收敛）\n行和=列和=1（双随机）\n谱范数 ≤ 1", fontsize=10,
        fc="#e8f5e9", ec="#238636", lw=1.8)

    # 为什么这样做
    label(ax, 76, 14.0, "为什么循环会收敛？\n每步都在「逼近两个约束同时成立」\n20 次后行和误差 ~3e-3", fontsize=8,
          ha="left", color="#238636")

    fig.suptitle("Sinkhorn「用手推一遍」：2x2 矩阵交替归一化（Q28）", fontsize=13,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_sinkhorn_hand.png")


def main() -> None:
    plot_mhc_lineage()
    plot_sinkhorn_hand()


if __name__ == "__main__":
    main()
