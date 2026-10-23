# -*- coding: utf-8 -*-
"""PI0.5 答疑 V2 —— 图组4：动作表示与执行/控制器接口（Q8/Q9）。

【回答的问题】
  Q8 动作表示与语义槽位？—— 动作是连续向量，action_dim=32（论文 18-19 DoF
      统一 padding 到 32），动作块长 H=50。无显式固定语义槽位表，靠
      zero-pad + prompt 控制模式标签 + 逐数据集分位归一化做跨本体弱对齐。
  Q9 动作如何配合底层控制器？—— 模型输出 50Hz 目标位姿/底盘速度，下游只有
      简单 PD 控制器（无 IK、无规划、无碰撞检测）；action chunk 通过队列
      open-loop 消费 + 周期重规划（openpi：open_loop_horizon=8 / replan_steps=5）。

【两张图】
  1. plot_action_representation.png —— 动作块维度布局（32 维中 18-19 有效，
      其余零填充；H=50 的 chunk 从去噪中产出）
  2. plot_action_execution.png —— 推理-执行闭环：chunk → 队列逐步下发 → PD
      控制器 50Hz → 机器人（含 open-loop 消费 + 周期重规划）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX, C_AUX2, C_WARN  # noqa: E402

setup_fonts()

ACTION_DIM = 32      # 本项目 full_scale 配置
H = 50               # 动作块长（论文 H=49，加当前步为 50 步）
DOF = 19             # 论文：state/action 空间 18-19 DoF


def _box(ax, x, y, w, h, text, fc, ec, fs=8.5, tc="black"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.3, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, lw=lw, color=color)
    ax.add_patch(a)


def plot_representation() -> None:
    """图1：动作块维度布局。"""
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")

    ax.text(6, 6.1, "Q8 动作块：连续向量，形状 (B, H=50, action_dim=32)，其中仅 18-19 维有效",
            ha="center", fontsize=11)

    # 顶部：去噪产出 chunk
    _box(ax, 3.0, 5.1, 5.2, 1.0, "Action Expert 10 步去噪\n输出动作块 (1, 50, 32)", "#d9f0e2", C_NEW, fs=8.5)
    _box(ax, 9.2, 5.1, 5.0, 1.0, "chunk 逐步下发：\n每步取第 t 步动作 a_t", "#dbe8f6", C_AUX, fs=8.5)
    _arrow(ax, 5.6, 5.1, 6.7, 5.1, C_NEW)

    # 中间：32 维布局条（18-19 有效 + 零填充）
    ax.text(1.0, 3.55, "action_dim", ha="center", fontsize=10)
    for i in range(ACTION_DIM):
        if i < DOF:
            color, label = "#a5d6a7", ""
        else:
            color, label = "#ef9a9a", ""
        ax.add_patch(Rectangle((1.2 + i * 0.28, 2.7), 0.25, 0.85, fc=color, ec="gray", lw=0.4))
    _box(ax, 5.2, 2.0, 4.6, 0.7, "前 18-19 维：臂/夹爪/升降/底盘（有效）", "#a5d6a7", C_NEW, fs=8)
    _box(ax, 9.6, 2.0, 3.6, 0.7, "其余维度：零填充", "#ef9a9a", C_STD, fs=8)

    # 底部：语义分组
    _box(ax, 2.0, 0.9, 3.2, 0.85, "7 关节 + 1 夹爪\n(臂)", "#c8e6c9", C_NEW, fs=7.6)
    _box(ax, 5.2, 0.9, 3.2, 0.85, "升降 (1-2 DoF)\n+ 底盘 (3 DoF)", "#c8e6c9", C_NEW, fs=7.6)
    _box(ax, 8.4, 0.9, 3.4, 0.85, "末端位姿 (6 DoF)\n(控制模式由 prompt 指定)", "#c8e6c9", C_NEW, fs=7.6)
    ax.text(6, 0.25, "论文未给固定语义槽位表：跨本体靠 zero-pad + <control mode> 标签 + 各数据集分位归一化 [1,1]",
            ha="center", fontsize=8.5, color="#333")

    fig.tight_layout()
    fig.savefig(OUT / "plot_action_representation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_action_representation.png")


def plot_execution() -> None:
    """图2：推理-执行闭环（openpi 部署侧真实参数）。"""
    fig, ax = plt.subplots(figsize=(13, 5.8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")

    ax.text(7, 6.8, "Q9 动作如何配合底层控制器：模型输出目标位姿/速度，PD 控制器跟踪（无 IK、无规划）",
            ha="center", fontsize=11)

    # 左：推理
    _box(ax, 2.2, 5.4, 3.8, 1.2, "策略推理\n10 步去噪 → 动作块 (H, 32)", "#d9f0e2", C_NEW, fs=8.5)
    _box(ax, 6.6, 5.4, 3.6, 1.2, "动作队列 deque\n(openpi: open_loop_horizon=8\n / replan_steps=5)", "#fdeecb", C_WARN, fs=7.8)
    _box(ax, 10.6, 5.4, 3.4, 1.2, "逐步下发\n每 20ms 一步 (50 Hz)", "#dbe8f6", C_AUX, fs=8.5)

    # 中：PD 控制器
    _box(ax, 3.2, 3.4, 4.2, 1.0, "PD 控制器\n跟踪目标位姿/夹爪/底盘速度", "#e3eefb", C_AUX, fs=8.5)
    _box(ax, 8.6, 3.4, 4.0, 1.0, "真机执行\n(关节位置目标, 无 IK/碰撞检测)", "#ece2f7", C_AUX2, fs=8.5)

    # 回环
    _box(ax, 12.2, 3.4, 3.2, 1.0, "观测回传\n(图像+proprio)", "#dbe8f6", C_AUX, fs=8.5)

    _arrow(ax, 4.1, 4.8, 4.1, 3.9, C_NEW)
    _arrow(ax, 10.6, 4.8, 10.6, 3.9, C_AUX)
    _arrow(ax, 5.3, 3.4, 6.6, 3.4, C_AUX)
    _arrow(ax, 8.6, 3.4, 6.6, 3.4, C_AUX)
    _arrow(ax, 2.2, 4.8, 6.6, 4.8, C_NEW, lw=1.0)
    _arrow(ax, 6.6, 4.8, 10.6, 4.8, C_WARN)
    _arrow(ax, 12.2, 2.9, 12.2, 2.2, C_AUX, lw=1.0, style="-")
    _arrow(ax, 12.2, 2.2, 6.2, 2.2, C_AUX, lw=1.0, style="-")
    _arrow(ax, 6.2, 2.2, 4.1, 3.9, C_AUX, lw=1.0)
    ax.text(9.3, 2.0, "chunk 用尽时重新推理（周期重规划）", ha="center", fontsize=8.5, color="#333")

    # 底部注释：频率与参数
    ax.text(7, 0.5,
            "真实参数：π0.5 论文 50Hz（每块 25 步、每 0.5s 重规划）；openpi DROID 15Hz open_loop_horizon=8（0.5s）；"
            "LIBERO replan_steps=5；Aloha 真机 50Hz + ActionChunkBroker",
            ha="center", fontsize=8, color="#333")

    fig.tight_layout()
    fig.savefig(OUT / "plot_action_execution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_action_execution.png")


def main() -> None:
    plot_representation()
    plot_execution()


if __name__ == "__main__":
    main()
