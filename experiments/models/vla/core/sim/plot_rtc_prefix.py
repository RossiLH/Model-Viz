# -*- coding: utf-8 -*-
"""【原理图 3/4】RTC 重推理（Receding Horizon Control 低水位机制）

【展示什么】
  Chunk 模式有个问题：如果环境变化比预期快（比如被推物体突然转向），
  队列里剩下的动作全是基于旧观测的，可能已经过时。

  RTC（Receding Horizon Control with re-inference）做了个简单改进：
  给动作队列设一个**低水位（rtc_max_delay）**。
  当队列长度降到低水位以下时，**立刻**用最新观测重新推理，
  不等 chunk 边界。这样视界滞后最多是 low_water 步，
  而不是最坏情况 s_hat 步。

【代码对应】
  VLAPolicyAgent.get_action() 里的 self.rtc_max_delay 判断
  （models/vla/core/sim/vla_agent.py）

【图里讲的故事】
  上半：普通 chunk 模式（每 4 步重推理），最坏滞后 3 步
  下半：RTC 低水位模式（low_water=2），队列剩 2 个就提前推理，
        最坏滞后只有 2 步（以多一次推理为代价换来更短滞后）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _shared import C_AUX, C_GREY, C_NEW, C_STD, C_WARN, OUT, setup_fonts

setup_fonts()

H = 8
S = 4   # 普通 chunk：每 4 步重推理
LOW = 2  # RTC low water：剩 2 个就重推理


def draw_timeline(ax, y_base, label, color):
    """画时间轴刻度，y_base 是轴的 y 位置。"""
    for t in range(16):
        ax.plot([t + 0.5, t + 0.5], [y_base - 0.1, y_base + 0.1],
                "-", color=C_GREY, lw=0.8)
        ax.text(t + 0.5, y_base - 0.25, str(t), ha="center", va="top",
                fontsize=7, color=C_GREY)
    ax.plot([0.5, 15.5], [y_base, y_base], "-", color=C_GREY, lw=1.0)
    ax.text(0.2, y_base, label, fontsize=10, color=color, weight="bold",
            va="center")


def draw_action_block(ax, t, y, h, label, color, alpha=0.9):
    """画一个动作块。"""
    rect = FancyBboxPatch(
        (t + 0.15, y), 0.7, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    ax.text(t + 0.5, y + h / 2, label, ha="center", va="center",
            fontsize=6.5, color="white")


def draw_inf_marker(ax, t, y, label, color=C_AUX):
    """画一个推理时间点标记。"""
    ax.plot([t + 0.5], [y], "v", ms=10, color=color, clip_on=False)
    ax.text(t + 0.5, y - 0.15, label, ha="center", va="top",
            fontsize=7, color=color, weight="bold")


def draw_queue_below(ax, t_inf, queue_len, y_bottom, color, label_text):
    """在某个时刻画"动作队列剩余长度"的可视化条。"""
    x_left = t_inf + 0.5 - queue_len * 0.35
    x_right = t_inf + 0.5 + 0.35
    rect = FancyBboxPatch(
        (x_left, y_bottom - 0.25), x_right - x_left, 0.3,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(t_inf + 0.5, y_bottom - 0.1, label_text,
            ha="center", va="center", fontsize=7, color="white")


def main():
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.5)

    # ============ 上半：普通 chunk ============
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-0.5, 16)
    ax1.set_ylim(-1.2, 2.5)
    ax1.axis("off")
    ax1.set_title("普通 Chunk（每 {} 步推理一次，最坏滞后 {} 步）".format(S, S - 1),
                  fontsize=12, color=C_STD, loc="left")

    draw_timeline(ax1, 0.0, "", C_GREY)

    # 推理点：0, 4, 8, 12
    for t in [0, 4, 8, 12]:
        draw_inf_marker(ax1, t, 1.6, "INF", C_AUX)
        # 产出的 H 步动作（只画前 S 步是"真执行"，后 H-S 步是"预产出但未用"）
        for k in range(H):
            if t + k >= 16:
                break
            alpha = 0.9 if k < S else 0.25
            draw_action_block(ax1, t + k, 0.4, 0.8, "a{}".format(k),
                              C_STD, alpha=alpha)

    # 最坏滞后标注：t=3 的动作
    ax1.annotate("最坏滞后：3 步\na_3 来自 t=0 推理",
                 xy=(3.5, 0.8), xycoords="data",
                 xytext=(5.5, 1.8), textcoords="data",
                 fontsize=8.5, color=C_WARN,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0",
                           ec=C_WARN, lw=1.0),
                 arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5))

    # ============ 下半：RTC 低水位 ============
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-0.5, 16)
    ax2.set_ylim(-1.5, 3.0)
    ax2.axis("off")
    ax2.set_title(
        "RTC 低水位（low_water={}，剩 {} 个动作就重推理，最坏滞后 {} 步）".format(
            LOW, LOW, LOW),
        fontsize=12, color=C_NEW, loc="left")

    draw_timeline(ax2, 0.0, "", C_GREY)

    # 手动模拟 RTC 模式的推理时序
    # t=0: 推理，队列长度 8
    # t=0~2: 消费 3 个，剩 5
    # t=2: 队列长度 5 > LOW=2，不触发
    # t=6: 消费到剩 2，触发重推理（在 t=6 时）
    # 我们展示一个具体的触发场景：t=0 推理，到 t=6 队列剩 2，触发

    # 第 1 次推理（t=0）
    draw_inf_marker(ax2, 0, 2.2, "INF #1", C_AUX)
    for k in range(6):  # 前 6 个被消费（到 t=6 时剩 2）
        draw_action_block(ax2, k, 0.4, 0.8, "a{}".format(k),
                          C_STD, alpha=0.9)
    for k in range(6, 8):  # 剩下 2 个未用（被新推理覆盖）
        draw_action_block(ax2, k, 0.4, 0.8, "a{}".format(k),
                          C_GREY, alpha=0.3)

    # 触发点标注（t=6，队列剩 2，达到低水位）
    ax2.plot([6.5, 6.5], [0.0, 1.6], "--", color=C_WARN, lw=1.5)
    ax2.text(6.5, 1.75, "低水位触发！\n队列剩 {} 个".format(LOW),
             ha="center", va="bottom", fontsize=8.5, color=C_WARN,
             weight="bold")

    # 第 2 次推理（t=6 触发，产出 a0..a7 基于 t=6 的最新观测）
    draw_inf_marker(ax2, 6, 2.2, "INF #2", C_NEW)
    for k in range(6):
        if 6 + k >= 16:
            break
        draw_action_block(ax2, 6 + k, 0.4, 0.8, "a{}'".format(k),
                          C_NEW, alpha=0.9)

    # 第 3 次推理（t=12 触发）
    draw_inf_marker(ax2, 12, 2.2, "INF #3", C_NEW)
    for k in range(4):
        if 12 + k >= 16:
            break
        draw_action_block(ax2, 12 + k, 0.4, 0.8, "a{}''".format(k),
                          C_NEW, alpha=0.9)

    # 对比标注：a_6 之前是 6 步前的推理，现在是 0 步前
    ax2.annotate("a_6 现在用 t=6 的最新观测\n（视界滞后 0 步，而非 6 步）",
                 xy=(6.5, 0.8), xycoords="data",
                 xytext=(9, 2.2), textcoords="data",
                 fontsize=8.5, color=C_NEW,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9",
                           ec=C_NEW, lw=1.0),
                 arrowprops=dict(arrowstyle="->", color=C_NEW, lw=1.5))

    # 底部：trade-off 总结
    ax2.text(0.2, -0.9,
             "trade-off：以多 1 次推理（16 步里 3 次 vs 4 次）为代价，"
             "把最坏视界滞后从 {} 步降到 {} 步".format(S - 1, LOW),
             fontsize=9.5, color=C_AUX,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f3e5f5",
                       ec=C_AUX, lw=1.0))

    fig.suptitle(
        "RTC 重推理低水位机制：队列剩 low_water 个就提前重推理",
        fontsize=13.5, y=0.98)
    path = OUT / "rtc_prefix.png"
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已生成 {}".format(path))


if __name__ == "__main__":
    main()
