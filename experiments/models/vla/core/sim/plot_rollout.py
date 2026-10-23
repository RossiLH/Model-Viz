# -*- coding: utf-8 -*-
"""【实测图 1/2】闭环 rollout 帧序列 + 轨迹对比

【展示什么】同一个初始状态（seed=200）下，脚本专家与未训练 PI0.5 策略
             各自的闭环表现。上两行是真实渲染的 RGB 帧，下面两栏是
             推杆/方块轨迹与"块到目标距离"随步数的变化。

【为什么这张图重要】它是整个项目"真实运行时驱动"卖点的直接证据：
             图上每个像素都来自 PushBlockEnv 的真实渲染，
             每条曲线都来自真实闭环 rollout，没有任何手工编造。

数据来源：PushBlockEnv + ScriptedPusher / VLAPolicyAgent 真实运行。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import (  # noqa: E402
    C_AUX, C_GREY, C_NEW, C_STD, C_WARN, OUT, build_pi05, make_env, setup_fonts,
)
from core.sim.base import Runtime  # noqa: E402
from models.vla.core.sim.push_block_env import PushBlockEnv  # noqa: E402
from models.vla.core.sim.subscriber import RolloutSubscriber  # noqa: E402
from models.vla.core.sim.teacher import ScriptedPusher  # noqa: E402
from models.vla.core.sim.vla_agent import VLAPolicyAgent  # noqa: E402

setup_fonts()

SEED = 200
MAX_STEPS = 40
N_FRAMES = 5      # 每行展示几帧


def run_and_record(agent, max_steps=MAX_STEPS):
    """跑一回合并录帧，返回 RolloutSubscriber 的 episode dict。"""
    env = make_env(max_steps=max_steps)
    sub = RolloutSubscriber(record_frames=True, frame_stride=1)
    Runtime(env, agent, [sub]).run_episode(max_steps=max_steps, seed=SEED)
    return sub.episodes[0]


def pick_frames(ep, n=N_FRAMES):
    """从帧序列里等距挑 n 帧（含首尾）。"""
    total = len(ep["frames"])
    idx = np.linspace(0, total - 1, min(n, total)).astype(int)
    return idx, [ep["frames"][i] for i in idx]


def main():
    torch.manual_seed(0)

    # ---- 采数据 --------------------------------------------------------
    ep_expert = run_and_record(ScriptedPusher())
    model, adapter = build_pi05()
    ep_model = run_and_record(VLAPolicyAgent(model, adapter))

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(3, N_FRAMES, height_ratios=[1.0, 1.0, 1.25],
                          hspace=0.42, wspace=0.12)

    # ---- 上两行：帧序列 -------------------------------------------------
    for row, (ep, name, color) in enumerate([
        (ep_expert, "脚本专家", C_NEW),
        (ep_model, "PI0.5（未训练）", C_STD),
    ]):
        idx, frames = pick_frames(ep)
        for col, (i, fr) in enumerate(zip(idx, frames)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(fr)
            ax.set_xticks([])
            ax.set_yticks([])
            d = ep["distances"][min(i, len(ep["distances"]) - 1)]
            ax.set_title("t={}  d={:.2f}".format(i, d), fontsize=8.5)
            for sp in ax.spines.values():
                sp.set_color(color)
                sp.set_linewidth(1.6)
            if col == 0:
                ax.set_ylabel(name, fontsize=10, color=color, labelpad=8)

    # ---- 第三行左：XY 轨迹 ---------------------------------------------
    ax1 = fig.add_subplot(gs[2, :2])
    for ep, name, color in [(ep_expert, "专家", C_NEW),
                            (ep_model, "PI0.5 未训练", C_STD)]:
        st = np.stack(ep["states"])
        ax1.plot(st[:, 0], st[:, 1], "-", color=color, lw=1.6,
                 label="{} 推杆".format(name), alpha=0.9)
        ax1.plot(st[:, 2], st[:, 3], "--", color=color, lw=1.6,
                 label="{} 方块".format(name), alpha=0.55)
        ax1.plot(st[0, 0], st[0, 1], "o", color=color, ms=5)
    # 目标位置（两个 rollout 同 seed，目标相同）
    st0 = np.stack(ep_expert["states"])
    ax1.plot(st0[0, 5], st0[0, 6], "*", color=C_WARN, ms=18,
             label="目标", zorder=5)
    circ = plt.Circle((st0[0, 5], st0[0, 6]),
                      PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD,
                      fill=False, ls=":", color=C_WARN, lw=1.2)
    ax1.add_patch(circ)
    ax1.set_xlabel("x（归一化）", fontsize=9)
    ax1.set_ylabel("y（归一化）", fontsize=9)
    ax1.set_title("推杆与方块轨迹（seed={}）".format(SEED), fontsize=10.5)
    ax1.legend(fontsize=7.2, loc="best", framealpha=0.9)
    ax1.grid(alpha=0.25)
    ax1.set_aspect("equal", adjustable="box")

    # ---- 第三行右：距离收敛曲线 ----------------------------------------
    ax2 = fig.add_subplot(gs[2, 2:])
    for ep, name, color in [(ep_expert, "脚本专家", C_NEW),
                            (ep_model, "PI0.5 未训练", C_STD)]:
        d = np.array(ep["distances"])
        ax2.plot(np.arange(len(d)), d, color=color, lw=2.0, label=name)
    ax2.axhline(PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD, ls="--", color=C_WARN,
                lw=1.3, label="成功阈值 {:.2f}".format(
                    PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD))
    ax2.set_xlabel("控制步", fontsize=9)
    ax2.set_ylabel("块到目标距离", fontsize=9)
    ax2.set_title("距离收敛：专家 {} 步达标 vs 未训练策略不收敛".format(
        ep_expert["steps"]), fontsize=10.5)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.25)

    fig.suptitle("闭环 rollout 实测：真实环境渲染 + 真实策略前向", fontsize=13)
    path = OUT / "rollout_frames.png"
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已生成 {}".format(path))
    print("  专家: {} 步, 末距离 {:.3f}, 成功={}".format(
        ep_expert["steps"], ep_expert["final_distance"], ep_expert["success"]))
    print("  PI0.5 未训练: {} 步, 末距离 {:.3f}, 成功={}".format(
        ep_model["steps"], ep_model["final_distance"], ep_model["success"]))


if __name__ == "__main__":
    main()
