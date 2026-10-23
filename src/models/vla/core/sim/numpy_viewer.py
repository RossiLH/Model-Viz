"""Numpy PushBlockEnv - matplotlib 实时动画

用脚本专家推方块到目标位置，实时显示渲染画面。
纯 numpy 实现，无需 MuJoCo。

运行:
    cd Model-Viz
    python src/models/vla/core/sim/numpy_viewer.py
"""
from __future__ import annotations

import sys
import pathlib
import time

# 从 src/models/vla/core/sim/ 向上 4 层到 src/ 目录
_SRC = pathlib.Path(__file__).resolve().parents[4]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

from models.vla.core.sim.push_block_env import PushBlockEnv
from models.vla.core.sim.teacher import ScriptedPusher


def collect_rollout(env, pusher, max_steps=300, frame_skip=2):
    """跑一回合，收集每步的渲染帧和状态。

    Args:
        env: PushBlockEnv
        pusher: ScriptedPusher
        max_steps: 最大步数
        frame_skip: 每几步存一帧
    Returns:
        frames: list of [H, W, 3] uint8
        states: list of dict
    """
    obs = env.reset(seed=42)
    frames = []
    states = []
    frames.append(env._render().copy())
    states.append({
        "step": 0,
        "block_pos": env.block_pos.copy(),
        "goal_pos": env.goal_pos.copy(),
        "dist": float(np.linalg.norm(env.block_pos - env.goal_pos)),
    })
    done = False
    step = 0
    while not done and step < max_steps:
        action = pusher.get_action(obs)
        obs, reward, done = env.step(action)
        step += 1
        if step % frame_skip == 0 or done:
            frames.append(env._render().copy())
            states.append({
                "step": step,
                "block_pos": env.block_pos.copy(),
                "goal_pos": env.goal_pos.copy(),
                "dist": float(np.linalg.norm(env.block_pos - env.goal_pos)),
            })
    return frames, states, step


def main():
    env = PushBlockEnv(image_size=128, max_steps=300)
    pusher = ScriptedPusher()

    print("=" * 60)
    print("Numpy PushBlockEnv - matplotlib 实时动画")
    print("=" * 60)

    print("正在采集回合数据...")
    t0 = time.perf_counter()
    frames, states, step = collect_rollout(env, pusher, max_steps=300, frame_skip=2)
    t1 = time.perf_counter()

    final_dist = states[-1]["dist"]
    success = final_dist < env.DEFAULT_SUCCESS_THRESHOLD
    print(f"仿真完成: {step} 步,  {'成功' if success else '未到达'}, 最终距离: {final_dist:.4f}")
    print(f"共 {len(frames)} 帧, 采集耗时: {t1 - t0:.2f}s")
    print()

    # matplotlib 动画
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    fig.patch.set_facecolor("black")
    im = ax.imshow(frames[0])
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("", fontsize=12, color="white", pad=10)

    def update(i):
        im.set_data(frames[i])
        s = states[i]
        status = "SUCCESS" if s["dist"] < env.DEFAULT_SUCCESS_THRESHOLD else f"dist {s['dist']:.3f}"
        title.set_text(
            f"Step {s['step']:3d}/{step} | "
            f"Block [{s['block_pos'][0]:+.2f}, {s['block_pos'][1]:+.2f}] | "
            f"{status}"
        )
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=40, repeat=True, blit=True,
    )
    plt.tight_layout()

    # 交互式后端 → 弹窗口
    # 非交互式后端（如 Agg）→ 保存为 GIF 并打开
    backend = matplotlib.get_backend().lower()
    if "agg" in backend or backend in ("svg", "pdf", "ps", "pgf", "cairo"):
        import pathlib
        import tempfile
        out_path = pathlib.Path(tempfile.gettempdir()) / "push_block_numpy.gif"
        print(f"非交互式后端 ({matplotlib.get_backend()})，保存动画到: {out_path}")
        ani.save(str(out_path), writer=PillowWriter(fps=25))
        print("已保存 GIF，正在打开...")
        from PIL import Image
        Image.open(str(out_path)).show()
    else:
        print("动画窗口已打开，关闭窗口退出。")
        plt.show()


if __name__ == "__main__":
    main()
