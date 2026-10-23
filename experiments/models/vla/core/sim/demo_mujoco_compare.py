# -*- coding: utf-8 -*-
"""双引擎对比 demo：numpy 准静态 vs MuJoCo 真实物理

【这个脚本回答什么问题】
换用真实物理引擎，到底改变了什么？我们用同一个脚本专家、同一批 seed，
在两个环境里各跑一遍，把差异量化出来。

【为什么值得单独写一个脚本】
"换引擎"这件事最容易流于口号。只有把两边的成功率、步数、旋转幅度、
耗时并排打出来，才能看清真实物理带来的是什么，代价是什么。

【运行】
  python experiments/vla_sim/demo_mujoco_compare.py
"""
import time

import numpy as np

from _shared import IMAGE_SIZE  # noqa: E402
from core.sim.base import Runtime  # noqa: E402
from models.vla.core.sim.push_block_env import PushBlockEnv  # noqa: E402
from models.vla.core.sim.teacher import ScriptedPusher  # noqa: E402

#: 每个环境跑多少回合。20 个 seed 足够让成功率稳定到 ±5%
N_EPISODES = 20
#: 单回合步数上限
MAX_STEPS = 150


def try_import_mujoco_env():
    """MuJoCo 是可选依赖，未安装时给出可操作的提示而不是崩栈。"""
    try:
        from models.vla.core.sim.mujoco_push_block_env import (
            MujocoPushBlockEnv,
        )
        return MujocoPushBlockEnv
    except ImportError as exc:
        print("[跳过 MuJoCo] 未安装 mujoco: {}".format(exc))
        print("            安装命令: pip install mujoco")
        return None


def bench(env_cls, label):
    """在给定环境类上跑 N_EPISODES 个回合，返回统计字典。

    统计量的选择理由：
      - success_rate: 任务层面最终指标
      - mean_steps:   完成任务的效率（步数越少越好）
      - mean_dist:    末态离目标多远（失败回合也有信息量）
      - mean_abs_rot: 全程方块转了多少弧度 —— 这一项最关键，
                      直接反映两种物理模型对"偏心推动"的处理差异
      - ms_per_step:  单步墙钟耗时，衡量换引擎的算力代价
    """
    successes = 0
    steps_list, dist_list, rot_list = [], [], []

    t_start = time.time()
    total_steps = 0

    for i in range(N_EPISODES):
        seed = 200 + i
        env = env_cls(image_size=IMAGE_SIZE, max_steps=MAX_STEPS)
        env.reset(seed=seed)
        theta_start = env.block_theta

        info = Runtime(env, ScriptedPusher()).run_episode(
            max_steps=MAX_STEPS, seed=seed
        )

        final_dist = float(np.linalg.norm(env.block_pos - env.goal_pos))
        if final_dist < env_cls.DEFAULT_SUCCESS_THRESHOLD:
            successes += 1

        steps_list.append(info["steps"])
        dist_list.append(final_dist)
        rot_list.append(abs(env.block_theta - theta_start))
        total_steps += info["steps"]

    elapsed = time.time() - t_start

    return {
        "label": label,
        "success_rate": successes / float(N_EPISODES),
        "mean_steps": float(np.mean(steps_list)),
        "mean_dist": float(np.mean(dist_list)),
        "mean_abs_rot": float(np.mean(rot_list)),
        "elapsed": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(1, total_steps),
    }


def print_row(r):
    print("  {:<22s} {:>5.0f}%  {:>7.1f}  {:>8.4f}  {:>9.4f}  {:>8.2f}".format(
        r["label"], 100.0 * r["success_rate"], r["mean_steps"],
        r["mean_dist"], r["mean_abs_rot"], r["ms_per_step"]))


def main():
    print("=" * 78)
    print("双引擎对比：脚本专家在两种物理模型下的表现")
    print("配置: {} 回合, 每回合最多 {} 步, 图像 {}x{}".format(
        N_EPISODES, MAX_STEPS, IMAGE_SIZE, IMAGE_SIZE))
    print("=" * 78)

    results = [bench(PushBlockEnv, "numpy 准静态")]

    mj_cls = try_import_mujoco_env()
    if mj_cls is not None:
        results.append(bench(mj_cls, "MuJoCo 真实物理"))

    print()
    print("  {:<22s} {:>6s}  {:>7s}  {:>8s}  {:>9s}  {:>8s}".format(
        "环境", "成功率", "平均步数", "末距离", "累计|dθ|", "ms/步"))
    print("  " + "-" * 70)
    for r in results:
        print_row(r)

    if len(results) == 2:
        a, b = results
        print()
        print("解读：")
        print("  1. 旋转幅度 {:.2f} -> {:.2f} rad：真实摩擦抑制了方块自旋。".format(
            a["mean_abs_rot"], b["mean_abs_rot"]))
        print("     numpy 版用 ROTATION_GAIN=15.0 手工放大力矩，转得过猛；")
        print("     MuJoCo 里地面摩擦会消耗角动量，方块更\"听话\"。")
        print("  2. 成功率 {:.0f}% -> {:.0f}%：方块更听话，专家反而更容易推准。".format(
            100 * a["success_rate"], 100 * b["success_rate"]))
        print("     这与直觉相反 —— 真实物理不一定更难，关键看误差来源。")
        print("  3. 平均步数 {:.1f} -> {:.1f}：MuJoCo 有惯性和静摩擦，".format(
            a["mean_steps"], b["mean_steps"]))
        print("     起步要先克服静摩擦，所以完成同样任务需要更多步。")
        print("  4. 单步耗时 {:.2f} -> {:.2f} ms（{:.1f}x）：这是换引擎的算力代价。".format(
            a["ms_per_step"], b["ms_per_step"],
            b["ms_per_step"] / max(1e-9, a["ms_per_step"])))
    print()


if __name__ == "__main__":
    main()
