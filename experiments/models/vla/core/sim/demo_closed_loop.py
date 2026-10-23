# -*- coding: utf-8 -*-
"""VLA 仿真闭环 demo —— 端到端跑通验证

【做什么】把 Task 1-8 的全部产物串起来实跑一遍，打印可核对的数字：
  1. 环境与专家：脚本专家跑 10 回合，报成功率（预期 100%）
  2. PI0.5 闭环：未训练模型跑 1 回合，报步数与末步距离（预期"跑得动但不成功"）
  3. PI0.7 闭环：同上，验证视频历史缓冲在闭环中正常推进
  4. chunk 退视界：ŝ=4 跑 12 步，报真实推理次数（预期 3 次而非 12 次）
  5. 行为克隆：PI0.5 训 60 步，报 loss 首尾对比
  6. 训练后闭环：报末步距离相对未训练模型的变化

【为什么要有这个脚本】测试保证"不崩"，demo 保证"数字合理"。
未训练模型成功率接近 0、训练后有可见改善，这两个事实才是
"整条链路真的在学东西"的证据 —— 单看测试通过是看不出来的。
"""
import numpy as np
import torch

from _shared import (  # noqa: E402
    ACTION_HORIZON, build_pi05, build_pi07, make_env, IMAGE_SIZE,
)
from core.sim.base import Runtime  # noqa: E402
from models.vla.core.sim.bc_train import train_bc  # noqa: E402
from models.vla.core.sim.dataset import build_chunk_dataset  # noqa: E402
from models.vla.core.sim.subscriber import RolloutSubscriber  # noqa: E402
from models.vla.core.sim.teacher import ScriptedPusher, generate_demos  # noqa: E402
from models.vla.core.sim.vla_agent import VLAPolicyAgent  # noqa: E402


def section(title):
    print("\n" + "=" * 64)
    print(" " + title)
    print("=" * 64)


def run_agent_episodes(agent, n_episodes=3, seed=0, max_steps=60):
    """跑若干回合，返回 RolloutSubscriber 的汇总。"""
    env = make_env(max_steps=max_steps)
    sub = RolloutSubscriber()
    runtime = Runtime(env, agent, [sub])
    for i in range(n_episodes):
        runtime.run_episode(max_steps=max_steps, seed=seed + i)
    return sub


def main():
    torch.manual_seed(0)

    # ---------------------------------------------------------------- 1
    section("[1/6] 环境 + 脚本专家（10 回合）")
    env = make_env()
    sub = RolloutSubscriber()
    runtime = Runtime(env, ScriptedPusher(), [sub])
    for i in range(10):
        runtime.run_episode(seed=i)
    s = sub.summary()
    print("  成功率      : {:.0%}".format(s["success_rate"]))
    print("  平均步数    : {:.1f}".format(s["mean_steps"]))
    print("  平均末距离  : {:.4f}".format(s["mean_final_distance"]))

    # ---------------------------------------------------------------- 2
    section("[2/6] PI0.5 闭环（未训练，直取模式，3 回合）")
    model05, adapter05 = build_pi05()
    n_params = sum(p.numel() for p in model05.parameters())
    print("  参数量      : {:,}".format(n_params))
    agent05 = VLAPolicyAgent(model05, adapter05)
    sub05 = run_agent_episodes(agent05, n_episodes=3, seed=0, max_steps=40)
    s05 = sub05.summary()
    print("  成功率      : {:.0%}  (未训练，接近 0 是预期)".format(s05["success_rate"]))
    print("  平均末距离  : {:.4f}".format(s05["mean_final_distance"]))

    # ---------------------------------------------------------------- 3
    section("[3/6] PI0.7 闭环（未训练，含 3 帧视频历史）")
    model07, adapter07 = build_pi07()
    print("  参数量      : {:,}".format(sum(p.numel() for p in model07.parameters())))
    print("  n_obs_steps : {}".format(model07.config.vision.n_obs_steps))
    agent07 = VLAPolicyAgent(model07, adapter07)
    sub07 = run_agent_episodes(agent07, n_episodes=3, seed=0, max_steps=40)
    s07 = sub07.summary()
    print("  成功率      : {:.0%}".format(s07["success_rate"]))
    print("  平均末距离  : {:.4f}".format(s07["mean_final_distance"]))
    print("  帧缓冲长度  : {} (回合结束后仍满)".format(len(adapter07._frames)))

    # ---------------------------------------------------------------- 4
    section("[4/6] chunk 退视界执行（H={}, s_hat=4）".format(ACTION_HORIZON))
    model_c, adapter_c = build_pi05()
    calls = {"n": 0}
    orig = model_c.sample_actions

    def counting(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    model_c.sample_actions = counting
    agent_c = VLAPolicyAgent(model_c, adapter_c, n_action_steps=4)
    env_c = make_env()
    obs = env_c.reset(seed=0)
    agent_c.reset()
    for _ in range(12):
        obs, _, done = env_c.step(agent_c.get_action(obs))
        if done:
            break
    print("  控制步数    : 12")
    print("  真实推理次数: {}  (直取模式需 12 次)".format(calls["n"]))
    print("  节省比例    : {:.0%}".format(1 - calls["n"] / 12))

    # ---------------------------------------------------------------- 5
    section("[5/6] 行为克隆训练（PI0.5，60 步）")
    demos = generate_demos(8, seed=0,
                           env_kwargs={"image_size": IMAGE_SIZE, "max_steps": 100})
    samples = build_chunk_dataset(demos, action_horizon=ACTION_HORIZON)
    print("  专家轨迹    : {} 条".format(len(demos)))
    print("  训练样本    : {} 个".format(len(samples)))

    model_bc, adapter_bc = build_pi05()
    losses = train_bc(model_bc, adapter_bc, samples, n_steps=60,
                      batch_size=16, lr=1e-3, seed=0, log_every=20)
    head = float(np.mean(losses[:5]))
    tail = float(np.mean(losses[-5:]))
    print("  前 5 步均值 : {:.4f}".format(head))
    print("  后 5 步均值 : {:.4f}".format(tail))
    print("  下降比例    : {:.1%}".format(1 - tail / head))

    # ---------------------------------------------------------------- 6
    section("[6/6] 训练前 vs 训练后闭环对比（PI0.5，5 回合）")
    model_un, adapter_un = build_pi05()
    sub_before = run_agent_episodes(
        VLAPolicyAgent(model_un, adapter_un), n_episodes=5, seed=200, max_steps=60)
    sub_after = run_agent_episodes(
        VLAPolicyAgent(model_bc, adapter_bc), n_episodes=5, seed=200, max_steps=60)
    b, a = sub_before.summary(), sub_after.summary()
    print("  {:<14}{:>12}{:>12}".format("指标", "训练前", "训练后"))
    print("  {:<14}{:>12.0%}{:>12.0%}".format(
        "成功率", b["success_rate"], a["success_rate"]))
    print("  {:<14}{:>12.4f}{:>12.4f}".format(
        "平均末距离", b["mean_final_distance"], a["mean_final_distance"]))
    print("  {:<14}{:>12.1f}{:>12.1f}".format(
        "平均步数", b["mean_steps"], a["mean_steps"]))

    section("Demo 完成 —— 所有环节跑通")


if __name__ == "__main__":
    main()
