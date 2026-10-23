"""Demo 3: 部分 Rollout — 如何处理不同长度的轨迹。

演示 λ-fraction 截断、暂停/恢复、收割、每 token 正则化。
"""

import sys, os
from pathlib import Path

# 本目录位于 experiments/models/llm/kimi_k3/posttrain, 向上 5 级即 Model-Viz 根
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from models.llm.kimi_k3.posttrain.rl.partial_rollout import (
    Trajectory, PartialRolloutScheduler,
    staleness_regularizer, clipped_policy_ratio,
)

def main():
    print("=" * 65)
    print("Demo 3: 部分 Rollout (Partial Rollout)")
    print("=" * 65)

    # ---------------------------------------------------------------
    # 场景: 4 个 prompt, 每个采样 2 个补全, λ=0.5
    # ---------------------------------------------------------------
    print("\n[1] 初始化: 4 prompts × 2 samples = 8 活跃轨迹")
    print("-" * 40)

    sched = PartialRolloutScheduler(n_prompts=4, k_samples=2, lam=0.5)
    print(f"  n_prompts = {sched.n_prompts}")
    print(f"  k_samples = {sched.k_samples}")
    print(f"  λ = {sched.lam}")
    print(f"  target_active = {sched.target_active}")

    # 启动第 1 轮
    batch = sched.start_iteration(new_prompts=[101, 102, 103, 104])
    print(f"\n  第 1 轮启动: {len(batch)} 条轨迹")
    for t in batch:
        print(f"    prompt={t.prompt_id}, sample={t.sample_id}, key={t.key}")

    # ---------------------------------------------------------------
    # 模拟生成: 不同轨迹在不同迭代完成
    # ---------------------------------------------------------------
    print("\n[2] 模拟生成: 不同轨迹在不同时间完成")
    print("-" * 40)

    # 模拟: prompt 101 的两个 sample 先完成, 其他的还在跑
    keys = list(sched.active.keys())
    print(f"  当前活跃轨迹: {len(keys)} 条")

    # 完成 prompt 101 的 sample 0
    sched.active[(101, 0)].done = True
    sched.active[(101, 0)].tokens = [101, 202, 303, 404, 505]
    sched.active[(101, 0)].token_iteration = [1, 1, 1, 1, 1]
    print(f"  完成 prompt=101, sample=0 (tokens={sched.active[(101, 0)].tokens})")

    # 完成 prompt 102 的 sample 0
    sched.active[(102, 0)].done = True
    sched.active[(102, 0)].tokens = [201, 202, 203]
    sched.active[(102, 0)].token_iteration = [1, 1, 1]
    print(f"  完成 prompt=102, sample=0 (tokens={sched.active[(102, 0)].tokens})")

    # 完成 prompt 103 的 sample 0
    sched.active[(103, 0)].done = True
    sched.active[(103, 0)].tokens = [301, 302, 303, 304]
    sched.active[(103, 0)].token_iteration = [1, 1, 1, 1]
    print(f"  完成 prompt=103, sample=0 (tokens={sched.active[(103, 0)].tokens})")

    done_count = sum(1 for t in sched.active.values() if t.done)
    print(f"\n  完成 {done_count}/{len(sched.active)} = {done_count/len(sched.active):.0%}")
    print(f"  λ = {sched.lam}")
    print(f"  should_pause = {sched.should_pause()}")

    # ---------------------------------------------------------------
    # 暂停剩余轨迹
    # ---------------------------------------------------------------
    print("\n[3] 暂停剩余轨迹 (pause_remaining)")
    print("-" * 40)

    paused = sched.pause_remaining()
    print(f"  暂停了 {len(paused)} 条轨迹:")
    for t in paused:
        print(f"    prompt={t.prompt_id}, sample={t.sample_id}, "
              f"paused_at_iter={t.paused_at_iteration}")

    stats = sched.stats()
    print(f"\n  统计: {stats}")

    # ---------------------------------------------------------------
    # 收割已完成的 prompt 组
    # ---------------------------------------------------------------
    print("\n[4] 收割已完成的 prompt 组 (harvest)")
    print("-" * 40)

    # 完成 prompt 101 的 sample 1 (使 101 的 K=2 个样本全部完成)
    sched.active[(101, 1)].done = True
    sched.active[(101, 1)].tokens = [111, 112, 113]
    sched.active[(101, 1)].token_iteration = [1, 1, 1]

    ready = sched.harvest()
    print(f"  收割了 {len(ready)} 个 prompt 组:")
    for group in ready:
        print(f"    prompt={group[0].prompt_id}:")
        for t in group:
            print(f"      sample={t.sample_id}, tokens={t.tokens}, done={t.done}")

    stats = sched.stats()
    print(f"\n  统计: {stats}")

    # ---------------------------------------------------------------
    # 第 2 轮: 恢复暂停的轨迹 + 新 prompt
    # ---------------------------------------------------------------
    print("\n[5] 第 2 轮: 恢复暂停的轨迹 + 新 prompt")
    print("-" * 40)

    batch2 = sched.start_iteration(new_prompts=[105])
    print(f"  第 2 轮启动: {len(batch2)} 条轨迹")
    for t in batch2:
        status = "恢复" if t.paused_at_iteration is None and t.key in sched.active else "新建"
        print(f"    prompt={t.prompt_id}, sample={t.sample_id} ({status})")

    # ---------------------------------------------------------------
    # 每 token 正则化
    # ---------------------------------------------------------------
    print("\n[6] 每 token 正则化 (staleness_regularizer)")
    print("-" * 40)

    # 场景: 新 token 的 ratio≈1 (几乎没变), 旧 token 的 ratio 偏离很大
    logp_new = torch.tensor([[0.0, -1.0, -2.0, -0.5]])
    logp_old = torch.tensor([[-0.05, -0.9, -5.0, -0.5]])  # 第 3 个 token 很旧

    # 逐 token ratio
    ratio = (logp_new - logp_old).exp()
    print(f"  logp_new:  {logp_new.tolist()}")
    print(f"  logp_old:  {logp_old.tolist()}")
    print(f"  ratio:     {ratio.tolist()}")

    # 每 token 正则化
    reg = staleness_regularizer(logp_new, logp_old, epsilon=0.2)
    print(f"  staleness_regularizer(ε=0.2): {reg.item():.4f}")

    # 对比: 如果不做正则化
    raw_loss = (ratio - 1.0).abs().mean()
    print(f"  无正则化 |ratio-1|.mean(): {raw_loss.item():.4f}")
    print(f"  正则化后: {reg.item():.4f} (被 clamp 限制)")

    # ---------------------------------------------------------------
    # PPO Clipped Ratio
    # ---------------------------------------------------------------
    print("\n[7] PPO Clipped Policy Ratio")
    print("-" * 40)

    advantages = torch.tensor([[1.0, -0.5, 0.3, 0.8]])
    loss = clipped_policy_ratio(logp_new, logp_old, advantages, epsilon=0.2)
    print(f"  advantages: {advantages.tolist()}")
    print(f"  clipped_policy_ratio loss: {loss.item():.4f}")

    # ---------------------------------------------------------------
    # Trajectory 的 staleness 计算
    # ---------------------------------------------------------------
    print("\n[8] Trajectory 的陈旧度计算")
    print("-" * 40)

    traj = Trajectory(
        prompt_id=1, sample_id=0,
        tokens=[10, 20, 30, 40, 50],
        token_iteration=[1, 1, 3, 5, 5],  # token 30 在 iter 3 生成, 已经很旧了
    )
    current_iter = 8
    staleness = traj.staleness(current_iter)
    print(f"  当前迭代: {current_iter}")
    print(f"  token 生成迭代: {traj.token_iteration}")
    print(f"  陈旧度: {staleness}")
    print(f"  (值越大表示越旧, 旧 token 的 ratio 会被正则化器 clamp)")

    print("\n" + "=" * 65)
    print("核心要点:")
    print("  1. λ-fraction: 完成 50% 就暂停, 不等 straggler")
    print("  2. 优先恢复: min-heap 按 paused_at_iteration 排序, 防饿死")
    print("  3. 收割: K 个样本全部完成的 prompt 组才能用于策略优化")
    print("  4. 每 token 正则化: 逐 token 独立约束, 旧 token 不炸掉更新")
    print("  5. EMA 指数移动平均: 新 token 的 ratio≈1 不受约束, 旧 token 被 clamp")
    print("=" * 65)

if __name__ == "__main__":
    main()
