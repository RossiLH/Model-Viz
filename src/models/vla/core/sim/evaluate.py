# -*- coding: utf-8 -*-
"""闭环策略评估工具

--------------------------------------------------------------------------------
为什么需要"专家带飞 k 步"这种评估方式
--------------------------------------------------------------------------------

行为克隆（BC）有一个著名的结构性缺陷：**协变量偏移**（covariate shift）。
训练数据全部来自专家轨迹，所以模型只见过"专家会到达的状态"。
一旦闭环里出现小误差、状态漂出这个分布，模型就没有任何"如何回来"的
监督信号，误差便逐步复合，最终跑到完全陌生的状态里胡乱输出。

这带来一个评估上的困难：**闭环成功率会同时受两件事影响** ——
(1) 模型有没有学到专家的映射，(2) 模型能不能从偏离中恢复。
只看成功率 0%，无法区分"根本没学会"和"学会了但撑不住复合误差"。

``evaluate_expert_handoff`` 就是用来拆开这两者的：先让专家开 k 步
（保证此刻状态严格在训练分布内），再把控制权交给模型。
  - 若 k=0 失败而 k 较大时成功率显著上升 → 模型学到了映射，
    问题在复合误差/协变量偏移；
  - 若各个 k 都失败 → 模型根本没学到映射。

这个诊断思路来自模仿学习里 DAgger 一系工作对 BC 失效模式的分析，
是比单一成功率信息量大得多的指标。
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from models.vla.core.sim.push_block_env import PushBlockEnv


def block_goal_distance(state) -> float:
    """从 8 维状态算"块中心 → 目标"距离（环境成功判据看的就是它）。"""
    state = np.asarray(state, dtype=np.float32)
    return float(np.linalg.norm(state[2:4] - state[5:7]))


def evaluate_closed_loop(agent, env_factory: Callable[[], PushBlockEnv],
                         n_episodes: int = 8, seed: int = 200,
                         max_steps: int = 80) -> Dict[str, Any]:
    """标准闭环评估。

    Args:
        agent:        实现 ``get_action`` / ``reset`` 的智能体
        env_factory:  无参可调用，返回一个新环境（每回合新建，避免状态串味）
        n_episodes:   回合数
        seed:         起始种子（第 i 回合用 seed + i）
        max_steps:    每回合步数上限

    Returns:
        dict: success_rate / mean_final_distance / mean_steps /
              final_distances（逐回合，便于画分布）
    """
    dists, steps, succ = [], [], []
    for i in range(n_episodes):
        env = env_factory()
        obs = env.reset(seed=seed + i)
        agent.reset()
        n = 0
        for _ in range(max_steps):
            obs, _, done = env.step(agent.get_action(obs))
            n += 1
            if done:
                break
        d = block_goal_distance(obs["state"])
        dists.append(d)
        steps.append(n)
        succ.append(d < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD)

    return {
        "success_rate": float(np.mean(succ)),
        "mean_final_distance": float(np.mean(dists)),
        "mean_steps": float(np.mean(steps)),
        "final_distances": dists,
    }


def evaluate_expert_handoff(agent, expert, env_factory: Callable[[], PushBlockEnv],
                            handoff_steps: int, n_episodes: int = 8,
                            seed: int = 200, max_steps: int = 80) -> Dict[str, Any]:
    """专家先开 ``handoff_steps`` 步，再把控制权交给模型。

    诊断价值见模块 docstring：把"学没学到映射"与"撑不撑得住复合误差"拆开。

    Args:
        agent:         待评估的策略智能体
        expert:        脚本专家（``ScriptedPusher``）
        env_factory:   无参可调用，返回新环境
        handoff_steps: 交接前由专家执行的步数。0 = 纯模型（等价标准闭环）
        n_episodes:    回合数
        seed:          起始种子
        max_steps:     每回合总步数上限（含专家那 k 步）

    Returns:
        dict: 同 ``evaluate_closed_loop``，另含 ``handoff_steps``
    """
    dists, succ = [], []
    for i in range(n_episodes):
        env = env_factory()
        obs = env.reset(seed=seed + i)
        expert.reset()
        agent.reset()

        done = False
        used = 0
        # ---- 阶段 1：专家开 k 步（把状态送到训练分布内的某一点）----
        for _ in range(min(handoff_steps, max_steps)):
            obs, _, done = env.step(expert.get_action(obs))
            used += 1
            if done:
                break

        # ---- 阶段 2：交给模型 ----
        if not done:
            for _ in range(max_steps - used):
                obs, _, done = env.step(agent.get_action(obs))
                if done:
                    break

        d = block_goal_distance(obs["state"])
        dists.append(d)
        succ.append(d < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD)

    return {
        "handoff_steps": handoff_steps,
        "success_rate": float(np.mean(succ)),
        "mean_final_distance": float(np.mean(dists)),
        "final_distances": dists,
    }


def evaluate_open_loop(model, adapter, samples: List[Dict[str, Any]],
                       n_samples: int = 100) -> Dict[str, Any]:
    """开环动作精度评估（不进仿真，只比对单步动作）。

    闭环成功率受协变量偏移主导，在小规模零预训练设定下会长期为 0，
    因此**开环指标才是"模型有没有在学"的可靠信号**。这里给三个量：

      - ``mean_l2``:      模型首步动作与专家动作的 L2 误差
      - ``baseline_l2``:  "永远输出数据集平均动作"的 L2 误差（朴素基线）
        —— 只有 ``mean_l2`` 明显低于它，才说明模型学到的是**条件映射**
        而不是塌缩到均值
      - ``mean_cos``:     模型动作与专家动作的余弦相似度（方向对不对）
        推块任务里方向比幅值重要得多，所以这个指标最贴近任务语义

    Args:
        model:     VLA 模型
        adapter:   对应适配器
        samples:   ``build_chunk_dataset`` 的输出
        n_samples: 取前多少个样本评估

    Returns:
        dict: mean_l2 / baseline_l2 / mean_cos / cos_within_45deg / n
    """
    import torch
    from models.vla.core.sim.dataset import sample_to_obs

    subset = samples[:n_samples]
    preds, tgts = [], []
    for s in subset:
        adapter.reset()
        inputs = adapter.encode(sample_to_obs(s))
        with torch.no_grad():
            act = model.sample_actions(**inputs)
        preds.append(adapter.decode(act))
        tgts.append(np.asarray(s["actions"][0], dtype=np.float32))

    preds = np.stack(preds)
    tgts = np.stack(tgts)

    l2 = np.linalg.norm(preds - tgts, axis=1)
    # 朴素基线：常量预测（数据集动作均值）
    baseline = np.linalg.norm(tgts.mean(0)[None] - tgts, axis=1)

    # 余弦相似度（跳过零向量，避免 0/0）
    pn = np.linalg.norm(preds, axis=1)
    tn = np.linalg.norm(tgts, axis=1)
    valid = (pn > 1e-6) & (tn > 1e-6)
    cos = np.sum(preds[valid] * tgts[valid], axis=1) / (pn[valid] * tn[valid])

    return {
        "mean_l2": float(l2.mean()),
        "baseline_l2": float(baseline.mean()),
        "mean_cos": float(cos.mean()) if cos.size else 0.0,
        # cos > cos(45°) = 0.707 表示方向基本对
        "cos_within_45deg": float((cos > 0.707).mean()) if cos.size else 0.0,
        "n": len(subset),
    }
