# -*- coding: utf-8 -*-
"""专家演示轨迹 -> 行为克隆训练样本

--------------------------------------------------------------------------------
为什么要"切 chunk"而不是直接用 (obs, action) 单步对
--------------------------------------------------------------------------------

VLA 模型的输出天生是**动作块**（action chunk）：一次预测未来 H 步。
所以监督目标也必须是 H 步动作序列，而不是单个动作。
把专家轨迹切成样本时的对齐关系是：

    样本 t: 观测 obs_t  ->  监督目标 [a_t, a_{t+1}, ..., a_{t+H-1}]

即"看到 obs_t 时，专家接下来 H 步会怎么做"。

轨迹末尾会不够 H 步（比如只剩 2 步却要 H=4 个目标）。两种处理：
  (a) 丢掉末尾样本 —— 简单，但丢掉的正是**最接近成功**的关键状态，
      而这些状态的监督信号对学会"精确对准目标"最重要；
  (b) 零填充 + ``actions_is_pad`` 掩码 —— 保留样本，让 loss 跳过填充步。

本模块采用 (b)，与官方 openpi 的 ``actions_is_pad`` 语义一致
（PI0.7 的 ``forward`` 原生支持这个参数）。PI0.5 的 ``compute_loss``
没有 pad 参数，训练器会在计算 loss 前把填充步的目标动作设为
"上一步真实动作的重复"（hold-last），这比填 0 更贴近物理直觉——
0 动作意味着"急停"，会给模型注入错误的减速信号。

--------------------------------------------------------------------------------
存储格式：.npz
--------------------------------------------------------------------------------

选 .npz 而非 pickle / torch.save：npz 是纯数据格式，不含可执行代码，
跨版本安全；且 numpy 是本项目仅有的硬依赖之一（PyTorch 1.11 + Python 3.7
的环境里能少一个序列化依赖就少一个坑）。

图像以 uint8 存储（而不是归一化后的 float32）：省 4 倍空间，
归一化留给适配器做——保证"训练时的预处理"与"推理时的预处理"
走的是**同一段代码**，从根上排除 train/serve skew。
"""

from typing import Any, Dict, List

import numpy as np


def pad_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    """把环境动作零填充到模型的 ``action_dim``。

    环境动作是 2 维（推杆 dx, dy），模型 action_dim 通常是 32
    （为了兼容各种机器人的自由度）。多余维度填 0，
    与适配器对 state 的处理一致。

    Args:
        actions:    (T, 2) 或 (T, D) 环境动作
        action_dim: 模型动作维度

    Returns:
        (T, action_dim) float32
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions 必须是 2 维 (T, D)，当前 {actions.ndim} 维")
    T, D = actions.shape
    out = np.zeros((T, action_dim), dtype=np.float32)
    n = min(D, action_dim)
    out[:, :n] = actions[:, :n]
    return out


def build_chunk_dataset(episodes: List[Dict[str, Any]],
                        action_horizon: int) -> List[Dict[str, Any]]:
    """把专家轨迹列表切成 chunk 样本列表。

    Args:
        episodes:       ``generate_demos`` 的返回值
        action_horizon: 动作块长度 H

    Returns:
        样本列表，每个样本是 dict：
          - image:          (H_img, W_img, 3) uint8   —— obs_t 的图像
          - state:          (8,) float32              —— obs_t 的状态
          - prompt:         str                       —— 语言指令
          - actions:        (H, 2) float32            —— 未来 H 步专家动作
          - actions_is_pad: (H,) bool                 —— True 表示该步是填充
    """
    if action_horizon < 1:
        raise ValueError("action_horizon 必须 >= 1")

    samples = []
    for ep in episodes:
        acts = ep["actions"]
        obss = ep["observations"]
        T = len(acts)
        for t in range(T):
            # 未来 H 步动作，不足则填充
            chunk = np.zeros((action_horizon, 2), dtype=np.float32)
            is_pad = np.zeros((action_horizon,), dtype=bool)
            for k in range(action_horizon):
                idx = t + k
                if idx < T:
                    chunk[k] = np.asarray(acts[idx], dtype=np.float32)[:2]
                else:
                    # 越界：hold-last（重复最后一个真实动作），并标记为 pad。
                    # 用 hold-last 而不是 0 是有意的：0 动作 = 急停，
                    # 会给模型注入错误的"到终点就刹车"信号；重复末动作
                    # 至少方向上是连续的。有 pad 掩码的模型会直接忽略这些步，
                    # 没有 pad 掩码的（PI0.5）也不会被 0 动作带偏。
                    chunk[k] = chunk[k - 1]
                    is_pad[k] = True

            obs = obss[t]
            samples.append({
                "image": np.asarray(obs["images"]["base_0_rgb"], dtype=np.uint8),
                "state": np.asarray(obs["state"], dtype=np.float32),
                "prompt": obs.get("prompt", ""),
                "actions": chunk,
                "actions_is_pad": is_pad,
            })
    return samples


def save_dataset(path: str, samples: List[Dict[str, Any]]) -> None:
    """把样本列表存成 .npz。

    做法是把同名字段堆叠成批张量（而不是存 N 个独立数组），
    这样加载时是 O(1) 次 IO 而非 O(N) 次，且文件更小。
    prompt 是字符串，用 numpy 的 ``<U`` dtype 存。
    """
    if len(samples) == 0:
        raise ValueError("样本列表为空，无可保存内容")

    np.savez_compressed(
        path,
        images=np.stack([s["image"] for s in samples]),
        states=np.stack([s["state"] for s in samples]),
        actions=np.stack([s["actions"] for s in samples]),
        actions_is_pad=np.stack([s["actions_is_pad"] for s in samples]),
        prompts=np.array([s["prompt"] for s in samples]),
    )


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """从 .npz 读回样本列表（与 ``save_dataset`` 对称）。"""
    data = np.load(path, allow_pickle=False)
    n = data["images"].shape[0]
    samples = []
    for i in range(n):
        samples.append({
            "image": data["images"][i],
            "state": data["states"][i],
            "prompt": str(data["prompts"][i]),
            "actions": data["actions"][i],
            "actions_is_pad": data["actions_is_pad"][i],
        })
    return samples


def sample_to_obs(sample: Dict[str, Any]) -> Dict[str, Any]:
    """把训练样本还原成环境观测格式，以便直接喂给适配器 ``encode``。

    这是"训练与推理共用同一段预处理"的关键：训练时不另写一套
    图像归一化/tokenize 逻辑，而是把样本包装成 obs dict 后
    走适配器的 ``encode`` —— 与闭环推理逐字节相同的代码路径。
    """
    return {
        "images": {"base_0_rgb": sample["image"]},
        "state": sample["state"],
        "prompt": sample["prompt"],
    }
