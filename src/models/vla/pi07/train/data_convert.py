# -*- coding: utf-8 -*-
"""PI0.7 数据转换（仿真 episodes + LeRobot -> PI0.7 视频窗口样本）。

把两种数据源（仿真 episodes / LeRobot 数据集）统一转换成 PI0.7 训练样本格式。
每个样本含 videos / state_history / metadata / subgoal 等 PI0.7 模型需要的全部字段。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models.vla.core.sim.hash_tokenizer import hash_tokenize
from models.vla.core.sim.dataset import pad_actions


def convert_episodes_to_pi07(
    episodes: List[Dict[str, Any]],
    *,
    n_obs_steps: int = 6,
    action_horizon: int = 50,
    state_dim: int = 32,
    prompt_max_length: int = 64,
    vocab_size: int = 256,
    n_metadata: int = 4,
    quality: float = 5.0,
    mistake: float = 0.0,
    control_mode: float = 0.0,
    speed_bin_size: int = 500,
    subgoal_keep_prob: float = 0.25,
    subgoal_offset_range: Tuple[int, int] = (10, 30),
    seed: Optional[int] = None,
    camera_key: str = "base_0_rgb",
) -> List[Dict[str, Any]]:
    """把专家轨迹列表转为 PI0.7 训练样本。

    与 PI0.5 build_chunk_dataset 的区别：
    1. 视频窗口：每个样本带 n_obs_steps 帧历史，episode 开头首帧重复填充
    2. 元数据：speed 由 episode 长度分箱得到
    3. 子目标图像：按 subgoal_keep_prob 概率注入未来帧
    """
    if n_obs_steps < 1:
        raise ValueError(f"n_obs_steps 必须 >= 1，得到 {n_obs_steps}")
    if action_horizon < 1:
        raise ValueError(f"action_horizon 必须 >= 1，得到 {action_horizon}")

    rng = np.random.RandomState(seed)
    samples: List[Dict[str, Any]] = []

    for ep in episodes:
        obss = ep["observations"]
        acts = np.asarray(ep["actions"], dtype=np.float32)
        T = len(acts)
        if T == 0:
            continue
        obs_T = min(len(obss), T)

        episode_length = float(T)
        speed_binned = _bin_speed(episode_length, speed_bin_size)

        metadata = np.zeros(n_metadata, dtype=np.float32)
        metadata[0] = speed_binned
        if n_metadata >= 2:
            metadata[1] = quality
        if n_metadata >= 3:
            metadata[2] = mistake
        if n_metadata >= 4:
            metadata[3] = control_mode
        metadata_masks = np.ones(n_metadata, dtype=bool)

        prompt = ep.get("prompt", "")
        text_tokens, text_masks = hash_tokenize(prompt, vocab_size, prompt_max_length)
        text_tokens = text_tokens.astype(np.int64)
        text_masks = text_masks.astype(bool)

        all_frames = _extract_frames(obss[:obs_T], camera_key)
        all_states = _extract_states(obss[:obs_T], state_dim)

        for t in range(T):
            videos, state_history, obs_is_pad = _build_history_window(
                all_frames, all_states, t, n_obs_steps
            )
            actions_chunk, actions_is_pad = _build_action_chunk(
                acts, t, action_horizon, state_dim
            )
            subgoal_image, subgoal_mask = _sample_subgoal(
                all_frames, t, T, subgoal_keep_prob,
                subgoal_offset_range, rng
            )

            samples.append({
                "videos": videos,
                "state_history": state_history,
                "obs_history_is_pad": obs_is_pad,
                "actions": actions_chunk,
                "actions_is_pad": actions_is_pad,
                "text_tokens": text_tokens.copy(),
                "text_masks": text_masks.copy(),
                "metadata": metadata.copy(),
                "metadata_masks": metadata_masks.copy(),
                "subgoal_image": subgoal_image,
                "subgoal_mask": bool(subgoal_mask),
                "response_tokens": None,
                "response_masks": None,
            })

    return samples

def _build_history_window(
    all_frames: np.ndarray,
    all_states: np.ndarray,
    t: int,
    n_obs_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造从 t 往前数 n_obs_steps 帧的历史窗口。

    约定：窗口内从旧到新排列（索引 0 = 最旧，索引 -1 = 当前 t）。
    t < n_obs_steps - 1 时，前面不足的位置用第 0 帧重复填充，
    obs_history_is_pad 对应位置标 True。
    """
    T_all = all_frames.shape[0]
    if t >= T_all:
        raise IndexError(f"t={t} 超出范围 [0, {T_all - 1}]")
    start = max(0, t - n_obs_steps + 1)
    actual_len = t - start + 1
    n_pad = n_obs_steps - actual_len

    real_frames = all_frames[start:t + 1]
    real_states = all_states[start:t + 1]

    if n_pad == 0:
        videos = real_frames
        states = real_states
        is_pad = np.zeros(n_obs_steps, dtype=bool)
    else:
        pad_frame = all_frames[start:start + 1]
        pad_state = all_states[start:start + 1]
        pad_frames = np.repeat(pad_frame, n_pad, axis=0)
        pad_states = np.repeat(pad_state, n_pad, axis=0)
        videos = np.concatenate([pad_frames, real_frames], axis=0)
        states = np.concatenate([pad_states, real_states], axis=0)
        is_pad = np.concatenate([
            np.ones(n_pad, dtype=bool),
            np.zeros(actual_len, dtype=bool),
        ])
    return videos, states, is_pad


def _build_action_chunk(
    all_actions: np.ndarray,
    t: int,
    action_horizon: int,
    model_action_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造从 t 开始的 H 步动作 chunk，末尾不足 hold-last 填充。

    与 PI0.5 build_chunk_dataset 语义一致。
    """
    T = all_actions.shape[0]
    chunk = np.zeros((action_horizon, all_actions.shape[1]), dtype=np.float32)
    is_pad = np.zeros(action_horizon, dtype=bool)
    for k in range(action_horizon):
        idx = t + k
        if idx < T:
            chunk[k] = all_actions[idx]
        else:
            chunk[k] = chunk[k - 1] if k > 0 else 0.0
            is_pad[k] = True
    chunk = pad_actions(chunk, model_action_dim)
    return chunk.astype(np.float32), is_pad


def _sample_subgoal(
    all_frames: np.ndarray,
    t: int,
    T: int,
    keep_prob: float,
    offset_range: Tuple[int, int],
    rng: np.random.RandomState,
) -> Tuple[Optional[np.ndarray], bool]:
    """按概率从未来帧中采样子目标图像。"""
    if rng.rand() >= keep_prob:
        return None, False
    min_off, max_off = offset_range
    actual_max = min(max_off, T - 1 - t)
    if actual_max < min_off:
        offset = max(0, T - 1 - t)
    else:
        offset = rng.randint(min_off, actual_max + 1)
    target_idx = min(t + offset, T - 1)
    return all_frames[target_idx].copy(), True


def _bin_speed(episode_length: float, bin_size: int) -> float:
    """speed 分箱：四舍五入到最近的 bin_size 倍数（论文 500 步一箱）。"""
    return float(round(episode_length / bin_size) * bin_size)


def _extract_frames(
    observations: List[Dict[str, Any]],
    camera_key: str,
) -> np.ndarray:
    """从观测列表提取视频帧，返回 (T, H, W, 3) uint8。"""
    if len(observations) == 0:
        raise ValueError("observations 为空")
    frames = []
    for obs in observations:
        if "images" in obs:
            img = obs["images"][camera_key]
        elif "image" in obs:
            img = obs["image"]
        else:
            raise KeyError(f"观测中找不到图像字段，已知键: {list(obs.keys())}")
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


def _extract_states(
    observations: List[Dict[str, Any]],
    state_dim: int,
) -> np.ndarray:
    """从观测列表提取状态序列，pad 到 state_dim，返回 (T, state_dim) float32。"""
    if len(observations) == 0:
        raise ValueError("observations 为空")
    states = []
    for obs in observations:
        s = np.asarray(obs.get("state", []), dtype=np.float32)
        if s.ndim == 0:
            s = s.reshape(1)
        out = np.zeros(state_dim, dtype=np.float32)
        n = min(s.shape[0], state_dim)
        out[:n] = s[:n]
        states.append(out)
    return np.stack(states, axis=0)

# ------------------------------------------------------------------
# npz 持久化（save / load）
# ------------------------------------------------------------------
def save_pi07_dataset(path: str, samples: List[Dict[str, Any]]) -> None:
    """把 PI0.7 样本列表存成 .npz。

    与 core/sim/dataset.py 的 save_dataset 同构：同名字段堆叠成批张量。
    subgoal_image 是可变长度字段（部分样本 None），单独存 subgoal_mask、
    subgoal_indices 和 subgoal_images（按索引还原）。
    """
    if len(samples) == 0:
        raise ValueError("样本列表为空")
    n = len(samples)
    videos = np.stack([s["videos"] for s in samples])
    state_history = np.stack([s["state_history"] for s in samples])
    obs_history_is_pad = np.stack([s["obs_history_is_pad"] for s in samples])
    actions = np.stack([s["actions"] for s in samples])
    actions_is_pad = np.stack([s["actions_is_pad"] for s in samples])
    text_tokens = np.stack([s["text_tokens"] for s in samples])
    text_masks = np.stack([s["text_masks"] for s in samples])
    metadata = np.stack([s["metadata"] for s in samples])
    metadata_masks = np.stack([s["metadata_masks"] for s in samples])
    subgoal_mask = np.array([s["subgoal_mask"] for s in samples], dtype=bool)

    sg_indices = np.where(subgoal_mask)[0]
    if len(sg_indices) > 0:
        subgoal_images = np.stack(
            [samples[i]["subgoal_image"] for i in sg_indices]
        )
    else:
        h_img = videos.shape[2]
        w_img = videos.shape[3]
        subgoal_images = np.zeros((0, h_img, w_img, 3), dtype=np.uint8)

    prompts = np.array([""] * n)
    np.savez_compressed(
        path,
        videos=videos,
        state_history=state_history,
        obs_history_is_pad=obs_history_is_pad,
        actions=actions,
        actions_is_pad=actions_is_pad,
        text_tokens=text_tokens,
        text_masks=text_masks,
        metadata=metadata,
        metadata_masks=metadata_masks,
        subgoal_mask=subgoal_mask,
        subgoal_indices=sg_indices,
        subgoal_images=subgoal_images,
        prompts=prompts,
    )


def load_pi07_dataset(path: str) -> List[Dict[str, Any]]:
    """从 .npz 读回 PI0.7 样本列表（与 save_pi07_dataset 对称）。"""
    data = np.load(path, allow_pickle=False)
    n = data["videos"].shape[0]
    subgoal_mask = data["subgoal_mask"]
    sg_indices = data.get("subgoal_indices", None)
    sg_images = data.get("subgoal_images", None)

    samples = []
    sg_ptr = 0
    for i in range(n):
        has_sg = bool(subgoal_mask[i])
        sg_img = None
        if has_sg and sg_images is not None and sg_indices is not None:
            sg_img = sg_images[sg_ptr].copy()
            sg_ptr += 1

        samples.append({
            "videos": data["videos"][i],
            "state_history": data["state_history"][i],
            "obs_history_is_pad": data["obs_history_is_pad"][i],
            "actions": data["actions"][i],
            "actions_is_pad": data["actions_is_pad"][i],
            "text_tokens": data["text_tokens"][i],
            "text_masks": data["text_masks"][i],
            "metadata": data["metadata"][i],
            "metadata_masks": data["metadata_masks"][i],
            "subgoal_image": sg_img,
            "subgoal_mask": has_sg,
            "response_tokens": None,
            "response_masks": None,
        })
    return samples
