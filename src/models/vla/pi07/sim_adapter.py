# -*- coding: utf-8 -*-
"""PI0.7 仿真适配器

把 PushBlockEnv 的观测字典转换为 ``Pi07PolicyModel.sample_actions`` 所需的
张量输入，并把模型输出解码为环境可执行的 2 维动作。

PI0.7 与 PI0.5 的四处签名差异（适配器必须吸收掉）：

    | # | 差异        | PI0.5                | PI0.7                        |
    |---|-------------|----------------------|------------------------------|
    | 1 | 视觉        | ``images (B,3,H,W)`` | ``videos (B,T,3,H,W)``       |
    | 2 | 语言掩码    | 无                   | ``lang_masks (B,L)`` 必需    |
    | 3 | state       | ``(B,D)`` 可选       | ``(B,T,D)`` 必需             |
    | 4 | ``delay``   | ``Tensor``           | ``int``（也接受 Tensor）     |

其中最关键的是 **时间维 T**：PI0.7 吃的是视频历史（``n_obs_steps`` 帧），
而仿真环境每步只给一帧。因此适配器内部维护两个 ``deque`` 缓冲：
帧缓冲与状态缓冲，长度均为 ``n_obs_steps``。

对应设计选择：
- D10：帧缓冲放在适配器而不是 Runtime——"需要几帧历史"是模型属性，
  环境不应该知道下游模型要几帧。
- D05：相机名 ``base_0_rgb``。
- D08：哈希 tokenizer；PI0.7 的 ``text_embed`` 是 ``nn.Embedding(256, ...,
  padding_idx=0)``，与哈希 tokenizer 的 PAD=0 约定一致。

首帧冷启动约定：
``reset()`` 后缓冲是空的。第一次 ``encode`` 时用当前帧**重复填满**缓冲，
并把 ``obs_history_is_pad`` 标成 ``[True, ..., True, False]``——
即"除最后一帧（真实当前帧）外，其余都是填充"。这与 openpi 处理
episode 开头不足 ``n_obs_steps`` 帧的做法一致。
"""

from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.vla.core.sim.hash_tokenizer import hash_tokenize
from models.vla.pi07.config import Pi07Config


class Pi07SimAdapter:
    """PI0.7 策略与 PushBlockEnv 之间的适配层。

    与 ``Pi05SimAdapter`` 同协议（encode / decode / reset），
    但内部**有状态**：维护 ``n_obs_steps`` 长度的帧缓冲与状态缓冲。
    因此每回合开始必须调用 ``reset()``，否则上一回合的历史帧会泄漏进来。
    """

    #: 语言词表大小。PI0.7 的 text_embed 是 nn.Embedding(256, ...)，
    #: 因此哈希 tokenizer 必须取模 256。
    LANG_VOCAB_SIZE = 256

    def __init__(self, config: Pi07Config,
                 device: Optional[torch.device] = None):
        """初始化适配器。

        Args:
            config: PI0.7 配置。用到 ``vision.image_size``、
                    ``vision.n_obs_steps``、``prompt_max_length``、
                    ``state_dim``、``action_horizon``、``action_dim``。
            device: 张量所在设备，默认 cpu。
        """
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

        self.n_obs_steps = config.vision.n_obs_steps
        self.image_size = config.vision.image_size
        self.state_dim = config.state_dim
        self.prompt_max_length = config.prompt_max_length

        # 帧缓冲与状态缓冲：定长队列，满后自动丢弃最旧元素
        self._frames = deque(maxlen=self.n_obs_steps)
        self._states = deque(maxlen=self.n_obs_steps)
        # 记录已经真正观测到的帧数（用于生成 obs_history_is_pad）
        self._n_real = 0

    def reset(self):
        """每回合开始时清空历史缓冲。

        必须调用：否则上一回合末尾的帧会被当作本回合的历史，
        导致策略基于错误的视觉历史决策。
        """
        self._frames.clear()
        self._states.clear()
        self._n_real = 0

    def _preprocess_frame(self, img_np: np.ndarray) -> torch.Tensor:
        """把单帧 uint8 (H,W,3) 处理成 float32 (3,H',W')。

        归一化到 [0,1]，并在尺寸不符时双线性插值到 ``image_size``。
        """
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(
                f"base_0_rgb 必须是 (H,W,3) uint8，当前形状 {img_np.shape}"
            )
        frame = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3,H,W)

        if frame.shape[1] != self.image_size or frame.shape[2] != self.image_size:
            frame = F.interpolate(
                frame.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return frame

    def _preprocess_state(self, obs_state) -> torch.Tensor:
        """把环境 8 维状态零填充到模型的 ``state_dim``（默认 32）。

        环境 state 已归一化到 [-1,1]，与模型状态投影的输入假设一致；
        多出来的维度填 0（对齐官方对不同机器人 DoF 的 zero-pad 做法）。
        """
        state_vec = np.zeros(self.state_dim, dtype=np.float32)
        if obs_state is not None:
            obs_state = np.asarray(obs_state, dtype=np.float32)
            n = min(obs_state.shape[0], self.state_dim)
            state_vec[:n] = obs_state[:n]
        return torch.from_numpy(state_vec)

    def encode(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """把环境观测编码为模型采样所需的 kwargs。

        输出字段：
          - videos:             (1, T, 3, H, W) float32
          - lang_tokens:        (1, L) int64
          - lang_masks:         (1, L) bool
          - state:              (1, T, state_dim) float32
          - obs_history_is_pad: (1, T) bool

        Args:
            obs: 环境观测，格式见 ``PushBlockEnv.get_observation``。

        Returns:
            可直接 unpack 给 ``Pi07PolicyModel.sample_actions`` 的字典。
        """
        frame = self._preprocess_frame(obs["images"]["base_0_rgb"])
        state = self._preprocess_state(obs.get("state"))

        # ---- 冷启动：首帧重复填满缓冲 ----
        if len(self._frames) == 0:
            for _ in range(self.n_obs_steps):
                self._frames.append(frame)
                self._states.append(state)
            self._n_real = 1
        else:
            # 正常推进：追加当前帧，deque 自动淘汰最旧一帧
            self._frames.append(frame)
            self._states.append(state)
            self._n_real = min(self._n_real + 1, self.n_obs_steps)

        # ---- 堆叠成时间维张量 ----
        videos = torch.stack(list(self._frames), dim=0).unsqueeze(0)  # (1,T,3,H,W)
        state_seq = torch.stack(list(self._states), dim=0).unsqueeze(0)  # (1,T,D)

        # ---- obs_history_is_pad：前 (T - n_real) 帧是填充 ----
        # 约定 True = 该时间步是填充（非真实观测）。
        n_pad = self.n_obs_steps - self._n_real
        pad_flags = [True] * n_pad + [False] * self._n_real
        obs_history_is_pad = torch.tensor(
            [pad_flags], dtype=torch.bool, device=self.device
        )  # (1,T)

        # ---- 语言：哈希 tokenizer + mask ----
        prompt = obs.get("prompt", "")
        tokens, mask = hash_tokenize(
            prompt, self.LANG_VOCAB_SIZE, self.prompt_max_length
        )
        lang_tokens = torch.tensor(
            tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        lang_masks = torch.tensor(
            mask.astype(bool), dtype=torch.bool, device=self.device
        ).unsqueeze(0)

        return {
            "videos": videos.to(self.device),
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state_seq.to(self.device),
            "obs_history_is_pad": obs_history_is_pad,
        }

    def decode(self, action_tensor: torch.Tensor) -> np.ndarray:
        """把模型输出的动作张量解码为环境可执行的 2 维 numpy 动作。

        Args:
            action_tensor: ``sample_actions`` 返回值，形状
                           (B, action_horizon, action_dim)。

        Returns:
            ndarray (2,)，取值范围 [-1, 1]。
        """
        if action_tensor.ndim != 3:
            raise ValueError(
                f"action_tensor 必须是 3 维 (B,H,A)，当前 {action_tensor.ndim} 维"
            )
        action = action_tensor[0, 0, :2].detach().cpu().numpy()
        return np.clip(action, -1.0, 1.0)
