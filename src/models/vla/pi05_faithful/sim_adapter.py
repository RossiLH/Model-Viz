# -*- coding: utf-8 -*-
"""PI0.5 仿真适配器

把 PushBlockEnv 的观测字典（images/state/prompt）转换为
``Pi05FaithfulPolicyModel.sample_actions`` 所需的张量输入，
并把模型输出的动作张量解码为环境可执行的 2 维 numpy 动作。

对应设计选择：
- D03：本地直接调用，不照搬 openpi 的 WebSocket 客户端-服务器架构。
- D05：相机名用 ``base_0_rgb``，与 openpi IMAGE_KEYS 对齐。
- D08：用哈希 tokenizer 代替真实 PaliGemma tokenizer，因为 small_scale
  的 vocab_size=256 与真实词表 257152 不匹配，且仓库无 tokenizer 权重。

注意：本适配器不维护动作 chunk 队列。模型一次推理产出
``(1, action_horizon, action_dim)`` 的动作块；``decode`` 只取第 0 步的前 2 维。
chunk 队列的执行策略由 ``ChunkExecutionPolicy``（Task 6）负责。
"""

from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models.vla.core.sim.hash_tokenizer import hash_tokenize
from models.vla.pi05_faithful.config import Pi05FaithfulConfig


class Pi05SimAdapter:
    """PI0.5 策略与 PushBlockEnv 之间的适配层。

    职责：
      1. ``encode``：把 SimObservation 转成 ``sample_actions`` 的 kwargs。
      2. ``decode``：把 ``sample_actions`` 返回的动作张量转成 (2,) numpy 动作。
      3. ``reset``：每回合重置内部状态（当前无状态，预留接口）。
    """

    def __init__(self, config: Pi05FaithfulConfig,
                 device: Optional[torch.device] = None):
        """初始化适配器。

        Args:
            config: PI0.5 配置，必须包含 image_size / vocab_size / text_len /
                    action_dim / action_horizon。
            device: 张量所在设备，默认 cpu。
        """
        self.config = config
        self.device = device if device is not None else torch.device("cpu")

    def reset(self):
        """每回合开始时重置适配器。

        PI0.5 适配器本身无状态（不维护 chunk 队列），
        但保留此接口以符合 Agent/适配器统一约定。
        """
        pass

    def encode(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """把环境观测编码为模型采样所需的 kwargs。

        输出字段：
          - images:       (1, 3, image_size, image_size) float32
          - text_tokens:  (1, text_len) int64
          - state:        (1, action_dim) float32（仅当 proprio_enabled=True 时）

        Args:
            obs: 环境观测，格式见 ``PushBlockEnv.get_observation``。

        Returns:
            可直接 unpack 给 ``Pi05FaithfulPolicyModel.sample_actions`` 的字典。
        """
        # ---- 图像：uint8 (H,W,3) -> float32 (1,3,H,W) ----
        img_np = obs["images"]["base_0_rgb"]  # (H, W, 3) uint8
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(
                f"base_0_rgb 必须是 (H,W,3) uint8，当前形状 {img_np.shape}"
            )

        # 转成张量并归一化到 [0, 1]
        img = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        img = img / 255.0  # (1, 3, H, W)

        # 若环境渲染尺寸与模型期望不一致，双线性插值到目标尺寸
        target_size = self.config.image_size
        if img.shape[2] != target_size or img.shape[3] != target_size:
            img = F.interpolate(
                img,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        img = img.to(self.device)

        # ---- 文本：哈希 tokenizer -> (1, text_len) ----
        prompt = obs.get("prompt", "")
        tokens, _ = hash_tokenize(
            prompt, self.config.vocab_size, self.config.text_len
        )
        text_tokens = torch.tensor(
            tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        result = {
            "images": img,
            "text_tokens": text_tokens,
        }

        # ---- 状态（可选）：仅在 proprio_enabled 时传入 ----
        if self.config.proprio_enabled:
            state_vec = np.zeros(self.config.action_dim, dtype=np.float32)
            obs_state = obs.get("state")
            if obs_state is not None:
                obs_state = np.asarray(obs_state, dtype=np.float32)
                n = min(obs_state.shape[0], self.config.action_dim)
                state_vec[:n] = obs_state[:n]
            result["state"] = torch.tensor(
                state_vec, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        return result

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
        # 取第 0 个 batch、第 0 个时间步、前 2 个动作维度
        action = action_tensor[0, 0, :2].detach().cpu().numpy()
        # 裁剪到环境动作空间
        return np.clip(action, -1.0, 1.0)
