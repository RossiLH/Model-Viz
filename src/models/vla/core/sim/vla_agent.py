# -*- coding: utf-8 -*-
"""VLA 策略智能体（把 PI0.5 / PI0.7 包装成仿真可用的 Agent）

本模块是整条链路的"接头"：把 ``core.sim.base.Agent`` 的极简接口
（``get_action(obs) -> action``）与 VLA 模型的复杂输入需求
（多路图像、语言 token、状态向量、历史帧、动作 chunk）对接起来。

--------------------------------------------------------------------------------
三层职责划分（为什么要拆成三层而不是写一个大类）
--------------------------------------------------------------------------------

    ┌─────────────────────────────────────────────────────────┐
    │ VLAPolicyAgent   ← 本模块                                │
    │   职责：实现 Agent 接口，串联适配器与执行策略             │
    │   不知道：图像怎么归一化、chunk 队列怎么管               │
    ├─────────────────────────────────────────────────────────┤
    │ SimAdapter (Pi05SimAdapter / Pi07SimAdapter)             │
    │   职责：obs dict ⇄ 模型张量的双向翻译                    │
    │   不知道：仿真怎么跑、动作何时重推理                     │
    ├─────────────────────────────────────────────────────────┤
    │ ChunkExecutionPolicy                                     │
    │   职责：退视界执行——一块 H 步动作执行前 ŝ 步、何时重推理  │
    │   不知道：观测字段叫什么（Task 6 已改为全透传）           │
    └─────────────────────────────────────────────────────────┘

这样切分的收益很实际：三层里任意一层换掉，另两层不用改。
换模型只改适配器；换执行策略（整块/退视界/每步重推理）只改策略层；
换环境（合成 → PyBullet）三层全不用动。

--------------------------------------------------------------------------------
两种执行模式（对应 D09 的动机）
--------------------------------------------------------------------------------

1. **直取模式**（``n_action_steps=None``，默认）：
   每个控制步都跑一次 ``sample_actions``，取 chunk 的第 0 步。
   最灵敏但最费算力，等价于 RTC 论文里的 s=1。
   教学演示与调试用这个——每一步的动作都能和当前画面严格对应上。

2. **chunk 退视界模式**（``n_action_steps=ŝ``）：
   走 ``ChunkExecutionPolicy``，执行 ŝ 步才重推理一次。
   推理次数降到 1/ŝ，是真实部署的做法（π0.7 论文 Ĥ∈{15,25}）。
   可视化里用来对比"推理次数 vs 响应灵敏度"的权衡。
"""

from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import Agent
from models.vla.pi07.modules.chunk_execution import ChunkExecutionPolicy


class VLAPolicyAgent(Agent):
    """把 VLA 策略模型包装成仿真 Agent。

    对 PI0.5 与 PI0.7 **完全通用**——差异全部被适配器吸收了。
    本类只依赖适配器的三方法协议（``encode`` / ``decode`` / ``reset``）
    与模型的 ``sample_actions``，不含任何模型专属分支。
    """

    def __init__(self, model, adapter,
                 n_action_steps: Optional[int] = None,
                 rtc_max_delay: int = 0,
                 num_steps: Optional[int] = None):
        """初始化

        Args:
            model:         VLA 策略模型（须有 ``sample_actions``）
            adapter:       仿真适配器（``Pi05SimAdapter`` 或 ``Pi07SimAdapter``）
            n_action_steps: 执行视界 ŝ。``None`` = 直取模式（每步重推理）；
                           给正整数则走 ``ChunkExecutionPolicy`` 退视界模式。
            rtc_max_delay: 重推理低水位阈值 d（仅 chunk 模式有效）。
                           队列剩余 ≤ d 时提前重推理，剩余动作作为 RTC 冻结前缀。
            num_steps:     去噪步数。``None`` = 用模型配置的
                           ``num_inference_steps``。
        """
        self.model = model
        self.adapter = adapter
        self.n_action_steps = n_action_steps
        self.num_steps = num_steps

        # chunk 模式才构造执行策略；直取模式下 policy 为 None。
        # 注意 ChunkExecutionPolicy 内部维护 deque，是**有状态**的，
        # 所以必须在 reset() 里连带清空（见下）。
        if n_action_steps is not None:
            self.policy = ChunkExecutionPolicy(
                model, n_action_steps=n_action_steps,
                rtc_max_delay=rtc_max_delay,
            )
        else:
            self.policy = None

    def reset(self):
        """回合开始时重置内部状态。

        两处状态都要清：
          1. 适配器的历史帧/状态缓冲（PI0.7 有，PI0.5 是空操作）
          2. chunk 执行队列（chunk 模式才有）

        漏掉任何一处都会造成**跨回合泄漏**：上一回合末尾的历史帧
        或未执行完的动作会污染新回合的第一批决策，让 before/after
        对比实验的数字不可信。
        """
        self.adapter.reset()
        if self.policy is not None:
            # 直接清 deque，而不是重建 policy 对象——重建会丢掉
            # 调用方可能持有的引用（测试就依赖 agent.policy 是同一个对象）
            self.policy._queue.clear()

    def get_action(self, obs: Dict[str, Any]) -> np.ndarray:
        """根据观测返回环境可执行的动作。

        Args:
            obs: 环境观测 dict（格式见 ``core.sim.base`` 模块 docstring）

        Returns:
            ndarray (2,)，取值 [-1, 1]
        """
        # ---- 1. 翻译：obs dict -> 模型 kwargs ----
        inputs = self.adapter.encode(obs)

        # 去噪步数按需覆盖（小规模演示常调小以提速）
        if self.num_steps is not None:
            inputs["num_steps"] = self.num_steps

        # ---- 2. 推理 ----
        # torch.no_grad 是必需的：闭环 rollout 里不需要梯度，
        # 少掉计算图能显著降低 CPU 内存与耗时。
        with torch.no_grad():
            if self.policy is not None:
                # chunk 模式：策略层可能直接从队列弹出，不一定真的推理。
                # 返回形状 (B, A) —— 已经是单步动作。
                action_step = self.policy(**inputs)
                # decode 期望 (B, H, A)，这里补出长度 1 的 chunk 维
                action_tensor = action_step.unsqueeze(1)
            else:
                # 直取模式：每步真推理，取 chunk 第 0 步（由 decode 负责取）
                action_tensor = self.model.sample_actions(**inputs)

        # ---- 3. 翻译回来：动作张量 -> 环境 2 维动作 ----
        return self.adapter.decode(action_tensor)
