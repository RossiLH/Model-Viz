# -*- coding: utf-8 -*-
"""仿真订阅者：rollout 数据采集

``core.sim.base.Subscriber`` 的具体实现，用于把 Runtime 主循环里的每步数据
旁路记录下来，供可视化与指标统计使用。

--------------------------------------------------------------------------------
为什么用订阅者而不是直接在循环里 append
--------------------------------------------------------------------------------

直接在 ``Runtime.run_episode`` 里塞记录代码，会让主循环逐渐长成
"仿真 + 日志 + 画图 + 存盘"的大杂烩。而这些需求彼此无关、变更频率也不同：
可视化需要什么字段三天两头在变，仿真逻辑却几乎不动。

用订阅者把它们隔开后：
  - Runtime 保持 30 行以内，只做"取动作 → 步进 → 通知"
  - 想加一种记录（比如录视频帧）就多挂一个订阅者，主循环零改动
  - 测试 Runtime 时挂一个计数订阅者就能断言调用次序，不用 mock

这就是 openpi 官方 ``subscriber.py`` 的设计意图，本项目原样保留。
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Subscriber
from models.vla.core.sim.push_block_env import PushBlockEnv


class RolloutSubscriber(Subscriber):
    """记录闭环 rollout 的逐步数据，供可视化脚本消费。

    记录内容（每回合一条）：
      - frames:    每步的 RGB 图像（可选，图像很占内存，默认不存）
      - states:    每步的 8 维状态
      - actions:   每步执行的 2 维动作
      - rewards:   每步奖励
      - distances: 每步块中心到目标的欧氏距离（成功判据看的就是它）
      - success:   回合是否成功
      - steps:     回合步数
    """

    def __init__(self, record_frames: bool = False,
                 frame_stride: int = 1):
        """初始化

        Args:
            record_frames: 是否保存 RGB 图像。默认 ``False`` ——
                           64x64x3 的图像每步 12KB，200 步 * 20 回合就是 48MB，
                           而多数可视化只需要状态轨迹。要画 rollout 帧序列时才开。
            frame_stride:  开启录帧时的采样间隔（每 N 步存一帧），进一步省内存。
        """
        self.record_frames = record_frames
        self.frame_stride = max(1, int(frame_stride))
        self.episodes: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._step_idx = 0

    # ---- 内部工具 ------------------------------------------------------

    @staticmethod
    def _distance(state: np.ndarray) -> float:
        """从 8 维状态算出"块中心 → 目标"的距离。

        state 布局（全部归一化到 [-1,1]）：
            [ee_x, ee_y, block_x, block_y, block_theta/pi, goal_x, goal_y, contact]
        取索引 2,3（块位置）与 5,6（目标位置）。
        """
        state = np.asarray(state, dtype=np.float32)
        block = state[2:4]
        goal = state[5:7]
        return float(np.linalg.norm(block - goal))

    def _maybe_record_frame(self, obs: Dict[str, Any]) -> None:
        """按 stride 记录一帧（仅在 record_frames 开启时）。"""
        if not self.record_frames:
            return
        if self._step_idx % self.frame_stride != 0:
            return
        self._current["frames"].append(
            np.asarray(obs["images"]["base_0_rgb"], dtype=np.uint8).copy()
        )
        self._current["frame_steps"].append(self._step_idx)

    # ---- Subscriber 钩子 ------------------------------------------------

    def on_episode_start(self, obs: Dict[str, Any]) -> None:
        """开新回合：初始化容器，并把初始状态/距离记为第 0 项。"""
        self._step_idx = 0
        self._current = {
            "frames": [],
            "frame_steps": [],
            "states": [np.asarray(obs["state"], dtype=np.float32).copy()],
            "actions": [],
            "rewards": [],
            # distances 比 actions 多一项：包含初始距离，
            # 这样画收敛曲线时横轴从第 0 步开始，能看到起点有多远
            "distances": [self._distance(obs["state"])],
            "success": False,
            "steps": 0,
        }
        self._maybe_record_frame(obs)

    def on_step(self, obs: Dict[str, Any], action, reward: float,
                done: bool) -> None:
        """每步记录动作、奖励、新状态与新距离。"""
        self._step_idx += 1
        self._current["actions"].append(
            np.asarray(action, dtype=np.float32).copy())
        self._current["rewards"].append(float(reward))
        self._current["states"].append(
            np.asarray(obs["state"], dtype=np.float32).copy())
        self._current["distances"].append(self._distance(obs["state"]))
        self._maybe_record_frame(obs)

    def on_episode_end(self, info: Dict[str, Any]) -> None:
        """回合结束：判定成功与否并归档。

        成功判定不看 ``info["done"]``（``done`` 也可能是被步数上限截断），
        而是直接看**末步距离是否进入阈值** —— 这与环境的成功判据一致，
        也让"截断但其实已经到位"的边界情形被正确计入。
        """
        self._current["steps"] = info.get("steps", self._step_idx)
        self._current["total_reward"] = info.get("total_reward", 0.0)
        final_dist = self._current["distances"][-1]
        self._current["final_distance"] = final_dist
        self._current["success"] = bool(
            final_dist < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD)
        self.episodes.append(self._current)
        self._current = None

    # ---- 汇总统计 ------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """汇总所有回合的指标。

        Returns:
            dict:
              - n_episodes:     回合数
              - success_rate:   成功率
              - mean_steps:     平均步数
              - mean_final_distance: 平均末步距离
                （成功率相同时，这个指标能区分"刚好压线"和"稳稳到位"）
        """
        n = len(self.episodes)
        if n == 0:
            return {"n_episodes": 0, "success_rate": 0.0,
                    "mean_steps": 0.0, "mean_final_distance": 0.0}
        return {
            "n_episodes": n,
            "success_rate": float(np.mean([e["success"] for e in self.episodes])),
            "mean_steps": float(np.mean([e["steps"] for e in self.episodes])),
            "mean_final_distance": float(
                np.mean([e["final_distance"] for e in self.episodes])),
        }
