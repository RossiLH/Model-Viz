# -*- coding: utf-8 -*-
"""PushBlockEnv: 2D 准静态推块仿真环境

本环境是 VLA 策略闭环演示的最小仿真载体。设计取舍：

1. 纯 numpy 实现，不依赖 PyBullet / MuJoCo / gymnasium。
   原因见 spec D01：当前机器未安装任何物理引擎，且教学演示更关注
   "策略如何把像素+语言映射到动作"这一链路，而非物理保真度。

2. 采用准静态（quasi-static）推动模型。
   推杆与方块之间无动量交换，接触时方块即时跟随推杆位移。
   这是 Mason 1986 推动力学在最低阶的近似，足以演示：
   - 接触 / 非接触的区分
   - 偏心推动产生的旋转
   - 目标导向的稀疏/稠密奖励

3. 观测格式对齐 openpi 约定：
   {
     "images": {"base_0_rgb": ndarray (H, W, 3) uint8},
     "state":  ndarray (8,) float32,
     "prompt": str,
   }
   其中 state = [ee_x, ee_y, block_x, block_y, block_theta_norm,
                  goal_x, goal_y, contact_flag]，
   所有位置坐标归一化到 [-1, 1]，theta_norm = theta / pi。

4. 默认 image_size=64，与 PI0.5 small_scale 的图像尺寸一致，
   方便后续直接把观测喂进模型。
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import Environment


class PushBlockEnv(Environment):
    """2D 准静态推块环境。

    坐标系：世界空间是一个 [-1, 1] x [-1, 1] 的矩形工作空间。
            x 轴向右，y 轴向上。渲染时 y 轴翻转以符合图像惯例。

    三个可动物体：
      - 推杆（end-effector）：半径 0.05 的圆盘，由动作直接控制平移。
      - 方块（block）：半边长 0.15 的旋转矩形，只能被推动。
      - 目标区（goal）：半边长 0.12 的方形区域，成功判据只看方块中心
        是否进入目标区。
    """

    #: 方块半边长（世界坐标）
    BLOCK_HALF_SIZE = 0.15
    #: 推杆半径（世界坐标）
    EE_RADIUS = 0.05
    #: 目标区半边长（世界坐标）
    GOAL_HALF_SIZE = 0.12
    #: 每步最大推杆位移（动作 [-1, 1] 映射到此范围）
    MAX_EE_STEP = 0.08
    #: 偏心推动的旋转增益（力矩 -> 角度变化）
    ROTATION_GAIN = 15.0
    #: 默认成功阈值：方块中心到目标中心距离小于该值即成功
    DEFAULT_SUCCESS_THRESHOLD = 0.10

    def __init__(self, image_size: int = 64, max_steps: int = 200,
                 success_threshold: Optional[float] = None):
        """初始化环境参数与渲染辅助数据结构。

        Args:
            image_size: 渲染图像边长，默认 64 对齐 PI0.5 small_scale。
            max_steps: 回合最大步数，达到后强制 done。
            success_threshold: 成功阈值，默认 0.10。
        """
        self._image_size = int(image_size)
        self._max_steps = int(max_steps)
        self._success_threshold = (
            self.DEFAULT_SUCCESS_THRESHOLD
            if success_threshold is None
            else float(success_threshold)
        )

        # 渲染坐标网格：每个像素对应的世界坐标。
        # x 从左到右 [-1, 1]，y 从上到下 [1, -1]（图像 y 向下为正）。
        xs = np.linspace(-1.0, 1.0, self._image_size)
        ys = np.linspace(1.0, -1.0, self._image_size)
        self._px_x, self._px_y = np.meshgrid(xs, ys)

        # 内部物理状态，reset 时填充
        self.rng = np.random.RandomState(0)
        self.ee_pos = np.zeros(2, dtype=np.float64)
        self.block_pos = np.zeros(2, dtype=np.float64)
        self.block_theta = 0.0  # 弧度，范围 [-pi, pi]
        self.goal_pos = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.contact_flag = 0.0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境，随机生成一轮推块任务。

        布局生成策略：
          1. 方块放在中心区域 [-0.8, 0.8]^2，避免贴边。
          2. 目标放在距方块 >= 0.45 的位置，确保短步数测试不会偶然成功。
          3. 推杆放在方块到目标连线的反方向，距离方块中心 0.35~0.55，
             这样脚本专家只需"朝目标方向推"即可完成。
          4. 加入 ±0.6 弧度的角度扰动，让演示数据更有多样性。
          5. 重试机制：若裁剪到工作空间后反而与方块接触，则重新采样。

        Args:
            seed: 随机种子。给定同一 seed 必须产生逐位相同的状态。

        Returns:
            初始观测 dict（格式见模块 docstring）。
        """
        self.rng = np.random.RandomState(seed)

        # 方块位置
        self.block_pos = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float64)

        # 方块朝向
        self.block_theta = float(self.rng.uniform(-np.pi, np.pi))

        # 目标位置：保证离方块足够远
        min_goal_dist = 0.45
        for _ in range(50):
            self.goal_pos = self.rng.uniform(-0.85, 0.85, size=2).astype(np.float64)
            if np.linalg.norm(self.goal_pos - self.block_pos) >= min_goal_dist:
                break
        else:
            # 极少数退化情况：直接放在右下对角线 0.5 处并裁剪
            fallback = self.block_pos + np.array([0.5, -0.5], dtype=np.float64)
            self.goal_pos = np.clip(fallback, -0.85, 0.85)

        # 推杆位置：在"方块 -> 目标"反方向上加扰动
        goal_dir = self.goal_pos - self.block_pos
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        base_angle = np.arctan2(-goal_dir[1], -goal_dir[0])

        self.ee_pos = self.block_pos.copy()
        for _ in range(20):
            angle_jitter = self.rng.uniform(-0.6, 0.6)
            dist = self.rng.uniform(0.35, 0.55)
            ee_angle = base_angle + angle_jitter
            offset = np.array([np.cos(ee_angle), np.sin(ee_angle)], dtype=np.float64) * dist
            candidate = self.block_pos + offset
            candidate = np.clip(candidate, -0.95, 0.95)
            # 检查是否接触：要求表面距离大于推杆半径 + 安全边距
            if not self._is_in_contact(candidate):
                self.ee_pos = candidate
                break
        else:
            # 若 20 次都没找到，退而求其次：直接放在反方向 0.5 处
            self.ee_pos = np.clip(
                self.block_pos - goal_dir * 0.5, -0.95, 0.95
            )

        self.step_count = 0
        self.contact_flag = 0.0

        return self.get_observation()

    def step(self, action) -> Tuple[Dict[str, Any], float, bool]:
        """执行一步动作。

        动作是 2 维 numpy 数组（或序列），每个分量在 [-1, 1] 之间，
        被映射为推杆在世界空间中的位移。超出范围会被裁剪。

        接触处理流程：
          1. 先按动作移动推杆。
          2. 计算推杆中心到旋转方块表面的距离。
          3. 若距离 < 推杆半径，判定为接触：
             - 沿法线方向消除穿透；
             - 把推杆位移在法线方向的分量传给方块；
             - 用接触点相对方块中心的偏移计算力矩，更新朝向。
          4. 裁剪方块位置与角度，保证状态始终在合法范围。

        Args:
            action: 动作，前 2 维有效。

        Returns:
            (obs, reward, done) 三元组。
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] < 2:
            action = np.pad(action, (0, 2 - action.shape[0]))
        action = np.clip(action[:2], -1.0, 1.0)

        old_ee = self.ee_pos.copy()
        ee_delta = action * self.MAX_EE_STEP
        self.ee_pos = np.clip(self.ee_pos + ee_delta, -1.0, 1.0)
        actual_delta = self.ee_pos - old_ee

        self.contact_flag = 0.0

        # 推杆到旋转方块表面的距离
        local_ee = self._rotate(self.ee_pos - self.block_pos, -self.block_theta)
        clamped = np.clip(local_ee, -self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE)
        surface_dist = float(np.linalg.norm(local_ee - clamped))

        if surface_dist < self.EE_RADIUS:
            self.contact_flag = 1.0
            self._resolve_contact(local_ee, clamped, surface_dist, actual_delta)

        # 保证物理量不越界
        self.block_pos = np.clip(self.block_pos, -1.0, 1.0)
        self.block_theta = self._wrap_angle(self.block_theta)

        self.step_count += 1

        reward = self._compute_reward()
        done = self._is_success() or self.step_count >= self._max_steps

        return self.get_observation(), float(reward), bool(done)

    def get_observation(self) -> Dict[str, Any]:
        """返回当前观测（纯读操作，不修改环境状态）。"""
        return {
            "images": {"base_0_rgb": self._render()},
            "state": self._get_state(),
            "prompt": "Push the block to the green goal.",
        }

    def _get_state(self) -> np.ndarray:
        """打包 8 维状态向量。"""
        return np.array(
            [
                float(self.ee_pos[0]),
                float(self.ee_pos[1]),
                float(self.block_pos[0]),
                float(self.block_pos[1]),
                float(self.block_theta / np.pi),  # 归一化到 [-1, 1]
                float(self.goal_pos[0]),
                float(self.goal_pos[1]),
                float(self.contact_flag),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        """稠密奖励：负距离，成功时额外 +1.0。"""
        dist = float(np.linalg.norm(self.block_pos - self.goal_pos))
        reward = -dist
        if dist < self._success_threshold:
            reward += 1.0
        return reward

    def _is_success(self) -> bool:
        """方块中心进入目标阈值即成功。"""
        return float(np.linalg.norm(self.block_pos - self.goal_pos)) < self._success_threshold

    def _is_in_contact(self, ee_pos: np.ndarray) -> bool:
        """给定推杆位置，判断是否接触方块。"""
        local_ee = self._rotate(ee_pos - self.block_pos, -self.block_theta)
        clamped = np.clip(local_ee, -self.BLOCK_HALF_SIZE, self.BLOCK_HALF_SIZE)
        surface_dist = float(np.linalg.norm(local_ee - clamped))
        return surface_dist < self.EE_RADIUS

    def _resolve_contact(self, local_ee: np.ndarray, clamped: np.ndarray,
                         surface_dist: float, ee_delta: np.ndarray) -> None:
        """处理接触：消除穿透、传递平移、计算旋转。

        Args:
            local_ee: 推杆中心在方块局部坐标系中的位置。
            clamped: 方块矩形上距离 local_ee 最近的点（局部坐标）。
            surface_dist: local_ee 到 clamped 的距离。
            ee_delta: 本步推杆实际位移（世界坐标）。
        """
        penetration = self.EE_RADIUS - surface_dist

        if surface_dist > 1e-8:
            normal_local = (local_ee - clamped) / surface_dist
        else:
            # 退化：推杆中心恰好落在 clamped 上，任取一个向外的法线
            normal_local = np.array([1.0, 0.0], dtype=np.float64)

        # 法线转到世界坐标。注意：它从方块表面指向推杆中心。
        normal_world = self._rotate(normal_local, self.block_theta)

        # 接触点（方块表面，世界坐标），用于计算力矩
        contact_world = self.block_pos + self._rotate(clamped, self.block_theta)

        # 1) 消除穿透：推杆嵌入方块，把方块沿 -normal_world 推开
        self.block_pos -= normal_world * penetration

        # 2) 传递推杆运动：只有"推杆朝方块内部移动"的分量才有效
        push_into_block = max(0.0, -np.dot(ee_delta, normal_world))
        if push_into_block > 0.0:
            self.block_pos -= normal_world * push_into_block

            # 3) 偏心产生力矩
            lever_arm = contact_world - self.block_pos
            force = -normal_world * push_into_block
            torque_z = lever_arm[0] * force[1] - lever_arm[1] * force[0]
            self.block_theta += self.ROTATION_GAIN * torque_z

    def _render(self) -> np.ndarray:
        """把当前状态渲染为 (H, W, 3) uint8 图像。

        绘制顺序（从底到顶）：
          1. 深色背景
          2. 绿色虚线目标框
          3. 蓝色旋转方块（接触时边缘高亮为黄色）
          4. 红色推杆圆盘（接触时偏橙）
        """
        size = self._image_size
        canvas = np.full((size, size, 3), 30, dtype=np.uint8)

        # ---- 目标框（绿色虚线） ----
        goal_mask = (
            (np.abs(self._px_x - self.goal_pos[0]) <= self.GOAL_HALF_SIZE)
            & (np.abs(self._px_y - self.goal_pos[1]) <= self.GOAL_HALF_SIZE)
        )
        goal_border = (
            (np.abs(self._px_x - self.goal_pos[0]) >= self.GOAL_HALF_SIZE - 0.03)
            | (np.abs(self._px_y - self.goal_pos[1]) >= self.GOAL_HALF_SIZE - 0.03)
        )
        # 虚线效果：按像素坐标奇偶做间隔
        dash = ((np.arange(size)[:, None] + np.arange(size)[None, :]) % 6) < 3
        canvas[goal_mask & goal_border & dash] = np.array([0, 200, 0], dtype=np.uint8)

        # ---- 旋转方块（蓝色填充 + 深色边框） ----
        local_px = self._rotate(
            np.stack([self._px_x - self.block_pos[0],
                      self._px_y - self.block_pos[1]], axis=-1),
            -self.block_theta,
        )
        in_block = (
            (np.abs(local_px[..., 0]) <= self.BLOCK_HALF_SIZE)
            & (np.abs(local_px[..., 1]) <= self.BLOCK_HALF_SIZE)
        )
        block_border = (
            (np.abs(local_px[..., 0]) >= self.BLOCK_HALF_SIZE - 0.02)
            | (np.abs(local_px[..., 1]) >= self.BLOCK_HALF_SIZE - 0.02)
        )

        if self.contact_flag:
            block_color = np.array([250, 180, 60], dtype=np.uint8)
        else:
            block_color = np.array([60, 120, 220], dtype=np.uint8)
        canvas[in_block] = block_color
        canvas[in_block & block_border] = np.array([30, 80, 180], dtype=np.uint8)

        # ---- 推杆圆盘（红色，接触时偏橙） ----
        dist_sq = (
            (self._px_x - self.ee_pos[0]) ** 2
            + (self._px_y - self.ee_pos[1]) ** 2
        )
        in_ee = dist_sq <= self.EE_RADIUS ** 2
        if self.contact_flag:
            ee_color = np.array([255, 100, 50], dtype=np.uint8)
        else:
            ee_color = np.array([220, 60, 60], dtype=np.uint8)
        canvas[in_ee] = ee_color

        return canvas

    @staticmethod
    def _rotate(vectors: np.ndarray, angle: float) -> np.ndarray:
        """把最后一维为 2 的向量数组绕原点旋转 angle 弧度。"""
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        original_shape = vectors.shape
        flat = vectors.reshape(-1, 2)
        rotated = flat @ rot.T
        return rotated.reshape(original_shape)

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        """把角度归一化到 [-pi, pi]。"""
        while theta > np.pi:
            theta -= 2.0 * np.pi
        while theta < -np.pi:
            theta += 2.0 * np.pi
        return theta
