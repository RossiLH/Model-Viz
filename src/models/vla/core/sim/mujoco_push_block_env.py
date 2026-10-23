# -*- coding: utf-8 -*-
"""MujocoPushBlockEnv: 基于 MuJoCo 物理引擎的 2D 推块环境

【定位】
PushBlockEnv 的 MuJoCo 版替代品。接口完全对齐，语义尽量对齐，
上层代码（ScriptedPusher / VLAPolicyAgent / BC 训练器 / 评估）
零改动即可运行。

【与 numpy 版的核心差异（必须知道）】
1. 物理是真实的：接触、摩擦、惯性都由 MuJoCo 解算，
   不再是准静态几何近似。因此推动会有滑移、方块会有惯性，
   脚本专家成功率会比 numpy 版低一些（正常现象）。
2. 偏心推动产生的旋转来自真实接触力学，不再靠 ROTATION_GAIN 手工算力矩。
   这是换 MuJoCo 最实质的收益，也是"为什么用真物理引擎"的教学答案。
3. state 前 7 维语义、坐标范围 [-1, 1]、成功阈值、类常量全部对齐，
   保证上层代码复用。

【世界坐标约定】
与 PushBlockEnv 相同：世界空间是 [-1, 1] x [-1, 1] 的方形工作空间，
x 向右、y 向上。MuJoCo 内部用米制，但通过 WORLD_SCALE 换算后，
所有对外接口（state / reset / step 的动作）都使用归一化坐标。
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from .base import Environment
import mujoco  # type: ignore


class MujocoPushBlockEnv(Environment):
    """MuJoCo 版推块环境。

    对外接口与 PushBlockEnv 完全一致（含三个类常量），保证专家/适配器/
    训练器全部零改动复用。内部用 MuJoCo 做真实物理求解，
    用 WORLD_SCALE 把 [-1, 1] 归一化空间映射到 MuJoCo 的米制坐标。
    """

    # 以下三个常量必须与 PushBlockEnv 逐位相等。
    # ScriptedPusher 直接引用 PushBlockEnv.XXX（teacher.py:100,113,145），
    # 改一个专家就会静默失效。
    BLOCK_HALF_SIZE = 0.15
    EE_RADIUS = 0.05
    MAX_EE_STEP = 0.08
    DEFAULT_SUCCESS_THRESHOLD = 0.10
    GOAL_HALF_SIZE = 0.12

    # 归一化坐标 [-1,1] 与 MuJoCo 世界坐标（米）的换算比例。
    WORLD_SCALE = 1.0

    # 每步动作对应的物理子步数（MuJoCo dt=0.002，10 子步 = 0.02s）
    _N_SUBSTEPS = 10

    def __init__(self, image_size=64, max_steps=200, success_threshold=None):
        self._image_size = int(image_size)
        self._max_steps = int(max_steps)
        self._success_threshold = (
            self.DEFAULT_SUCCESS_THRESHOLD if success_threshold is None
            else float(success_threshold))

        self._model = self._build_model()
        self._data = mujoco.MjData(self._model)

        # 离屏渲染器（与环境同生命周期，避免每帧 new 一个）
        self._renderer = mujoco.Renderer(
            self._model, height=self._image_size, width=self._image_size)

        # mocap id：ee 是第 0 个，goal_marker 是第 1 个
        self._ee_mocap_id = 0
        self._goal_mocap_id = 1

        # 语义层状态副本（对外暴露，归一化 [-1, 1]）
        self.ee_pos = np.zeros(2, dtype=np.float64)
        self.block_pos = np.zeros(2, dtype=np.float64)
        self.block_theta = 0.0
        self.goal_pos = np.zeros(2, dtype=np.float64)
        self.step_count = 0
        self.contact_flag = 0.0
        self.rng = np.random.RandomState(0)

    # ------------------------------------------------------------------
    # 环境接口
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """重置环境，随机生成一轮推块任务。

        布局生成策略与 PushBlockEnv 完全一致，保证同 seed 下两者
        的初始状态尽可能对齐（差异仅来自物理引擎的数值解）。
        """
        self.rng = np.random.RandomState(seed)

        self.block_pos = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float64)
        self.block_theta = float(self.rng.uniform(-np.pi, np.pi))

        # 目标：离方块至少 0.45
        min_goal_dist = 0.45
        for _ in range(50):
            self.goal_pos = self.rng.uniform(
                -0.85, 0.85, size=2).astype(np.float64)
            if np.linalg.norm(self.goal_pos - self.block_pos) >= min_goal_dist:
                break
        else:
            fb = self.block_pos + np.array([0.5, -0.5], dtype=np.float64)
            self.goal_pos = np.clip(fb, -0.85, 0.85)

        # 推杆：在反方向 + 角度扰动
        goal_dir = self.goal_pos - self.block_pos
        goal_dir = goal_dir / (np.linalg.norm(goal_dir) + 1e-8)
        base_angle = np.arctan2(-goal_dir[1], -goal_dir[0])
        self.ee_pos = self.block_pos.copy()
        for _ in range(20):
            aj = self.rng.uniform(-0.6, 0.6)
            dist = self.rng.uniform(0.35, 0.55)
            ang = base_angle + aj
            off = np.array(
                [np.cos(ang), np.sin(ang)], dtype=np.float64) * dist
            cand = np.clip(self.block_pos + off, -0.95, 0.95)
            if np.linalg.norm(cand - self.block_pos) > (
                    self.BLOCK_HALF_SIZE + self.EE_RADIUS + 0.02):
                self.ee_pos = cand
                break
        else:
            self.ee_pos = np.clip(
                self.block_pos - goal_dir * 0.5, -0.95, 0.95)

        self.step_count = 0
        self.contact_flag = 0.0
        self._sync_mujoco_state()
        return self.get_observation()

    def step(self, action):
        """执行一步动作。

        动作是 2 维 numpy 数组（或序列），每个分量在 [-1, 1] 之间，
        被映射为推杆在归一化空间中的位移（乘以 MAX_EE_STEP）。
        与 PushBlockEnv 行为完全一致：超范围裁剪、维度不足补齐。
        """
        action = np.asarray(action, dtype=np.float64).flatten()
        if action.shape[0] < 2:
            action = np.pad(action, (0, 2 - action.shape[0]))
        action = np.clip(action[:2], -1.0, 1.0)

        ee_delta = action * self.MAX_EE_STEP
        self.ee_pos = np.clip(self.ee_pos + ee_delta, -1.0, 1.0)

        # 写入 mocap 目标位置（位置伺服）
        s = self.WORLD_SCALE
        self._data.mocap_pos[self._ee_mocap_id][0] = float(
            self.ee_pos[0] * s)
        self._data.mocap_pos[self._ee_mocap_id][1] = float(
            self.ee_pos[1] * s)

        # 步进物理
        for _ in range(self._N_SUBSTEPS):
            mujoco.mj_step(self._model, self._data)

        # 读回方块状态
        self.block_pos = np.array([
            float(self._data.qpos[0]) / s,
            float(self._data.qpos[1]) / s,
        ], dtype=np.float64)
        self.block_theta = float(self._data.qpos[2])

        self.contact_flag = 1.0 if self._data.ncon > 0 else 0.0
        self.block_pos = np.clip(self.block_pos, -1.0, 1.0)
        self.step_count += 1

        reward = self._compute_reward()
        done = self.is_success() or self.step_count >= self._max_steps
        return self.get_observation(), float(reward), bool(done)

    def get_observation(self):
        """返回当前观测（纯读操作，不改变内部状态）。"""
        return {
            "images": {"base_0_rgb": self._render()},
            "state": self._get_state(),
            "prompt": "Push the block to the green goal.",
        }

    def is_success(self):
        """方块中心进入目标阈值即成功。"""
        d = float(np.linalg.norm(self.block_pos - self.goal_pos))
        return d < self._success_threshold

    # ------------------------------------------------------------------
    # 测试用钩子（仅供测试注入特殊状态，生产代码不要依赖）
    # ------------------------------------------------------------------

    def _set_block(self, pos, theta):
        """手动设置方块位置和朝向（测试用）。"""
        self.block_pos = np.asarray(pos, dtype=np.float64).copy()
        self.block_theta = float(theta)
        self._sync_mujoco_state()

    def _set_ee(self, pos):
        """手动设置推杆位置（测试用）。"""
        self.ee_pos = np.asarray(pos, dtype=np.float64).copy()
        s = self.WORLD_SCALE
        self._data.mocap_pos[self._ee_mocap_id][0] = float(
            self.ee_pos[0] * s)
        self._data.mocap_pos[self._ee_mocap_id][1] = float(
            self.ee_pos[1] * s)
        for _ in range(5):
            mujoco.mj_step(self._model, self._data)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_state(self):
        """8 维状态向量（与 PushBlockEnv 逐位同语义）。"""
        return np.array([
            float(self.ee_pos[0]),
            float(self.ee_pos[1]),
            float(self.block_pos[0]),
            float(self.block_pos[1]),
            float(self.block_theta / np.pi),
            float(self.goal_pos[0]),
            float(self.goal_pos[1]),
            float(self.contact_flag),
        ], dtype=np.float32)

    def _compute_reward(self):
        """稠密奖励：负距离 + 成功 bonus（与 numpy 版对齐）。"""
        d = float(np.linalg.norm(self.block_pos - self.goal_pos))
        r = -d
        if d < self._success_threshold:
            r += 1.0
        return r

    def _render(self):
        """离屏渲染顶视 RGB 图像（uint8, HxWx3）。"""
        self._renderer.update_scene(self._data, camera="top")
        return self._renderer.render().copy()

    def _build_model(self):
        """构建 MuJoCo 模型（XML 字符串）。

        场景构成：
          - floor: 地面
          - ee: 推杆 mocap 体（cylinder，位置伺服）
          - goal_marker: 目标标记 mocap 体（薄 box，纯视觉，不参与碰撞）
          - block: 方块（平面关节 x/y/yaw，高阻尼近似准静态）
          - top: 顶视相机

        关节加了高阻尼，让方块表现出"推就动、不推就停"的准静态特征。
        目标用 mocap 体：运行时改位置方便，且不参与碰撞计算。
        """
        s = self.WORLD_SCALE
        bh = self.BLOCK_HALF_SIZE * s
        er = self.EE_RADIUS * s
        bz = bh * 0.5
        gh = self.GOAL_HALF_SIZE * s

        lines = []
        ap = lines.append
        ap('<mujoco model="mujoco_push_block">')
        ap('  <option timestep="0.002" gravity="0 0 -9.81"/>')
        ap('  <worldbody>')
        ap('    <light pos="0 0 %g" dir="0 0 -1"/>' % (s * 1.5))
        ap('    <geom name="floor" type="plane" size="%g %g 0.05"'
           ' rgba=".12 .12 .12 1"/>' % (s * 1.2, s * 1.2))
        # 推杆 mocap
        ap('    <body name="ee" mocap="true" pos="0 0 %g">' % bz)
        ap('      <geom type="cylinder" size="%g %g"'
           ' rgba=".86 .24 .24 1" contype="1" conaffinity="0"/>' % (er, bz))
        ap('    </body>')
        # 目标标记 mocap（纯视觉，contype=0 conaffinity=0 不参与碰撞）
        ap('    <body name="goal_marker" mocap="true" pos="0 0 0.003">')
        ap('      <geom type="box" size="%g %g 0.001"'
           ' rgba="0 .78 0 1" contype="0" conaffinity="0"/>' % (gh, gh))
        ap('    </body>')
        # 方块（平面关节：x/y 滑动 + z 轴旋转，高阻尼）
        ap('    <body name="block" pos="0 0 %g">' % bz)
        ap('      <joint name="bx" type="slide" axis="1 0 0" damping="8"/>')
        ap('      <joint name="by" type="slide" axis="0 1 0" damping="8"/>')
        ap('      <joint name="bth" type="hinge" axis="0 0 1" damping="0.6"/>')
        ap('      <geom type="box" size="%g %g %g"'
           ' rgba=".24 .47 .86 1" contype="1" conaffinity="1"'
           ' friction="0.9 .01 .0001"/>' % (bh, bh, bz))
        ap('    </body>')
        # 顶视相机（fovy=90 在 z=1.0 处视场宽 2.0，恰好覆盖 [-1,1]）
        ap('    <camera name="top" pos="0 0 %g"'
           ' xyaxes="1 0 0 0 1 0" mode="fixed" fovy="90"/>'
           % (s * 1.0))
        ap('  </worldbody>')
        ap('</mujoco>')
        xml = '\n'.join(lines) + '\n'
        return mujoco.MjModel.from_xml_string(xml)

    def _sync_mujoco_state(self):
        """把语义层状态同步到 MuJoCo 物理状态（reset / 手动注入用）。

        步骤：
        1. 设置方块 qpos（x, y, yaw）和 qvel（清零）
        2. 设置两个 mocap 体的位置（ee 和 goal_marker）
        3. 跑几步正向动力学让系统稳定，然后再把方块复位一次
           （防止 mocap 伺服在初始化阶段把方块推偏了一点）
        """
        s = self.WORLD_SCALE
        bz = self.BLOCK_HALF_SIZE * s * 0.5

        # 方块 qpos = [x, y, yaw]（三个关节顺序）
        self._data.qpos[0] = float(self.block_pos[0] * s)
        self._data.qpos[1] = float(self.block_pos[1] * s)
        self._data.qpos[2] = float(self.block_theta)
        self._data.qvel[:] = 0.0

        # 推杆 mocap
        self._data.mocap_pos[self._ee_mocap_id][0] = float(
            self.ee_pos[0] * s)
        self._data.mocap_pos[self._ee_mocap_id][1] = float(
            self.ee_pos[1] * s)
        self._data.mocap_pos[self._ee_mocap_id][2] = float(bz)

        # 目标标记 mocap
        self._data.mocap_pos[self._goal_mocap_id][0] = float(
            self.goal_pos[0] * s)
        self._data.mocap_pos[self._goal_mocap_id][1] = float(
            self.goal_pos[1] * s)
        self._data.mocap_pos[self._goal_mocap_id][2] = 0.003

        # 正向动力学，让接触、mocap 焊接等约束生效
        mujoco.mj_forward(self._model, self._data)

        # 跑几步子步稳定系统，然后把方块状态强制复位一次
        # （mocap 焊接可能会产生微小扰动）
        for _ in range(20):
            mujoco.mj_step(self._model, self._data)
        self._data.qpos[0] = float(self.block_pos[0] * s)
        self._data.qpos[1] = float(self.block_pos[1] * s)
        self._data.qpos[2] = float(self.block_theta)
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
