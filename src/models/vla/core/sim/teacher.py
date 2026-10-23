# -*- coding: utf-8 -*-
"""ScriptedPusher: 推块任务的脚本专家（用于生成行为克隆演示数据）

设计思路见 spec D11：三阶段状态机。

- 阶段 1（绕行 / APPROACH）：推杆尚未对齐"推线"时，绕到方块后方
  （即 goal 所在方向的反方向）。
- 阶段 2（推动 / PUSH）：沿 block -> goal 方向持续推。
- 阶段 3（回绕 / RECOVER）：当推杆因方块旋转或滑移偏离推线时，
  重新回到阶段 1。

实现上，这三个阶段被统一成一个"几何伺服"控制器：
每一帧都把推杆朝"方块背面的理想接触点"驱动，并让推杆轻微嵌入方块。
这样推杆在靠近时自然完成"绕行/回绕"，到位后则持续向前施压，
自动实现状态机的三个功能，而不需要显式维护离散状态。

关键几何细节：
方块会旋转，因此"背面"不能简单地用 ``block - push_dir * offset`` 计算。
我们先把目标方向变换到方块的局部坐标系，沿"与目标方向相反"的射线
与方块矩形边界求交，得到真正的背面接触点，再转回世界坐标。
这样推杆始终顶在真正的"背面"上，产生的旋转力矩最小，推动最稳定。

伺服控制：
返回的动作不是固定全速，而是 ``delta / MAX_EE_STEP`` 并裁剪到 [-1, 1]。
环境会把动作重新乘以 ``MAX_EE_STEP``，因此推杆每帧实际位移 ≈ delta，
不会出现超调振荡，这是高成功率的关键。

为什么需要脚本专家？

约束 C02 指出仓库内没有任何预训练权重。若用随机初始化的 VLA 策略跑闭环，
动作是随机游走，根本完不成推块任务，也无法产生可用于训练的正样本。
脚本专家用解析几何生成近似最优动作，是行为克隆的最小可行数据源，
也是 before/after 对比中"before"（专家）与"after"（学习策略）的基准。
"""

import numpy as np

from .base import Agent, Runtime, Subscriber
from models.vla.core.sim.push_block_env import PushBlockEnv


class ScriptedPusher(Agent):
    """基于几何伺服的脚本推杆专家。

    核心思想：把"推杆中心"伺服到"方块背面的理想接触点"。
    该目标点根据方块的实时朝向 ``block_theta`` 动态计算，
    并采用比例控制消除超调振荡。
    """

    #: 推杆嵌入方块的深度（世界坐标）。嵌入保证接触持续，
    #: 但过深会导致推杆在方块内部穿行、旋转失控。
    OVERLAP = 0.03

    def __init__(self):
        # 保留 mode 字段仅用于日志/可视化，控制器本身不依赖离散状态
        self.mode = "approach"

    def reset(self):
        """每回合开始时重置内部状态。

        Runtime.run_episode 会在 env.reset 之后调用本方法。
        """
        self.mode = "approach"

    @staticmethod
    def _rotate_vector(v, angle):
        """把 2 维向量 v 绕原点旋转 angle 弧度。"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array(
            [c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64
        )

    def _compute_target(self, ee, block, goal, theta):
        """计算理想推点：方块背面外侧、沿目标方向嵌入一点。

        Args:
            ee: 推杆位置 (2,)
            block: 方块位置 (2,)
            goal: 目标位置 (2,)
            theta: 方块朝向（弧度）

        Returns:
            target: 推杆应该到达的 2D 点。
        """
        push_dir = goal - block
        dist = np.linalg.norm(push_dir)
        if dist < 1e-8:
            return ee.copy()
        push_dir = push_dir / dist

        # 把目标方向变换到方块局部坐标系
        local_push = self._rotate_vector(push_dir, -theta)
        local_push_norm = np.linalg.norm(local_push)
        if local_push_norm < 1e-8:
            return ee.copy()

        # "背面"方向在局部坐标系就是 -local_push 方向。
        # 从方块中心沿该方向出发，与矩形 [-half, half]^2 边界求交。
        back_dir = -local_push / local_push_norm
        half = PushBlockEnv.BLOCK_HALF_SIZE
        # 分别计算与 x=±half、y=±half 的交点参数 t，取最小正值
        t_x = half / (abs(back_dir[0]) + 1e-12)
        t_y = half / (abs(back_dir[1]) + 1e-12)
        t = min(t_x, t_y)
        local_contact = back_dir * t

        # 把局部接触点转回世界坐标
        world_contact = block + self._rotate_vector(local_contact, theta)

        # 理想推杆中心：从接触点沿 -push_dir 方向退后
        # (EE_RADIUS - OVERLAP)，使推杆边缘嵌入方块表面 OVERLAP 深度。
        target = world_contact - push_dir * (
            PushBlockEnv.EE_RADIUS - self.OVERLAP
        )
        return target

    def get_action(self, obs):
        """根据当前观测返回 2 维动作（已裁剪到 [-1, 1]）。

        动作为比例控制量：``delta / MAX_EE_STEP``。
        环境会再把动作乘以 MAX_EE_STEP，因此推杆实际位移 ≈ delta，
        避免超调。

        Args:
            obs: 观测 dict，state 为 8 维 float32 数组。

        Returns:
            ndarray (2,)，每个分量在 [-1, 1] 之间。
        """
        state = obs["state"]
        # 转成 float64 做内部计算，避免角度和距离累加时精度损失
        ee = state[0:2].astype(np.float64)
        block = state[2:4].astype(np.float64)
        goal = state[5:7].astype(np.float64)
        theta = float(state[4]) * np.pi  # state[4] 是 theta / pi

        target = self._compute_target(ee, block, goal, theta)

        delta = target - ee

        # 用于可视化：距离较远时标记为 approach，贴近后标记为 push
        self.mode = "push" if np.linalg.norm(delta) < 0.12 else "approach"

        # 比例控制：让单步实际位移 ≈ delta，避免超调振荡
        raw_action = delta / PushBlockEnv.MAX_EE_STEP
        return np.clip(raw_action, -1.0, 1.0)


class DemoRecorder(Subscriber):
    """订阅者：把 Runtime 主循环数据录制成行为克隆可用的轨迹。

    轨迹结构：
        observations: [obs_0, obs_1, ..., obs_T]   长度 T+1
        actions:      [a_0,   a_1,   ..., a_{T-1}] 长度 T
        rewards:      [r_0,   r_1,   ..., r_{T-1}] 长度 T
        dones:        [d_0,   d_1,   ..., d_{T-1}] 长度 T
    其中 obs_t 是采取 a_t 之前看到的观测，符合 VLA 训练惯例。
    """

    def __init__(self):
        self.episodes = []
        self._current = None

    def on_episode_start(self, obs):
        """回合开始时初始化空轨迹。"""
        self._current = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "dones": [],
        }

    def on_step(self, obs, action, reward, done):
        """每步记录动作、奖励、终止标志和新的观测。"""
        self._current["actions"].append(np.asarray(action).copy())
        self._current["rewards"].append(float(reward))
        self._current["dones"].append(bool(done))
        self._current["observations"].append(obs)
        if done:
            self.episodes.append(self._current)
            self._current = None

    def on_episode_end(self, info):
        """回合结束钩子。数据已在 done 时归档，这里无需额外操作。"""
        # 保留空实现，作为 Subscriber 的范例
        pass


def generate_demos(n_episodes, seed=None, env_kwargs=None):
    """生成专家演示轨迹。

    Args:
        n_episodes: 要生成的回合数。
        seed: 起始随机种子；第 i 个回合使用 seed + i，保证可复现。
        env_kwargs: 传给 PushBlockEnv 的关键字参数字典，例如
                    {"image_size": 64, "max_steps": 200}。

    Returns:
        list[dict]: 每个 dict 是一条轨迹，包含 observations/actions/rewards/dones。
    """
    env_kwargs = env_kwargs if env_kwargs is not None else {}
    env = PushBlockEnv(**env_kwargs)
    agent = ScriptedPusher()
    recorder = DemoRecorder()
    runtime = Runtime(env, agent, [recorder])

    for i in range(n_episodes):
        episode_seed = None if seed is None else seed + i
        runtime.run_episode(seed=episode_seed)

    return recorder.episodes
