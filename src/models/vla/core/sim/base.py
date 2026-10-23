# -*- coding: utf-8 -*-
"""仿真层通用抽象接口（Environment / Agent / Subscriber / Runtime）

================================================================================
设计来源与取舍
================================================================================

本模块参考 openpi 官方的四件套抽象
（`packages/openpi-client/src/openpi_client/runtime/` 下的
 `environment.py` / `agent.py` / `subscriber.py` / `runtime.py`），
但**去掉了 WebSocket 客户端-服务器架构**，改为本地进程内直接调用。

为什么这样取舍？官方那套是为「策略服务器 + 多机器人客户端」设计的：
策略跑在 GPU 服务器上，真实机器人或仿真作为客户端通过 msgpack 序列化
连过来。而本项目是单进程的教学演示，跨进程通信只会带来额外复杂度，
却换不来任何教学价值。

但我们**完整保留了同样的类边界**——这一点很关键。将来若要接真实机器人，
只需另写一个 Agent 实现（内部走网络请求），Runtime 与 Environment
一行都不用改。抽象的价值正在于此。

================================================================================
观测格式约定（重要）
================================================================================

所有 Environment 的 `reset()` / `get_observation()` / `step()` 返回的观测
统一为如下 dict 结构：

    {
      "images": {camera_name: ndarray (H, W, 3) uint8},   # 多路相机
      "state":  ndarray (D,) float32,                     # 状态向量
      "prompt": str,                                      # 语言指令
    }

其中相机名约定使用 ``base_0_rgb``（第三人称主视角）。这个命名不是随意取的，
而是刻意对齐 openpi 的硬约定：

    IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

哪怕我们的合成环境只有一路相机，用同名也让将来复用官方的
`aloha_policy.py` / `libero_policy.py` 变换层时是零成本的。

动作则统一为 numpy 数组，具体维度由环境定义（本项目的 2D 推块环境是 2 维）。
"""
from abc import ABC, abstractmethod
# 注意：Python 3.7 不支持在运行时求值的注解里使用内建泛型（如 dict[str, int]），
# 因此这里统一从 typing 导入 Optional / Dict / Tuple / List。
from typing import Any, Dict, List, Optional, Tuple


class Environment(ABC):
    """环境抽象基类

    所有仿真环境与真实机器人都实现这个接口。接口刻意做得很小——
    只有三个方法——这样将来换物理引擎（PyBullet / MuJoCo）时，
    需要写的代码量最少，而上层的 Runtime / Agent / 训练循环全部可以复用。

    观测格式见模块 docstring。
    """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """重置环境到初始状态，返回初始观测

        Args:
            seed: 随机种子。给定同一个 seed 必须产生逐位相同的初始状态，
                  这是可复现性的基础（测试与 before/after 对比都依赖它）。

        Returns:
            初始观测 dict（格式见模块 docstring）
        """
        ...

    @abstractmethod
    def step(self, action) -> Tuple[Dict[str, Any], float, bool]:
        """执行一步动作，推进环境

        Args:
            action: numpy 数组，维度由具体环境定义

        Returns:
            (obs, reward, done) 三元组：
              - obs:    执行动作后的新观测
              - reward: 本步奖励（稠密奖励便于画收敛曲线）
              - done:   回合是否结束（成功或达到步数上限）
        """
        ...

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """返回当前观测

        约定：本方法**不得改变**环境内部状态（纯读操作）。
        这样可视化脚本或调试代码可以随时取观测而不干扰仿真进程。
        """
        ...


class Agent(ABC):
    """智能体抽象基类

    接收观测、返回动作。三类典型实现：
      1. 策略模型包装器（本项目的 VLAPolicyAgent，内部跑 PI0.5 / PI0.7）
      2. 脚本专家（本项目的 ScriptedPusher，用解析解产生演示数据）
      3. 远程客户端（将来接真实机器人时走网络）
    """

    @abstractmethod
    def get_action(self, obs: Dict[str, Any]):
        """根据观测返回动作

        Args:
            obs: 观测 dict（格式见模块 docstring）

        Returns:
            动作，通常是 numpy 数组
        """
        ...

    def reset(self):
        """重置智能体内部状态

        这个方法**不是**抽象方法，默认空实现。理由：无状态的智能体
        （例如纯函数式的脚本专家）没有东西需要重置，强制它们写一个
        空的 reset 只是噪音。而有状态的智能体（如维护 action chunk
        队列的 VLAPolicyAgent、或维护历史帧缓冲的 PI0.7 适配器）
        必须覆写本方法，在回合开始时清空缓冲。
        """
        pass


class Subscriber(ABC):
    """订阅者抽象基类（横切关注点的挂载点）

    用于日志、可视化、数据录制等**与仿真逻辑无关**的旁路需求。
    把这些从 Runtime 主循环里剥离出去，Runtime 才能保持简洁。

    三个钩子**全部都有默认空实现**，子类按需覆写即可。这也意味着
    本类虽然继承 ABC，但没有声明任何抽象方法，因此 ``Subscriber()``
    是可以直接实例化的（一个什么都不做的空订阅者）。
    """

    def on_episode_start(self, obs: Dict[str, Any]) -> None:
        """回合开始时调用，参数是初始观测"""
        pass

    def on_step(self, obs: Dict[str, Any], action, reward: float, done: bool) -> None:
        """每执行一步后调用

        Args:
            obs:    执行动作**之后**的新观测
            action: 本步执行的动作
            reward: 本步奖励
            done:   本步之后回合是否结束
        """
        pass

    def on_episode_end(self, info: Dict[str, Any]) -> None:
        """回合结束时调用，参数是汇总信息（steps / total_reward / done）"""
        pass


class Runtime:
    """仿真主循环驱动器

    职责单一：把 Environment 与 Agent 串起来跑一个回合，并在恰当的时机
    通知所有 Subscriber。它对下游是什么模型、上游是什么物理引擎**一无所知**，
    这正是它能被 PI0.5 与 PI0.7 共用的原因。

    注意这是一个**具体类**而非抽象类——主循环的逻辑对所有场景都一样，
    没有需要子类定制的地方。
    """

    def __init__(self, env: Environment, agent: Agent,
                 subscribers: Optional[List[Subscriber]] = None):
        """初始化

        Args:
            env:         环境实例
            agent:       智能体实例
            subscribers: 订阅者列表。默认 None 表示不挂任何订阅者。
                         这里刻意用 None 而非 ``[]`` 作默认值——可变对象
                         作默认参数是 Python 的经典陷阱（所有实例会共享
                         同一个列表）。
        """
        self.env = env
        self.agent = agent
        # None -> 空列表，保证后续 for 循环无需判空
        self.subscribers = subscribers if subscribers is not None else []

    def run_episode(self, max_steps: int = 200,
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """运行一个完整回合

        执行流程：
            reset 环境 -> reset 智能体 -> 通知回合开始
              -> 循环 { 取动作 -> 步进环境 -> 累加奖励 -> 通知本步 -> 若 done 则跳出 }
              -> 汇总 info -> 通知回合结束 -> 返回 info

        Args:
            max_steps: 步数上限。环境迟迟不报 done 时的兜底，防止死循环。
            seed:      传给 ``env.reset()`` 的随机种子

        Returns:
            info dict，含三个键：
              - steps:        实际执行的步数
              - total_reward: 累计奖励
              - done:         回合是否**因环境报告结束**而终止
                              （False 表示是被 max_steps 截断的）
        """
        # ---- 阶段 1：重置 ----
        obs = self.env.reset(seed=seed)
        self.agent.reset()
        for sub in self.subscribers:
            sub.on_episode_start(obs)

        # ---- 阶段 2：主循环 ----
        steps = 0
        total_reward = 0.0
        done = False

        for _ in range(max_steps):
            # 智能体决策
            action = self.agent.get_action(obs)
            # 环境步进（注意 obs 被就地替换为新观测）
            obs, reward, done = self.env.step(action)
            total_reward += float(reward)
            steps += 1
            # 通知所有订阅者（传的是执行后的新观测）
            for sub in self.subscribers:
                sub.on_step(obs, action, reward, done)
            if done:
                break

        # ---- 阶段 3：汇总与收尾 ----
        info = {
            "steps": steps,
            "total_reward": total_reward,
            # 这里的 bool() 转换**不是**多余的：环境的 step() 完全可能返回
            # numpy.bool_（numpy 比较运算的自然产物），而 numpy.bool_ 无法
            # 通过 `is True` 检查（它不是 Python 单例 True）。调用方与测试
            # 都按原生 bool 来断言，所以必须在这里归一化。请勿"简化"掉。
            "done": bool(done),
        }
        for sub in self.subscribers:
            sub.on_episode_end(info)
        return info
