"""Chunk 部分执行（receding horizon / execution horizon）—— 退视界 + deque

================================================================================
归属考证（重要）：本机制**不属于 π0.5**，故正宅在 pi07/ 下
================================================================================

一手论文证据：

1. **π0.5 论文（2504.16054）没有"执行视界"这个概念。**
   全文检索 `execution horizon` / `receding` —— **零命中**。
   π0.5 谈 action chunk 时只说"预测 a_{t:t+H}"，从不讨论"只执行前 ŝ 步
   然后重推理"这件事，也没有对应的超参。

2. **`execution horizon` 是 RTC 论文（2506.07339）正式定名的。**
       "When action chunking policies are rolled out, only the first s ≤ H
        actions from each chunk are executed. We call s the **execution
        horizon**; often it is shorter than the prediction horizon, but still
        much greater than 1 (e.g., s ≈ H/2)."                    (p2-p3)
   该文还给出关键约束 `d ≤ s ≤ H − d`（p4），把"延迟 d"与"执行视界 s"
   绑成一对必须协同选择的超参 —— 这正是本模块 rtc_max_delay 与
   n_action_steps 之间那条校验规则的理论来源。

3. **π0.7（2604.15483）明确采用并给出取值。**
       "often a shorter horizon of actions, **ˆH < H**, is executed"   (p3)
       "we use 5 denoising steps to generate the 50-step action chunks and
        execute **ˆH ∈ {15, 25}** steps out of the chunk."             (p7)
   即 H=50、ŝ∈{15,25}，约为 H/3 ~ H/2，与 RTC 论文的经验值 s≈H/2 吻合。

结论：π0.5 → 无；概念定名 → 2506.07339；π0.7 → 采用，H=50 / ŝ∈{15,25}。

--------------------------------------------------------------------------------
机制本身：为什么不把整块 50 步执行完？
--------------------------------------------------------------------------------

我们先想一下这里的核心张力。模型一次预测 H=50 步（1 秒的动作）。有两个极端：

  · **整块执行完再重推理（s=H）**：推理次数最少、最省算力，但代价是
    整整 1 秒内机器人对新观测**完全失聪**。物体被人挪走了、抓滑脱了，
    它都要等这 1 秒走完才知道。RTC 论文的说法是
    "A long execution horizon reduces a policy's responsiveness to new
     information"。

  · **每步都重推理（s=1）**：最灵敏，但推理频率 = 控制频率 = 50Hz，
    对一个几十亿参数的 VLA 完全不现实（单次推理就 76~127ms）。

退视界（receding horizon）是这条张力上的折中：**只执行前 ŝ 步，
然后带着最新观测重新推理**。这与经典模型预测控制（MPC）的滚动优化
是同一个思想 —— RTC 论文自己也点了这层关系（p9："like our method,
MPC generates plans over a **receding** time horizon"）。

`s ≈ H/2` 这个经验值的直觉：留一半余量给"下一次推理还没算完"的这段时间，
既保证动作永不断供（实时性约束 d ≤ s），又不至于让反应迟钝超过半秒。

--------------------------------------------------------------------------------
实现语义：与 RTC 天然咬合
--------------------------------------------------------------------------------

- 模型 `sample_actions` **始终输出全 H 步**（模型不关心执行策略）；
- 策略层维护一个 deque（maxlen = n_action_steps），每个控制步弹出一个动作；
- 当队列耗尽、或剩余 ≤ rtc_max_delay 时触发重推理：
    把队列**剩余动作**当作 `action_prefix`（这些是既定要执行的），
    `delay = 剩余数`，调 `sample_actions(action_prefix=..., delay=...)`，
    新 chunk 的 `[delay : delay + n_action_steps]` 入队（跳过冻结段）。

这里有个值得说清的设计动机：**为什么低水位（剩余 ≤ delay）就要重推理，
而不是等队列彻底空？** 因为推理本身要花 d 步的时间。若等空了才启动，
这 d 步就无动作可发 —— 机器人只能停顿，正是 RTC 要消灭的那种停顿。
提前 d 步启动，新 chunk 恰好在旧队列耗尽时抵达，动作流才连续。
而这提前启动的 d 步，其动作已经在队列里、必然会被执行，
所以它们正是 RTC 的"冻结前缀"。—— 两个机制在这里严丝合缝地对接上了。

`n_action_steps = action_horizon`（默认）时退化为整块执行，行为与不启用一致。
"""
from collections import deque

import torch


class ChunkExecutionPolicy:
    """退视界 chunk 执行包装器（execution horizon ŝ + 冻结前缀重推理）

    职责边界：这是一个**策略层包装器**，不是模型的一部分。
    模型只管"给定观测和可选冻结前缀，产出一整块动作"；
    "执行多少步、何时重推理、把谁当冻结前缀"这些部署决策都在这里。
    这样分层的好处是模型可以独立测试，而执行策略可以替换（整块/退视界/
    每步重推理）而无需改模型一行代码。
    """

    def __init__(self, model, n_action_steps=None, rtc_max_delay=0):
        """初始化

        Args:
            model: 具备 sample_actions(images, text_tokens, action_prefix=, delay=) 的对象
            n_action_steps: 执行视界 ŝ，每块执行的步数（None = 取 model.config.action_horizon）
            rtc_max_delay: 触发重推理的低水位阈值 d（默认 0 = 耗尽才重推理）
        """
        self.model = model
        cfg = getattr(model, "config", None)
        self.n_action_steps = (
            n_action_steps if n_action_steps is not None
            else (cfg.action_horizon if cfg is not None else 1)
        )
        self.rtc_max_delay = rtc_max_delay
        self._queue = deque()

    def __call__(self, *args, **model_kwargs):
        """返回下一个控制步要执行的动作。

        签名故意做成**全透传**（*args + **kwargs），不硬编码任何模型字段名。
        原因是 π0.5 与 π0.7 的 ``sample_actions`` 入参完全不同：

            π0.5: sample_actions(images, text_tokens, state=None, ...)
            π0.7: sample_actions(videos, lang_tokens, lang_masks, state, ...)

        执行策略层根本不需要理解这些字段的含义 —— 它只关心"动作 chunk 从哪来、
        执行几步、何时带着冻结前缀重推理"。把观测字段名写进策略层的签名，
        等于让部署逻辑绑死某一代模型的输入格式，是错误的耦合方向。
        因此这里只负责**追加** ``action_prefix`` / ``delay`` 两个 RTC 相关参数，
        其余原样转发。

        Args:
            *args: 位置参数，原样转发（π0.5 老用法 ``policy(images, text)``）
            **model_kwargs: 关键字参数，原样转发
                （π0.7 用法 ``policy(**adapter.encode(obs))``）

        Returns:
            (B, A) 下一步动作
        """
        # 队列耗尽、或剩余 ≤ rtc_max_delay（低水位）时重推理。
        # 低水位而非耗尽才触发，是为了让新 chunk 在旧队列走完前抵达（见模块 docstring）。
        if len(self._queue) == 0 or len(self._queue) <= self.rtc_max_delay:
            action_prefix = None
            delay = 0
            if len(self._queue) > 0:
                # 队列剩余动作 = 推理期间必然被执行的动作 = RTC 的冻结前缀
                prefix_actions = list(self._queue)
                delay = len(prefix_actions)
                action_prefix = torch.stack(prefix_actions, dim=1)  # (B, d, A)
                # sample_actions 期望 delay 为 (B,) long 张量
                delay = torch.full(
                    (action_prefix.shape[0],), delay,
                    dtype=torch.long, device=action_prefix.device,
                )

            chunk = self.model.sample_actions(
                *args,
                action_prefix=action_prefix, delay=delay, **model_kwargs,
            )  # (B, H, A)

            # 入队新 chunk 的 [delay : delay + n_action_steps]：
            # 跳过 [0, delay) 这段——它们是冻结前缀，本轮已经执行过了。
            d = int(delay[0].item()) if torch.is_tensor(delay) else int(delay)
            self._queue.clear()
            for i in range(d, d + self.n_action_steps):
                self._queue.append(chunk[:, i])

        return self._queue.popleft()
