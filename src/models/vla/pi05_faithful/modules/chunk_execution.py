"""Chunk 部分执行 兼容层 —— 实现已迁往 pi07（归属修正）

================================================================================
本文件不再包含实现。退视界 chunk 执行的正宅是：
    src/models/vla/pi07/modules/chunk_execution.py
================================================================================

为什么搬走：经论文取证，「执行视界（execution horizon）+ 退视界重推理」
**不是 π0.5 的机制**。

| 机制                    | 出处            | 关键证据 |
| ----------------------- | --------------- | -------- |
| π0.5                    | **无此概念**    | 2504.16054 全文检索 `execution horizon` / `receding` 零命中 |
| `execution horizon` 定名 | 2506.07339 p2-3 | "only the first s ≤ H actions from each chunk are executed. We call s the **execution horizon**"；并给出约束 `d ≤ s ≤ H − d`（p4） |
| π0.7 采纳并给出取值      | 2604.15483      | p3 "often a shorter horizon **ˆH < H** is executed"；p7 "execute **ˆH ∈ {15, 25}** steps out of the chunk"（H=50） |

详细考证与机制原理见 `pi07/modules/chunk_execution.py` 的模块 docstring，
以及文档 `docs/PI07/RTC与Chunk执行-梳理-V1.md`。

保留 re-export 的理由同 `rtc.py`：`n_action_steps` 默认 = `action_horizon`
（即退化为整块执行），开关打开后可做 before/after 对比，但需明确
**这是 π0.7 的部署策略，不是 π0.5 的能力**。
新代码请直接 `from models.vla.pi07.modules.chunk_execution import ...`。
"""
# 从 pi07 正宅重导出，保持既有 import 与 demo 可用
from models.vla.pi07.modules.chunk_execution import ChunkExecutionPolicy  # noqa: F401

__all__ = ["ChunkExecutionPolicy"]
