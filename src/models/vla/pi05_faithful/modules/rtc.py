"""RTC 兼容层 —— 实现已迁往 pi07（归属修正）

================================================================================
本文件不再包含实现。RTC 的正宅是：
    src/models/vla/pi07/modules/rtc.py
================================================================================

为什么搬走：经论文取证，RTC **不是 π0.5 的机制**。

| 机制                | 出处                                   | 关键证据 |
| ------------------- | -------------------------------------- | -------- |
| π0.5                | **无此机制**                           | 2504.16054 全文检索 `real-time chunking`/`delay`/`frozen`/`inpaint` 零命中 |
| RTC（推理期 inpainting） | 2506.07339（Black/Galliker/Levine） | 该文把 π0.5 当作**被加速的 base policy**（p2），附录还列 "Vanilla π0.5 76ms" 作对照（p24） |
| RTC（**训练期**，即本实现） | 2512.05964，首发于 **π0.6**      | "simulating inference delay at **training time** and conditioning on action prefixes directly"（p1） |
| π0.7 采纳           | 2604.15483 p7                          | "π0.7 **also employs the training-time version of RTC** [107,108]... simulate delays of **0 to 12** timesteps... **240ms on a 50Hz** robot" |

详细考证与机制原理见 `pi07/modules/rtc.py` 的模块 docstring，
以及文档 `docs/PI07/RTC与Chunk执行-梳理-V1.md`。

--------------------------------------------------------------------------------
为什么保留这个 shim 而不直接删掉
--------------------------------------------------------------------------------

`pi05_faithful/model.py` 的 `compute_loss` / `sample_actions` 里带有
`rtc_max_delay` 开关（默认 **0 = 关闭**）。这条代码路径的价值在于：
它让本项目可以在同一个 π0.5 模型上做「关 RTC vs 开 RTC」的 before/after
对比实验 —— 这正是本项目的核心卖点。但必须讲清楚：
**开关打开时，跑的是 π0.7/π0.6 的机制，不是 π0.5 的能力。**

因此保留 re-export，既不破坏既有 import 与测试，又在文档层面把归属钉死。
新代码请直接 `from models.vla.pi07.modules.rtc import ...`。
"""
# 从 pi07 正宅重导出，保持 pi05_faithful 既有 import 可用
from models.vla.pi07.modules.rtc import (  # noqa: F401
    make_prefix_mask,
    masked_mse,
    sample_delay,
)

__all__ = ["sample_delay", "make_prefix_mask", "masked_mse"]
