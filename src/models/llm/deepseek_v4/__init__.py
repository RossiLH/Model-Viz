from __future__ import annotations

"""deepseek_v4 包：DeepSeek-V4 架构实现（基于 D4 三方比对决策）。"""

from models.llm.deepseek_v4.config import (
    DeepSeekV4Config,
    DeepSeekV4ModelConfig,
)
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention

__all__ = [
    "DeepSeekV4Config",
    "DeepSeekV4ModelConfig",
    "MQAV4Attention",
]
