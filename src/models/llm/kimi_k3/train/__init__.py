"""训练模块。

公开 API:
    - Trainer: 端到端训练器
    - TrainState: 训练状态
    - RunningMean: 滑动窗口均值
"""
from models.llm.kimi_k3.train.loop import Trainer, TrainState
from models.llm.kimi_k3.train.metrics import RunningMean

__all__ = ["Trainer", "TrainState", "RunningMean"]
