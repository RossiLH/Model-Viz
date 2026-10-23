"""状态离散化模块

将连续状态向量离散化为 token 序列，用于 PI0.5 的离散状态输入。

官方做法：
- 状态向量 digitize 到 0-255 的 bins
- 转为 token 字符串拼接到 prompt 中
- 格式："Task: {text}, State: {s1} {s2} ...;\nAction: "
"""
import numpy as np
import torch
from typing import Optional


class StateTokenizer:
    """状态离散化器
    
    将连续状态向量转换为离散 token 序列。
    """
    
    def __init__(self, num_bins: int = 256, state_range: tuple = (-1.0, 1.0)):
        """初始化状态离散化器
        
        Args:
            num_bins: 离散化 bins 数量（默认 256，对应 0-255）
            state_range: 状态值范围（默认 -1 到 1）
        """
        self.num_bins = num_bins
        self.state_range = state_range
        # 预计算 bins 边界
        self.bins = np.linspace(state_range[0], state_range[1], num_bins + 1)[:-1]
    
    def tokenize(self, text: str, state: Optional[torch.Tensor]) -> str:
        """将文本和状态组合为 token 序列
        
        Args:
            text: 任务描述文本
            state: 状态向量 (action_dim,) 或 None
            
        Returns:
            格式化的 token 字符串
        """
        if state is None:
            return f"Task: {text};\nAction: "
        
        # 将状态离散化到 0-255
        state_np = state.detach().cpu().numpy()
        discretized = np.digitize(state_np, self.bins)
        # 确保在 0-255 范围内
        discretized = np.clip(discretized, 0, self.num_bins - 1)
        
        # 转为字符串
        state_str = " ".join(map(str, discretized.tolist()))
        
        return f"Task: {text}, State: {state_str};\nAction: "
