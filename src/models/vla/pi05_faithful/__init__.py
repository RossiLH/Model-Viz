"""PI0.5 忠实还原实现

基于官方 openpi-main 代码，忠实还原 PI0.5 架构：
- 双专家联合注意力（DualExpertJointAttention）
- adaRMSNorm 条件化（时间步注入）
- 门控残差连接（gated residual）
- 状态离散化（StateTokenizer）
- 仿真适配器（Pi05SimAdapter）
"""
from .config import Pi05FaithfulConfig
from .sim_adapter import Pi05SimAdapter

__all__ = ["Pi05FaithfulConfig", "Pi05SimAdapter"]
