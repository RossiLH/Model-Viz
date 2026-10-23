"""Model-Viz 并行（parallel）知识点的实现包。

对应 KIMK3 论文 §5.2 系统协同设计下的训练并行栈：
  - ``groups``         进程网格构建（PP x EP x CP x DP mesh，CPU 可运行）
  - ``kcp``            KDA 上下文并行（前缀扫描，通信与序列长度无关）
  - ``moonep``         专家并行 MoonEP（动态冗余专家，完美负载均衡）
  - ``zero``           ZeRO-1 + Pipeline ZeRO-2（优化器/梯度分片，梯度驻 CPU）
  - ``pipeline``       流水线并行（虚拟 stage、1F1B、AttnRes 增量通信、激活均衡）
  - ``activation_mgr`` 统一激活管理器（KEEP/FP8/OFFLOAD/REMOTE/RECOMPUTE）
"""

from models.llm.kimi_k3.parallel.activation_mgr import (
    ActivationManager,
    BlockFP8,
    Policy,
    SavedTensor,
    dequantize_blockwise,
    quantize_blockwise,
)
from models.llm.kimi_k3.parallel.groups import ParallelGroups, mesh_ranks
from models.llm.kimi_k3.parallel.kcp import KDAContextParallel, kcp_forward, prefix_scan_states
from models.llm.kimi_k3.parallel.moonep import MoonEP, RedundancyPlan, plan_redundant_experts
from models.llm.kimi_k3.parallel.pipeline import (
    AttnResCache,
    PipelineSchedule,
    activation_balance,
    one_f_one_b_schedule,
    partition_layers,
)
from models.llm.kimi_k3.parallel.zero import ShardedGradients, ZeroConfig, shard_ranges

__all__ = [
    "ActivationManager",
    "AttnResCache",
    "BlockFP8",
    "KDAContextParallel",
    "MoonEP",
    "ParallelGroups",
    "PipelineSchedule",
    "Policy",
    "RedundancyPlan",
    "SavedTensor",
    "ShardedGradients",
    "ZeroConfig",
    "activation_balance",
    "dequantize_blockwise",
    "kcp_forward",
    "mesh_ranks",
    "one_f_one_b_schedule",
    "partition_layers",
    "plan_redundant_experts",
    "prefix_scan_states",
    "quantize_blockwise",
    "shard_ranges",
]