r"""流水线并行 + 虚拟 stage + AttnRes 缓存通信（论文 §5.2）。

复用与之相关的前置知识点
-----------------------
- K3 混合架构每 block = 3 层 KDA + 1 层 Gated MLA（+ MoE FFN）。
  block 是天然的"流水线切分层"（见 AGENTS.md / KDA 文档）。
- 论文 §5.2 有两处 K3 特性在这里复现：
  1. **Cache-based AttnRes 通信**：Block AttnRes 让每层都能看到前面所有
     block 的表征。若跨 stage 边界把"全集"搬过去，通信量会是
     ``O(层数 × 表征)``。K3 只增量传输**本 stage 新产生**的 block 表征，
     用完即释放，其驻留量从不超过"在途 micro-batch 的活 block 数"，
     从而把 AttnRes 结构的内存成本压到理论下界。
  2. **激活均衡**：interleaved 1F1B 下，前段 stage 在 warmup 期积压更多驻留
     micro-batch，激活分布不均。与其所有 rank 都按最坏的容量配内存，不如把
     富余激活**远程 offload 到其他 pipeline rank 的内存**里，拉平 footprint。
- 与此同时，经典的 interleaved 1F1B 调度（NVIDIA 论文）也在这里提供：
  每个 rank 的 forward/backward 顺序 = warmup -> 交替 -> drain。

本模块是 CPU 可运行的"调度机制 + 通信记账"复现：
  - partition_layers：把层数切成 pp*virtual_pipeline_stages 段，
    不要求整除（93 层 / 16 stage 给多的一层）。
  - one_f_one_b_schedule：虚拟 interleaved 1F1B，输出每 rank 的 step 序列，
    实测 warmup 前 goroutine 的微批次更多（这就是激活均衡要解决的 skew）。
  - AttnResCache：跨 stage 只传新增 block 表征、退役即释放、记账字节数。
  - activation_balance：规划"远程 offload"步骤把每 stage 控制在 capacity 内。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

__all__ = [
    "partition_layers",
    "PipelineSchedule",
    "one_f_one_b_schedule",
    "AttnResCache",
    "activation_balance",
]


def partition_layers(n_layers: int, pp: int, virtual_pipeline_stages: int = 1) -> list[list[int]]:
    """把层切成 ``pp * virtual_pipeline_stages`` 段（段长尽量相等）。

    Args:
        n_layers: 总层数。
        pp: 流水线 stage 数（rank 数）。
        virtual_pipeline_stages: 虚拟 stage（每个 rank 持有几段）。

    Returns:
        ``[[layer ids], ...]``，每段是一个连续的层 id 列表。
    """
    n_chunks = pp * virtual_pipeline_stages
    if n_chunks > n_layers:
        raise ValueError(f"不能把 {n_layers} 层切成 {n_chunks} 段")
    base, extra = divmod(n_layers, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < extra else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


@dataclass
class PipelineSchedule:
    """单个 rank 上 forward / backward 的微批步骤顺序。"""

    stage: int
    n_stages: int
    n_micro_batches: int
    virtual_pipeline_stages: int = 1
    steps: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def warmup_steps(self) -> int:
        """稳态开始前的纯 forward micro-batch 数。"""
        return min(
            self.n_micro_batches,
            (self.n_stages - self.stage - 1) * self.virtual_pipeline_stages
            + self.virtual_pipeline_stages - 1,
        )

    def fwd_count(self) -> int:
        return sum(1 for kind, _, _ in self.steps if kind == "forward")

    def bwd_count(self) -> int:
        return sum(1 for kind, _, _ in self.steps if kind == "backward")


def one_f_one_b_schedule(
    stage: int, n_stages: int, n_micro_batches: int, virtual_pipeline_stages: int = 1
) -> PipelineSchedule:
    """经典 interleaved 1F1B 调度（论文 §5.2 的 baseline）。

    回顾：GPU 上反向依赖前向结果，所以 pipeline 各 stage 必须"前向先进、反向
    按序"；1F1B = 一旦有可执行的反向就优先做（backward 只有在本 stage 本层
    完成才可做）。warmup 期只做 forward，达到稳态后"一 forward 一 backward"
    交替，最后 drain 全部 backward。

    Args:
        stage: 本 rank 的 stage id（0..n_stages-1）。
        n_stages: 总 stage 数。
        n_micro_batches: 每 step 的 micro-batch 数。
        virtual_pipeline_stages: 虚拟 pipelining 倍数。

    Returns:
        PipelineSchedule，steps 为 ``(kind, micro_idx, layer_idx)`` 序列。
    """
    sched = PipelineSchedule(stage, n_stages, n_micro_batches, virtual_pipeline_stages)
    # warmup：early stage 需要等后面 stage 的表征就绪，先发自己的 forward。
    warmup = min(n_micro_batches, (n_stages - stage - 1) * virtual_pipeline_stages
                 + virtual_pipeline_stages - 1)
    steps: list[tuple[str, int, int]] = []
    for m in range(warmup):
        steps.append(("forward", m, 0))
    fwd, bwd = warmup, 0
    while bwd < n_micro_batches:
        if fwd < n_micro_batches:
            steps.append(("forward", fwd, 0))
            fwd += 1
        steps.append(("backward", bwd, 0))
        bwd += 1
    sched.steps = steps
    return sched


@dataclass
class AttnResCache:
    """跨 pipeline stage 增量传递 block 表征的缓存。

    每个 stage 收到上游完成的 block，追加自己生成的 block，
    只把**新增**的部分转给下游。micro-batch 结束即释放，
    驻留内存 <= 在途 micro-batch 的 alive block 数。

    属性：
      blocks: ``micro_batch -> list[Tensor]``（按生成顺序）。
      transferred / bytes_transferred: 记账，供实验对比 naive 重发。
    """

    blocks: dict[int, list[Tensor]] = field(default_factory=dict)
    transferred: int = 0
    bytes_transferred: int = 0

    def receive(self, micro_batch: int, incoming: list[Tensor]) -> None:
        self.blocks[micro_batch] = list(incoming)

    def append(self, micro_batch: int, block: Tensor) -> None:
        self.blocks.setdefault(micro_batch, []).append(block)

    def outgoing(self, micro_batch: int, n_already_sent: int) -> list[Tensor]:
        """只返回本 stage 新增的 block（n_already_sent 之后的部分）。"""
        new = self.blocks.get(micro_batch, [])[n_already_sent:]
        self.transferred += len(new)
        self.bytes_transferred += sum(t.numel() * t.element_size() for t in new)
        return new

    def retire(self, micro_batch: int) -> None:
        """micro-batch 完成，释放其所有 block。"""
        self.blocks.pop(micro_batch, None)

    def resident_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for blocks in self.blocks.values()
            for t in blocks
        )

    def naive_bytes(self, n_stages: int) -> int:
        """如果不走增量而每次边界都重发全集——需要几倍字节。"""
        return self.resident_bytes() * n_stages


def activation_balance(
    per_stage_micro_batches: list[int], capacity: int
) -> list[tuple[int, int, int]]:
    """规划远程激活 offload（把富集 stage 的富余激活搬到贫 stage）拉平内存。

    Args:
        per_stage_micro_batches: 每 stage 当前驻留 micro-batch 数。
        capacity: 每 stage 允许的容量。

    Returns:
        ``[(from_stage, to_stage, count), ...]`` 移动计划，执行后
        所有 stage 的载荷都 <= capacity（若可能）。
    """
    loads = list(per_stage_micro_batches)
    moves: list[tuple[int, int, int]] = []
    for src in range(len(loads)):
        while loads[src] > capacity:
            dst = min(range(len(loads)), key=lambda i: loads[i])
            if loads[dst] >= capacity:
                break
            move = min(loads[src] - capacity, capacity - loads[dst])
            loads[src] -= move
            loads[dst] += move
            moves.append((src, dst, move))
    return moves