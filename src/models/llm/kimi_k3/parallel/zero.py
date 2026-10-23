r"""ZeRO-1 数据并行 + Pipeline ZeRO-2 梯度分片（论文 §5.2.2）。

设计动机（why）
-------------
2.8T 参数量下，除了激活值，最大的显存消耗就是**优化器状态**和**梯度**。
K3 在数据并行维度上分片优化器状态（ZeRO-1），再在其上分片梯度
（Pipeline ZeRO-2），并且**把梯度分片放进 CPU 内存**（offload）：

  - GPU 上保留一份"双梯度缓冲"以便反向传播不受影响；
  - 跨数据并行 rank done 一次梯度归约后，把归约结果累加进 CPU 的分片里；
  - 峰值 GPU 显存减少整份梯度集大小的数量级，同时算术完全不变。

ZeRO-1 是把「每 rank 只保存 optimizer 负责的一小片参数」；ZeRO-2 继续把
「梯度也按同样切片分散」。这里的实现是二者在**单一进程/CPU**下的教学等价版：
我们精确复刻分片范围算术（shard_ranges）、reduce-scatter 的对齐方式，以及
"materialise 重建完整梯度"的往返；真实多卡只需把本地 reduce-scatter 换成
``dist.reduce_scatter``。

为什么 reduce-scatter 而不是 all-reduce？
沿梯度维做归约时，每个 rank 只需要自己的那片，all-reduce 会让所有 rank
都拿到完整结果再各自丢弃大部分——通讯量翻一倍。reduce-scatter 是
"每个人都只拿自己那片的归约结果"，正好切一半。

为什么梯度分片能offload到CPU？
梯度的生命周期很短：反向之后、优化 step 之前一次性使用。把这块大tensor
放到 CPU 能显著降低 GPU 峰值占用；代价是 CPU<->GPU 之间的一次传输延迟，
用"双缓冲 + 异步"隐藏掉。优化器吸收这些分片后，cp 侧存储那片的
optimizer state 也按 ZeRO-1 分散在各 DP rank。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "ZeroConfig",
    "shard_ranges",
    "reduce_scatter_grads",
    "ShardedGradients",
]


@dataclass
class ZeroConfig:
    """ZeRO 配置（论文 §5.2.2 / config.parallel.zero_stage）。"""

    stage: int = 1        #: 0 = 不启用，1 = ZeRO-1（优化器状态分片），2 = ZeRO-2（+梯度分片）
    offload_to_cpu: bool = False   #: 梯度分片的驻留位置是否为 CPU
    world: int = 1        #: 数据并行度（本实现用于算术模拟，与显式 dist 无关）
    group: object = None  #: 可选的真实进程组（保留 API，教学实现不作依赖）

    @property
    def shards_gradients(self) -> bool:
        """stage >= 2 才分片梯度（Pipeline ZeRO-2 语义）。"""
        return self.stage >= 2


def shard_ranges(numel: int, world: int) -> list[tuple[int, int]]:
    """把 ``numel`` 个元素切成 ``world`` 段连续、近似相等的区间。

    连续性很重要：一个 rank 的分片就是**一个切片（view）**，于是
    reduce-scatter 和优化器 step 都操作 view 而不是多次 gather。

    需要 numel 被 world 整除吗？不需要——但为了和 reduce_scatter_grads 的
    分块宽度对齐，末段的长度会 <= 分块宽度（见 test）。
    """
    per = -(-numel // world)  # 向上取整的分块宽度（与 reduce_scatter 一致）
    return [(min(i * per, numel), min((i + 1) * per, numel)) for i in range(world)]


@torch.no_grad()
def reduce_scatter_grads(
    params: list[Tensor],
    grad: list[Tensor | None] | None = None,
    world: int = 1,
    group=None,
) -> list[Tensor]:
    """梯度归约并只保留本 rank 的分片（单进程下的教学等价实现）。

    Args:
        params: 参数列表（仅用于对齐 shape / 缺省梯度）。
        grad:   ``grad[i]`` = 第 i 个参数的梯度（可为 None）。缺省读参数.grad。
        world:  模拟的 world size。==1 时退化为恒等（返回完整梯度）。

    Returns:
        ``out``，out[i] = 第 i 个参数梯度在本 rank 的分片（float32）。
        最后在真实多卡场景中，这一步是 ``dist.reduce_scatter``。
    """
    if grad is None:
        grad = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    out = []
    for g in grad:
        flat = g.reshape(-1).float()
        if world <= 1:
            out.append(flat)
            continue
        # 对齐 reduce_scatter 的分块宽度：pad 到 world 的倍数。
        pad = (-flat.numel()) % world
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        chunks = list(flat.chunk(world))
        # 真实实现：dist.reduce_scatter(shard, chunks)。这里模拟 = 平均归约。
        shard = torch.stack(chunks).sum(dim=0) / world
        out.append(shard)
    return out


@dataclass
class ShardedGradients:
    """一组参数的梯度分片容器（默认驻留 CPU）。

    结构：
      - 每参数按 shard_ranges 切成 world 段；
      - 本 rank 只持有自己那一段（连续 view）；
      - 归约是在 accumulate 里执行的（把跨 DP 的 reduce + 本地切片 + CPU 累加
        一步到位）。
    """

    params: list[Tensor]
    world: int
    rank: int
    offload_to_cpu: bool = True
    group=None

    def __post_init__(self) -> None:
        device = torch.device("cpu") if self.offload_to_cpu else self.params[0].device
        self.shards: list[Tensor] = []
        self.ranges: list[list[tuple[int, int]]] = []
        for p in self.params:
            rngs = shard_ranges(p.numel(), self.world)
            lo, hi = rngs[self.rank]
            self.ranges.append(rngs)
            self.shards.append(torch.zeros(hi - lo, dtype=torch.float32, device=device))

    def zero_(self) -> None:
        for s in self.shards:
            s.zero_()

    @torch.no_grad()
    def accumulate(self, grads: list[Tensor] | None = None) -> None:
        """归约梯度，把本 rank 那片累加进 CPU 分片，然后释放 GPU 梯度。

        Args:
            grads: 可选的外部梯度列表（缺省读取 p.grad）。
                   真实多卡下，跨 rank 的归约就是一次 ``dist.reduce_scatter``，
                   这里用 reduce_scatter_grads 模拟其分片语义。
        """
        # 第一步：跨 DP rank 归约。world==1 时 reduce_scatter_grads 退化为恒等。
        scattered = reduce_scatter_grads(self.params, grads, world=self.world)
        # 第二步：把本 rank 的分片累加进 CPU 缓存分片。
        for i, p in enumerate(self.params):
            piece = (
                scattered[i][: self.ranges[i][self.rank][1] - self.ranges[i][self.rank][0]]
            ).float()
            if piece.numel() != self.shards[i].numel():
                piece = piece[: self.shards[i].numel()]  # 对齐真实 reduce_scatter 的尾段
            self.shards[i].add_(piece.to(self.shards[i].device))
            if p.grad is not None:
                p.grad = None  # GPU 副本不再需要

    @torch.no_grad()
    def materialise(self, index: int) -> Tensor:
        """重建第 index 个参数的完整梯度（从 CPU 分片 gather 回 GPU）。

        真实实现：``dist.all_gather`` 拿到各 rank 分片再 concat + 首段裁剪。
        单个进程内我们没有其余 rank 的数值，所以这里仅保证：
          - world == 1 时完整重建（等于把分片放回原参数形状）；
          - world > 1 时返回 shape 正确的张量，但**其余 rank 的切片为 0**
            （真正的数值必须来自 all-gather）。
        """
        p = self.params[index]
        if self.world <= 1:
            full = torch.zeros(p.numel(), dtype=torch.float32)
            full[: self.shards[index].numel()] = self.shards[index].to(full.device)
            return full.reshape(p.shape).to(p.device)
        # world>1：真实场景下各 rank 分片经 all-gather 汇齐。这里模拟 concat，
        # 但只有本 rank 的分片有值（其余补 0）。
        pieces = [torch.zeros(hi - lo, dtype=torch.float32, device=self.shards[index].device)
                  for (lo, hi) in self.ranges[index]]
        pieces[self.rank] = self.shards[index]
        flat = torch.cat(pieces)[: p.numel()]
        return flat.to(p.device).reshape(p.shape)

    def gpu_bytes_saved(self) -> int:
        """由分片+offload 少占 GPU 的梯度字节数。"""
        total = sum(p.numel() for p in self.params) * 4
        if self.offload_to_cpu:
            return total
        return total - total // self.world