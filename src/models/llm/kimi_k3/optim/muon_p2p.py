"""分布式 P2P Muon 正交化（论文 §5.2.2）。

设计动机（why）
-------------
在分布式训练中，每个数据并行（DP）rank 只持有完整参数的一个分片。
但 Newton-Schulz 正交化需要**完整的参数矩阵**——朴素做法是 all-gather
整个模型到每个 rank，这会在每个 rank 上额外复制一整份模型参数，
且通信量随模型规模线性增长，成为主要瓶颈。

P2P Muon 的核心思想：**每个 rank 只组装和正交化自己"拥有"的参数**。
通过 round-robin 将参数所有权分配给各 rank，然后通过点对点（P2P）通信
（irecv / isend）让拥有者从其他 rank 收集对应分片，组装完整矩阵后正交化，
再把正交化后的分片传回各 rank。没有任何 rank 需要同时持有它不拥有的参数。

内存节省：
  - All-Gather 方案: 每个 rank 需要 O(N) 额外内存（N = 总参数数）
  - P2P 方案: 每个 rank 仅需 O(N/world_size) 额外内存（约 1/world）

与 kimi3-master muon_p2p.py 的关系：
  本文件忠实复刻 kimi3-master/kimi3/optim/muon_p2p.py 的核心逻辑，
  同时做以下适配：
  1. shard_ranges 从 src/parallel/zero.py 导入（与 ZeRO-1/2 共享）
  2. newton_schulz 从 src/optim/newton_schulz.py 导入（与本地 NS 共享）
  3. 当 torch.distributed 不可用时，自动退化为单进程模式（world=1）
  4. 添加详尽中文注释，覆盖每个设计决策

单进程测试模式：
  当 dist.is_available() 为 False 或 dist.is_initialized() 为 False 时，
  所有 P2P 通信退化为本地 clone/copy，等价于 all-gather 语义。
  这使得单元测试无需启动多进程即可验证算法正确性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.distributed as dist
from torch import Tensor

from models.llm.kimi_k3.optim.newton_schulz import newton_schulz

__all__ = [
    "OwnershipMap",
    "P2PMuonOrthogonalizer",
    "orthogonalize_owned",
    "shard_ranges",
]


# ============================================================
# 1. 分片范围计算（本地实现，与 kimi3.parallel.zero.shard_ranges 等价）
# ============================================================

def shard_ranges(numel: int, world: int) -> List[Tuple[int, int]]:
    """将 numel 个元素切成 world 段连续、近似相等的区间。

    为什么需要这个函数？
    在分布式训练中，一个完整张量需要均匀分配给 world 个 rank。
    每个 rank 持有张量的一个连续切片（shard），切片的起止位置
    由 shard_ranges 计算。

    连续性为什么重要？
    - 一个 rank 的分片是**一个连续 view**，reduce-scatter 和优化器 step
      都操作 view 而不是多次 gather
    - 连续切片使 GPU 内存访问模式友好（coalesced memory access）

    算法：向上取整分块
      per = ceil(numel / world)  # 每块的最大宽度
      rank i 持有 [i*per, min((i+1)*per, numel))

    为什么用向上取整？
    末段可能比前几段短，但所有段的总和恰好等于 numel。
    这与 reduce_scatter_grads 的分块宽度对齐。

    Args:
        numel: 张量的总元素数
        world: rank 数量（数据并行度）

    Returns:
        列表，第 i 个元素为 (lo, hi) 表示 rank i 持有的区间 [lo, hi)
    """
    per = -(-numel // world)  # 向上取整的分块宽度
    return [(min(i * per, numel), min((i + 1) * per, numel)) for i in range(world)]


# ============================================================
# 2. 所有权映射：决定哪个 rank 拥有（并正交化）哪个参数
# ============================================================

@dataclass
class OwnershipMap:
    """每个参数的所有权归属表。

    在 P2P Muon 中，每个参数必须有一个"拥有者"rank 负责正交化。
    OwnershipMap 定义了这种归属关系。

    核心设计：
    - owners[i] = 拥有第 i 个参数的 rank 编号
    - world = 总 rank 数
    - rank = 本 rank 的编号（用于判断 "mine()"）

    分配策略（round-robin）：
    参数 i 归 rank (i % world)。这种轮询分配保证：
    1. 每个 rank 大致持有 N/world 个参数（负载均衡）
    2. 分配是确定性的（所有 rank 用相同规则，无需通信）
    3. 分配是静态的（参数顺序不变则分配不变）
    """

    owners: List[int]   # owners[i] = 第 i 个参数的拥有者 rank
    world: int           # 总 rank 数
    rank: int            # 本 rank 的编号

    @classmethod
    def round_robin(cls, n_params: int, world: int, rank: int) -> "OwnershipMap":
        """Round-robin 分配：参数 i 归 rank (i % world)。

        为什么用 round-robin 而不是按大小分配？
        - 确定性：所有 rank 用相同规则，无需通信
        - 均匀性：假设参数数量 >> world，每个 rank 大致持有 N/world 个
        - 简单性：实现只需一行代码

        Args:
            n_params: 参数总数
            world: rank 数量
            rank: 本 rank 编号

        Returns:
            OwnershipMap 实例
        """
        return cls(
            owners=[i % world for i in range(n_params)],
            world=world,
            rank=rank,
        )

    def mine(self) -> List[int]:
        """返回本 rank 拥有的参数索引列表。

        这些参数的正交化由本 rank 负责——需要从其他 rank 收集分片、
        组装完整矩阵、正交化、再传回分片。

        Returns:
            本 rank 拥有的参数索引列表
        """
        return [i for i, owner in enumerate(self.owners) if owner == self.rank]


# ============================================================
# 3. 分布式 P2P Muon 正交化器
# ============================================================

class P2PMuonOrthogonalizer:
    """分布式 P2P Muon 正交化器。

    核心流程（step 方法）：
    1. gather_owned():  每个 rank 通过 P2P 通信收集自己拥有的参数的完整矩阵
    2. orthogonalize(): 对每个拥有的参数做 Newton-Schulz 正交化
    3. scatter_back():  将正交化后的分片传回各 rank

    内存优势：
    - All-Gather: 每个 rank 需要持有所有 N 个参数的完整矩阵 → O(N)
    - P2P: 每个 rank 只组装自己拥有的 ~N/world 个参数 → O(N/world)

    通信模式：
    - 不使用集合通信（all_gather / all_reduce）
    - 使用点对点通信（irecv / isend），每个 rank 只与 owner 通信
    - 通信量 = 每个 rank 发送自己持有分片给 owner，接收 owner 正交化后的分片

    单进程兼容：
    当 torch.distributed 不可用时，退化为本地操作（clone/copy），
    等价于 all-gather 语义，可用于单元测试。

    Attributes:
        params: 需要正交化的参数列表（2D 张量）
        group: 分布式进程组（None = 默认组）
        ns_steps: Newton-Schulz 迭代步数
        heads: 每个参数的注意力头数列表（None = 不做 per-head 正交化）
        world: 总 rank 数
        rank: 本 rank 编号
        ownership: 所有权映射
    """

    def __init__(
        self,
        params: List[Tensor],
        *,
        group: Optional[object] = None,
        ns_steps: int = 5,
        heads: Optional[List[Optional[int]]] = None,
    ):
        """初始化 P2P Muon 正交化器。

        Args:
            params: 需要正交化的 2D 参数张量列表。通常是从 param_groups
                    中筛选出的 use_muon=True 的参数。
            group: 分布式进程组。None 表示使用默认全局组。
            ns_steps: Newton-Schulz 正交化迭代步数。5 步在实践中
                      已足够收敛到 ||X*X^T - I|| < 1e-6。
            heads: 每个参数对应的注意力头数。
                   heads[i] = None 或 1 → 标准全矩阵正交化
                   heads[i] = N > 1 → Per-Head 正交化（按头拆分）
                   若不提供，默认对所有参数做标准正交化。
        """
        self.params = params
        self.group = group
        self.ns_steps = ns_steps
        # heads 列表：每个参数对应一个头数（None 表示不做 per-head）
        self.heads = heads or [None] * len(params)

        # 获取分布式环境信息
        # world = 总 rank 数，rank = 本 rank 编号
        # 当 dist 不可用或未初始化时，退化为单进程（world=1, rank=0）
        self.world = dist.get_world_size(group) if _dist_available() else 1
        self.rank = dist.get_rank(group) if _dist_available() else 0

        # 构建所有权映射：round-robin 分配
        self.ownership = OwnershipMap.round_robin(
            len(params), self.world, self.rank
        )

    # ----------------------------------------------------------
    # 3.1 第一阶段：收集（Gather）
    # ----------------------------------------------------------

    @torch.no_grad()
    def gather_owned(self, updates: List[Tensor]) -> Dict[int, Tensor]:
        """通过 P2P 通信收集本 rank 拥有的参数的完整矩阵。

        这是 P2P Muon 的核心通信步骤。流程：
        1. 遍历所有参数，判断每个参数的 owner
        2. 如果本 rank 是 owner：
           - 分配一个完整大小的缓冲区
           - 将自己持有的分片直接复制到缓冲区对应位置
           - 向其他 rank 发送 irecv 请求，接收它们持有的分片
        3. 如果本 rank 不是 owner：
           - 将自己持有的分片通过 isend 发送给 owner
        4. 等待所有通信完成

        为什么用 irecv/isend 而不是 all_gather？
        - all_gather 是集合通信，每个 rank 都拿到完整结果
          → 内存开销 O(N)，通信量 O(N * world)
        - irecv/isend 是点对点通信，每个 rank 只收发自己那片
          → 内存开销 O(N/world)，通信量 O(N)

        Args:
            updates: 所有参数的梯度更新张量列表。
                     updates[i] 是第 i 个参数的更新（2D 张量）。

        Returns:
            字典 {参数索引: 完整矩阵}，仅包含本 rank 拥有的参数。
        """
        full: Dict[int, Tensor] = {}

        # 单进程模式：直接 clone 即可（无需通信）
        if not _dist_available() or self.world == 1:
            for i in self.ownership.mine():
                full[i] = updates[i].clone()
            return full

        # 多进程模式：P2P 通信
        reqs: list = []  # 存储所有未完成的通信请求

        for i, upd in enumerate(updates):
            owner = self.ownership.owners[i]  # 第 i 个参数的拥有者 rank
            # 计算每个 rank 对这个参数的分片范围
            rngs = shard_ranges(upd.numel(), self.world)
            # 将 2D 张量展平为 1D，便于按元素范围切片
            flat = upd.reshape(-1)

            if owner == self.rank:
                # 本 rank 是拥有者：分配缓冲区，收集所有 rank 的分片
                buf = torch.empty_like(flat)
                # 先把自己持有的分片复制到缓冲区
                lo, hi = rngs[self.rank]
                buf[lo:hi] = flat[lo:hi]
                # 向其他 rank 发送接收请求
                for src in range(self.world):
                    if src == self.rank:
                        continue
                    slo, shi = rngs[src]
                    if shi > slo:
                        reqs.append(
                            dist.irecv(buf[slo:shi], src=src, group=self.group)
                        )
                # 组装完整矩阵（保持原始 shape）
                full[i] = buf.view_as(upd)
            else:
                # 本 rank 不是拥有者：将自己的分片发送给 owner
                lo, hi = rngs[self.rank]
                if hi > lo:
                    reqs.append(
                        dist.isend(
                            flat[lo:hi].contiguous(),  # contiguous 确保内存连续
                            dst=owner,
                            group=self.group,
                        )
                    )

        # 等待所有 P2P 通信完成
        for r in reqs:
            r.wait()

        return full

    # ----------------------------------------------------------
    # 3.2 第二阶段：分发（Scatter）
    # ----------------------------------------------------------

    @torch.no_grad()
    def scatter_back(
        self, orthogonalized: Dict[int, Tensor]
    ) -> List[Tensor]:
        """将正交化后的完整矩阵的分片传回各 rank。

        这是 gather_owned 的逆操作。流程：
        1. 遍历所有参数
        2. 如果本 rank 是 owner：
           - 将完整矩阵展平
           - 把自己需要的分片复制到输出缓冲区
           - 通过 send 将其他 rank 需要的分片发送出去
        3. 如果本 rank 不是 owner：
           - 通过 recv 从 owner 接收自己那片分片

        为什么用 send/recv 而不是散射？
        - send/recv 是同步的，确保数据就绪后才继续
        - 每个 rank 只接收自己需要的分片，不浪费内存

        Args:
            orthogonalized: 正交化后的完整矩阵字典。
                           {参数索引: 完整矩阵}，仅包含本 rank 拥有的参数。

        Returns:
            长度为 len(params) 的列表，第 i 个元素是第 i 个参数
            在本 rank 持有的分片（1D 张量）。
        """
        out: List[Tensor] = []

        for i, p in enumerate(self.params):
            # 计算第 i 个参数的分片范围
            rngs = shard_ranges(p.numel(), self.world)
            lo, hi = rngs[self.rank]  # 本 rank 持有的区间

            # 单进程模式：直接切片
            if not _dist_available() or self.world == 1:
                out.append(orthogonalized[i].reshape(-1)[lo:hi])
                continue

            owner = self.ownership.owners[i]
            # 分配接收缓冲区
            buf = torch.empty(hi - lo, dtype=p.dtype, device=p.device)

            if owner == self.rank:
                # 本 rank 是拥有者：将分片分发给各 rank
                flat = orthogonalized[i].reshape(-1)
                # 先把自己需要的分片复制到输出缓冲区
                buf.copy_(flat[lo:hi])
                # 通过 send 将其他 rank 需要的分片发送出去
                for dst in range(self.world):
                    if dst == self.rank:
                        continue
                    dlo, dhi = rngs[dst]
                    if dhi > dlo:
                        dist.send(
                            flat[dlo:dhi].contiguous(),
                            dst=dst,
                            group=self.group,
                        )
            else:
                # 本 rank 不是拥有者：从 owner 接收分片
                dist.recv(buf, src=owner, group=self.group)

            out.append(buf)

        return out

    # ----------------------------------------------------------
    # 3.3 完整步骤：收集 → 正交化 → 分发
    # ----------------------------------------------------------

    @torch.no_grad()
    def step(self, updates: List[Tensor]) -> List[Tensor]:
        """完整的 P2P Muon 正交化步骤。

        三阶段流程：
        1. gather_owned(): 通过 P2P 通信收集本 rank 拥有的参数的完整矩阵
        2. orthogonalize(): 对每个拥有的参数做 Newton-Schulz 正交化
           （支持标准模式和 Per-Head 模式）
        3. scatter_back(): 将正交化后的分片传回各 rank

        Args:
            updates: 所有参数的梯度更新张量列表（2D 张量）。

        Returns:
            长度为 len(params) 的列表，第 i 个元素是第 i 个参数
            正交化后在本 rank 持有的分片。
        """
        # 阶段 1：收集本 rank 拥有的参数的完整矩阵
        full = self.gather_owned(updates)

        # 阶段 2：对每个拥有的参数做正交化
        # _orthogonalize 内部根据 heads[i] 决定标准模式或 Per-Head 模式
        ortho: Dict[int, Tensor] = {
            i: _orthogonalize(mat, self.ns_steps, self.heads[i])
            for i, mat in full.items()
        }

        # 阶段 3：将正交化后的分片传回各 rank
        return self.scatter_back(ortho)

    # ----------------------------------------------------------
    # 3.4 内存分析工具
    # ----------------------------------------------------------

    def peak_buffer_elements(self) -> int:
        """计算 P2P 方案下本 rank 需要持有的峰值元素数。

        这是 P2P Muon 的核心优势：每个 rank 只组装自己拥有的参数，
        而不是 all-gather 整个模型。

        Round-robin 分配下，每个 rank 大约拥有 N/world 个参数，
        因此峰值缓冲约为 all-gather 的 1/world。

        Returns:
            本 rank 需要持有的峰值元素数
        """
        return sum(self.params[i].numel() for i in self.ownership.mine())

    def all_gather_buffer_elements(self) -> int:
        """计算 All-Gather 方案下每个 rank 需要的元素数（用于对比）。

        All-Gather 方案需要每个 rank 持有所有参数的完整矩阵，
        这是 P2P 方案要避免的。

        Returns:
            All-Gather 方案下的峰值元素数（= 所有参数总元素数）
        """
        return sum(p.numel() for p in self.params)


# ============================================================
# 4. 辅助函数
# ============================================================

def _dist_available() -> bool:
    """检查 torch.distributed 是否可用且已初始化。

    在单进程测试模式下，dist.is_available() 可能为 True
    但 dist.is_initialized() 为 False。两者都满足才认为
    分布式环境可用。

    Returns:
        True 如果分布式环境可用且已初始化
    """
    return dist.is_available() and dist.is_initialized()


def _orthogonalize(
    update: Tensor, steps: int, heads: Optional[int]
) -> Tensor:
    """对单个参数矩阵做正交化（标准模式或 Per-Head 模式）。

    根据 heads 参数决定正交化方式：
    - heads is None 或 heads <= 1: 标准全矩阵 Newton-Schulz
    - heads > 1: Per-Head 模式，按注意力头拆分后独立正交化

    Per-Head 模式的原理：
    将 [out_dim, in_dim] 的矩阵拆分为 [heads, head_dim, in_dim]，
    对每个头独立做 NS 正交化，再合并回 [out_dim, in_dim]。
    这样每个头的更新 RMS 独立，避免头间不均衡。

    Args:
        update: 待正交化的 2D 张量 [out_dim, in_dim]
        steps: Newton-Schulz 迭代步数
        heads: 注意力头数。None 或 <=1 表示标准模式。

    Returns:
        正交化后的张量，shape 与输入相同
    """
    if heads is None or heads <= 1:
        # 标准模式：全矩阵正交化
        return newton_schulz(update, steps)
    # Per-Head 模式：按头拆分 → 独立正交化 → 合并
    out_dim, in_dim = update.shape
    return newton_schulz(
        update.view(heads, out_dim // heads, in_dim), steps
    ).view(out_dim, in_dim)


# ============================================================
# 5. 一次性便捷函数
# ============================================================

def orthogonalize_owned(
    updates: List[Tensor],
    *,
    group: Optional[object] = None,
    ns_steps: int = 5,
    heads: Optional[List[Optional[int]]] = None,
) -> List[Tensor]:
    """一次性 P2P 正交化：创建临时 P2PMuonOrthogonalizer 并执行一步。

    这是一个便捷函数，适用于不需要缓存 orthogonalizer 实例的场景。
    内部流程与 P2PMuonOrthogonalizer.step() 完全一致。

    使用示例：
        # 在训练循环中
        ortho_updates = orthogonalize_owned(
            muon_updates,
            ns_steps=5,
            heads=[4, 4, 4, None, None],  # 前3个参数 per-head=4，后2个标准
        )

    Args:
        updates: 所有参数的梯度更新张量列表（2D 张量）。
        group: 分布式进程组。
        ns_steps: Newton-Schulz 迭代步数。
        heads: 每个参数的注意力头数列表。

    Returns:
        正交化后的分片列表，第 i 个元素是第 i 个参数的分片。
    """
    orth = P2PMuonOrthogonalizer(
        updates, group=group, ns_steps=ns_steps, heads=heads
    )
    return orth.step(updates)
