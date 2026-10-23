r"""进程网格构建（论文 §5.2 的 PP x EP x CP x DP mesh）。

K3 预训练把流水线并行 + 虚拟 stage、专家并行、ZeRO-1 数据并行、Pipeline
ZeRO-2 梯度分片、上下文并行组合在一起。为了让其余代码能直接问
"我的专家并行组是哪些 rank"而不重复做 rank 算术，本模块统一持有这个 mesh。

rank 布局约定（与 kimi3-master 一致）：
    rank = dp * (pp * ep * cp) + pp_idx * (ep * cp) + cp_idx * ep + ep_idx
即 **ep 最快**，之后 cp、pp、最新是 dp。为什么 ep 最快？
专家派发（all-to-all）是通信最密的 collective，把它放到相邻 rank 上，
hop 距离最短，这正是 mesh 布局的最重要考量。

本实现是 CPU 可运行的"算术 + 网格成员"复现（不依赖 torch.distributed）：
  - ParallelGroups 记录 rank 坐标（pp_rank / ep_rank / cp_rank / dp_rank）；
  - group_of(kind) 返回当前 rank 在对应轴上的组员 rank（list[int]）；
  - mesh_ranks(pp, ep, cp, dp) 生成完整网（用于 {} 可视化/测试）。
真实多卡中，``dist.new_group`` 按同样的 start/stride 构造进程组
（每组内 rank 互为 "new_group 的成员"，见 kimi-master 的 build()）。

依赖约定：world == pp * ep * cp * dp（dp 由除法推出）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

__all__ = [
    "DeviceGroup",
    "ParallelGroups",
    "mesh_ranks",
    "infer_dp",
    "groups_of",
    "group_is_valid",
]


class DeviceGroup(ABC):
    """设备组的抽象接口（并行底层可插拔）。

    保留给现有骨架/接口测试使用；具体实现见 :class:`ParallelGroups`。
    """

    @property
    @abstractmethod
    def world_size(self) -> int: ...


@dataclass
class ParallelGroups:
    """单个 rank 上的网格坐标与该网格的全部组。

    Attributes:
        world_size / rank: 全局序号。
        pp / ep / cp / dp: 各轴度数（乘积 = world_size）。
        pp_rank, ep_rank, cp_rank, dp_rank: 本 rank 在各轴的坐标。
        groups: kind -> list[list[int]]，每轴的全部组（rank 名单）。
    """

    world_size: int
    rank: int
    pp: int = 1
    ep: int = 1
    cp: int = 1
    dp: int = 1
    _groups: dict[str, list[list[int]]] = field(default_factory=dict, repr=False)

    # -- 坐标 ----------------------------------------------------------------

    @property
    def pp_rank(self) -> int:
        return (self.rank // (self.ep * self.cp)) % self.pp

    @property
    def ep_rank(self) -> int:
        return self.rank % self.ep

    @property
    def cp_rank(self) -> int:
        return (self.rank // self.ep) % self.cp

    @property
    def dp_rank(self) -> int:
        return self.rank // (self.ep * self.cp * self.pp)

    def group_of(self, kind: str) -> list[int]:
        """返回本 rank 所在的 kind 轴组（rank 名单，按升序）。"""
        return [ranks for ranks in self._groups_kind(kind) if self.rank in ranks][0]

    def _groups_kind(self, kind: str) -> list[list[int]]:
        if kind in self._groups:
            return self._groups[kind]

        stride, size = {
            "ep": (1, self.ep),
            "cp": (self.ep, self.cp),
            "pp": (self.ep * self.cp, self.pp),
            "dp": (self.ep * self.cp * self.pp, self.dp),
        }[kind]
        return groups_of(self.world_size, stride, size)

    def describe(self) -> str:
        return (
            f"rank {self.rank}/{self.world_size}: "
            f"pp={self.pp_rank}/{self.pp} ep={self.ep_rank}/{self.ep} "
            f"cp={self.cp_rank}/{self.cp} dp={self.dp_rank}/{self.dp}"
        )


def infer_dp(world_size: int, pp: int, ep: int, cp: int) -> int:
    """dp 不是配置参数，而是由 world // (pp*ep*cp) 推出。"""
    return world_size // (pp * ep * cp)


def mesh_ranks(pp: int, ep: int, cp: int, dp: int) -> list[list[int]]:
    """生成 R x R 的完整 mesh（矩阵 m[r][axis] = r 在 axis 的坐标）。

    Returns:
        ``ranks[[dp, pp, cp, ep]]``——二维列表，便于 node 可视化 / 测试。
    """
    size = dp * pp * ep * cp
    out = []
    for r in range(size):
        epc = r % ep
        cpc = (r // ep) % cp
        ppc = (r // (ep * cp)) % pp
        dpc = r // (ep * cp * pp)
        out.append([dpc, ppc, cpc, epc])
    return out


def groups_of(world_size: int, stride: int, size: int) -> list[list[int]]:
    """生成所有沿一个轴的 rank 组（与 kimi3-master 的 ``build()`` 逐位一致）。

    semantics：以 ``start`` 为底，每隔 ``stride`` 取 ``size`` 个 rank。
    有效的组必须落在 ``[0, world_size)`` 之内。写法直接对照参考实现：
    ``if start % (stride * size) >= stride: continue`` —— 它保证 start 是轴内
    的"起点"（即轴位坐标的最低位）。
    """
    groups: list[list[int]] = []
    for start in range(world_size):
        if start % (stride * size) >= stride:
            continue
        ranks = [start + i * stride for i in range(size)]
        if any(r >= world_size for r in ranks):
            continue
        groups.append(ranks)
    return groups


def group_is_valid(pp: int, ep: int, cp: int, world_size: int) -> bool:
    """世界是否整除 pp*ep*cp（否则网格非法）。"""
    return world_size % (pp * ep * cp) == 0


def _demo_asserts() -> None:  # pragma: no cover
    g = ParallelGroups(world_size=32, rank=7, pp=2, ep=4, cp=2, dp=2)
    assert g.ep_rank == 3
    assert g.cp_rank == 1
    assert g.pp_rank == 0
    assert g.dp_rank == 1