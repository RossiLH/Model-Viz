r"""MoonEP：完美均衡的专家并行调度（论文 §5.2.1 + 附录 E）。

设计动机（why）
-------------
专家并行的老问题是**路由负载不均**：Top-K 路由下不同专家在不同
micro-batch、不同层的负载都是波动的，某几个"热"专家会把 token 堆到少数
rank 上，带来三重代价：
  1. 计算瓶颈：hot rank 变慢，拖慢整次同步；
  2. 显存碎片：路由后的激活 shape 每步每层都不同，无法预分配固定缓冲；
  3. host 同步停滞：形状未知，host 每层都要和设备同步一次才能知道该分配多大
     显存、怎么发 grouped GEMM，直接打穿流水线。

MoonEP 的关键洞察（附录 E）
----------------------------
只要允许 **复制**（redundant expert），就可以让每个 TF 拿到的 token 数
**严格相等**（每个 rank 恰好 ``S*K`` 个），从而重获静态 shape 的三大好处：
  - 通信缓冲固定为 ``S*K``（而不是最坏 ``S*K*R``）；
  - grouped GEMM 的形状在 launch 前已知，无需 host 同步；
  - 负载零波动，无 hot rank。

问题变成：复制多少份才够？**附录 E 的定理**给出了答案：
    M(I) <= E/R  恒成立，且界基本是紧的。
即每 rank 预留给"非本地主家专家"的槽位（redundant slots）只需 ``E/R`` 个，
就**永远**存在一个可行规划——训练永不因规划失败中断。

构造性证明（plan_redundant_experts 就是它的直接翻译）：
  反复挑选"欠载 rank u"和"过载 rank o"，把 ``target - load(u)`` 个 token
  从 o 搬到 u。因为 ``load(o) > target >= deficit(u)``，o 永远拿得出足够
  token，u 一次就被填到 target、之后不再被碰。每轮填死一个 rank，至多
  ``R-1`` 轮结束；并且因为每个 rank 至多被填一次，它收来的全部远端 token
  来自**同一个**其他 rank，那个 rank 只 home ``E/R`` 个专家 —— 于是每 rank
  冗余副本 ≤ ``E/R``。

本实现在 CPU 上把"规划层"完整落地（plan_redundant_experts → RedundancyPlan
→ build_permutation / dispatch / combine），并用数值实验验证"恒定 S*K 缓冲
vs 最坏 S*K*R"以及 M(T)<=E/R 的界。真实 GPU 上的对称内存 / zero-copy
dispatch / prefetch 等内核级优化不属于本教学实现的范围（见 4.10 升华）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = [
    "RedundancyPlan",
    "plan_redundant_experts",
    "redundancy_lower_bound",
    "MoonEP",
]


@dataclass
class RedundancyPlan:
    """一次规划的结果：每个 (rank, expert) 持有的 token 数。"""

    n_experts: int
    n_ranks: int
    experts_per_rank: int
    #: ``holdings[rank][expert] = 该 rank 要为 expert 计算的 token 数``。
    holdings: list[dict[int, int]] | None = None
    #: 每个 rank 计算的**非本地**（redundant）专家集合。
    redundant: list[set[int]] | None = None
    #: 规划器执行的"搬运"次数。
    moves: int = 0

    def __post_init__(self) -> None:
        if self.holdings is None:
            self.holdings = [{} for _ in range(self.n_ranks)]
        if self.redundant is None:
            self.redundant = [set() for _ in range(self.n_ranks)]

    def home_rank(self, expert: int) -> int:
        """专家 ``expert`` 的主家（home）rank：按连续切片分配。"""
        return expert // self.experts_per_rank

    def tokens_per_rank(self) -> list[int]:
        """每个 rank 实际计算的 token 数（应全部等于 target）。"""
        return [sum(h.values()) for h in self.holdings]

    def max_redundant(self) -> int:
        """所有 rank 中冗余副本数目的最大值（界要 <= E/R）。"""
        return max((len(r) for r in self.redundant), default=0)

    @property
    def capacity(self) -> int:
        """附录 E 的界：E/R 个冗余槽位/rank（向上取整）。"""
        return -(-self.n_experts // self.n_ranks)

    def is_balanced(self) -> bool:
        """所有 rank 的负载严格相同。"""
        counts = self.tokens_per_rank()
        return len(set(counts)) <= 1

    def respects_bound(self) -> bool:
        """冗余副本数不超过容量界。"""
        return self.max_redundant() <= self.capacity

    def summary(self) -> dict[str, float]:
        return {
            "balanced": float(self.is_balanced()),
            "max_redundant": float(self.max_redundant()),
            "capacity": float(self.capacity),
            "moves": float(self.moves),
        }


def plan_redundant_experts(
    loads: list[int] | Tensor,
    n_ranks: int,
    *,
    experts_per_rank: int | None = None,
) -> RedundancyPlan:
    """规划一个完美均衡的分配（附录 A 构造的直接翻译）。

    Args:
        loads: ``loads[e]`` = 路由到专家 ``e`` 的 token-专家指派个数。
        n_ranks: 专家并行度 R。
        experts_per_rank: 每 rank 本地专家数（缺省 = ceil(E/R)）。

    Returns:
        均衡的 RedundancyPlan。要求总指派数能被 R 整除
        （Top-K × S × R 的算术让 target 恰为 S*K）。

    算法（一步步验证界）：
      1. 初始：每个 rank 只持有它 home 专家的 token（``home = min(e // epr, R-1)``）。
      2. 主循环：
         - 找第一个欠载 rank ``u``（load(u) < target）；
         - 找当前最过载的 rank ``o``；
         - 从 o 的 holdings 中（从大专家开始，控制冗余副本数）取出
           ``need = target - load(u)`` 个 token 搬到 u；
         - o 一定是够拿的（load(o) > target >= need），所以 u 恰好落 target；
      3. 终止：某个 rank 一旦被填满再也不被触碰，至多 R-1 轮收敛。
    """
    if isinstance(loads, Tensor):
        loads = [int(x) for x in loads.tolist()]
    n_experts = len(loads)
    experts_per_rank = experts_per_rank or -(-n_experts // n_ranks)

    total = sum(loads)
    if total % n_ranks != 0:
        raise ValueError(
            f"总 token-专家指派数 {total} 不能被 {n_ranks} 个 rank 整除；"
            "MoonEP 的静态 shape 需要一个能整除的总量"
        )
    target = total // n_ranks

    plan = RedundancyPlan(n_experts, n_ranks, experts_per_rank)
    # 初始摆放：每家放自家专家的 token。
    for e, load in enumerate(loads):
        if load:
            home = min(e // experts_per_rank, n_ranks - 1)
            plan.holdings[home][e] = plan.holdings[home].get(e, 0) + load

    current = [sum(h.values()) for h in plan.holdings]

    while True:
        # 找到第一个欠载 rank——它的负载将被精确补齐。
        under = next((r for r in range(n_ranks) if current[r] < target), None)
        if under is None:
            break  # 都已均衡（含整除舍入内）
        # 找最过载的 rank 作为"供应者"。
        over = max(range(n_ranks), key=lambda r: current[r])
        if current[over] <= target:
            break  # 整数舍入内已均衡
        need = target - current[under]

        # 从 over 手里按"大专家优先"搬 need 个 token 给 under。
        # 大专家优先 => 尽可能少搬不同的专家 => 冗余副本数少。
        for expert in sorted(plan.holdings[over], key=lambda e: -plan.holdings[over][e]):
            if need == 0:
                break
            take = min(plan.holdings[over][expert], need)
            plan.holdings[over][expert] -= take
            if plan.holdings[over][expert] == 0:
                del plan.holdings[over][expert]
            plan.holdings[under][expert] = plan.holdings[under].get(expert, 0) + take
            # 非本地专家级别才计入冗余。
            if expert // experts_per_rank != under:
                plan.redundant[under].add(expert)
            need -= take

        current = [sum(h.values()) for h in plan.holdings]
        plan.moves += 1
        if plan.moves > n_ranks * n_ranks:  # 保险丝：理论上不会触发
            raise RuntimeError("MoonEP 规划器未收敛")

    return plan


def redundancy_lower_bound(loads: list[int], n_ranks: int, experts_per_rank: int) -> int:
    """冗余副本数的一个下界（廉价健全性检查）。

    只要有任一 rank 的 home 负载低于 target，它就必须至少算 1 个非本地专家，
    所以 max_redundant >= 1 一旦路由不均衡就成立。论文用整数线性规划求精确
    最优作参考，这里用下界防止规划器做匪夷所思的事。
    """
    target = sum(loads) // n_ranks
    home = [0] * n_ranks
    for e, load in enumerate(loads):
        home[min(e // experts_per_rank, n_ranks - 1)] += load
    return 1 if any(h < target for h in home) else 0


class MoonEP:
    """静态 shape 下的专家派发与汇总。

    plan -> build_permutation -> dispatch/combine 三段式。由于每个 rank 都拿到
    恰好 ``S*K`` 个 token，通信缓冲形状已知，permutation 一步到位（fuse 的
    permute/unpermute），grouped GEMM 的形状也能在 launch 前确定。
    """

    def __init__(self, n_experts: int, n_ranks: int):
        self.n_experts = n_experts
        self.n_ranks = n_ranks
        self.experts_per_rank = -(-n_experts // n_ranks)
        self.last_plan: RedundancyPlan | None = None

    # -- 规划 ----------------------------------------------------------------

    def plan(self, expert_ids: Tensor) -> RedundancyPlan:
        """从一维路由专家 id 张量规划。

        Args:
            expert_ids: ``(N*K,)`` 每 token 的 K 个路由专家 id（扁平）。

        Returns:
            均衡计划。单进程实现不跨 rank 收集，loads 直接用 bincount。
        """
        loads = torch.bincount(expert_ids.reshape(-1), minlength=self.n_experts)
        plan = plan_redundant_experts(loads, self.n_ranks, experts_per_rank=self.experts_per_rank)
        self.last_plan = plan
        return plan

    # -- 派发与合成 ----------------------------------------------------------

    def build_permutation(
        self, expert_ids: Tensor, plan: RedundancyPlan
    ) -> tuple[Tensor, Tensor, Tensor]:
        """返回 ``(order, destination_rank, expert_of_slot)``。

        - ``order``：把"扁平指派序列"重排为"按 rank / 按专家分组"的顺序，
          派发只需做一次 gather；
        - ``expert_of_slot``：目标槽位该算哪个专家，供 grouped GEMM 不看数据
          直接 launch。
        """
        flat = expert_ids.reshape(-1)

        # 每个 (rank, expert) 的剩余容量，直接来自 plan。
        remaining = [dict(h) for h in plan.holdings]
        dest = torch.empty_like(flat)
        cursor = {}
        for idx in range(flat.numel()):
            e = int(flat[idx])
            r = cursor.get(e)
            if r is None or remaining[r].get(e, 0) == 0:
                r = next(
                    (rr for rr in range(self.n_ranks) if remaining[rr].get(e, 0) > 0),
                    None,
                )
                if r is None:
                    raise RuntimeError(f"专家 {e} 已无容量")
                cursor[e] = r
            remaining[r][e] -= 1
            dest[idx] = r

        # 按 (rank, expert) 排序 => 每个 rank 的切面连续、段内按专家连续，
        # 恰好构成一个 grouped GEMM。
        key = dest.to(torch.int64) * self.n_experts + flat.to(torch.int64)
        # 老 torch（1.11）的 argsort 不支持 stable=，加微小扰动保证原序内的稳定。
        idx = torch.arange(key.numel(), device=key.device, dtype=torch.int64)
        order = torch.argsort(key * key.numel() + idx)
        return order, dest, flat[order]

    def dispatch(self, tokens: Tensor, order: Tensor) -> Tensor:
        """fused permute：一次 gather 进专家分组的槽位。"""
        return tokens.index_select(0, order)

    def combine(self, expert_out: Tensor, order: Tensor) -> Tensor:
        """fused unpermute：dispatch 的精确逆操作。"""
        out = torch.empty_like(expert_out)
        return out.index_copy_(0, order, expert_out)

    def static_shape(self, tokens_per_rank: int) -> tuple[int, ...]:
        """每个 rank 使用的计算 shape（launch 前已知）。"""
        return (tokens_per_rank,)