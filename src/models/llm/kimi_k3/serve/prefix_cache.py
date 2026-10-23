"""KDA-aware 前缀缓存（KDA-Infra.md §2 双粒度）。

设计动机
--------
混合架构（K3 的 KDA:MLA = 3:1）里两类注意力对「前缀复用」的约束完全不同：

1. **MLA 侧**：KV cache 是**逐 token 条目**。命中前缀时可以把整段 KV 直接
   从索引取回来用，不需要任何额外前置条件——因此它可以做**细粒度**缓存，
   以 512 token 的 hash block 为单位，逐块比对 chained hash。
   这是经典的 vLLM 式前缀缓存思路：``hash(prefix)`` 命中即复用。

2. **KDA 侧**：递归状态 ``S(B,H,dk,dv)`` 是**递推得到的**，代价与序列长度成
   正比。要跳过这段递推，必须在某个**稀疏边界**把当时的中间状态快照下来
   （checkpoint）。分层存储「边界 token 位置 -> 该位置每个 KDA group 的
   状态快照」，且状态是一个逐层递推的整体，各层的 checkpoint 必须**原子地**
   一起存在（少一层就无法正确恢复）。

因此「可安全复用的前缀」不是简单的、连续的 MLA 块集合，而是「MLA 物理块能
逐块匹配 且 最深边界处每个 KDA group 都有 checkpoint」的交集。这要求两阶段
lookup：

* **阶段一**：按 chained hash 逐 block 匹配 MLA 侧，能走多远走多远；
* **阶段二**：从最深匹配边界往回退，找到一个「每个 KDA group 都存了
  checkpoint」的边界，才真正命中——hit 永远是 hash block 的端点，
  **不要求**是 physical block 的整数倍。

设计决策
--------
* **chained hash 而非每块独立 hash**：每块 digest 认证「直到该端点的整个
  前缀」。这样任意两个请求只要在某个端点 hash 相同，就保证其前面的全部 token
  完全相同——不存在「高 prefix 撞车」的歧义，命中=内容完全一致。
* **ready 标志**：注册是异步的（先登记 digest、后填充内容）。`ready=False`
  的块参与索引但不参与 lookup，避免拿到「登记过但 cache 尚未填好」的 false hit。
* **physical_block**：`physical_block_tokens`（如 6144）= 12 个 hash block。
  记录每个 hash block 落在哪个 physical block，用于给上层（页池复用/共享）
  提供「我需要 pin 住哪些物理块」的答案，见 ``lookup`` 阶段二。
* **原子 checkpoint**：`kda_checkpoints` 按 group 存集合，`checkpoint_available`
  要求**所有 group** 都有该边界，缺一个就视为不可用——这正是原子性的表达。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

__all__ = ["PrefixCache", "HashBlock", "LookupResult", "chained_hash"]


def chained_hash(previous: str, tokens: list[int]) -> str:
    """认证到该端点的整个前缀。

    `previous` 是上一个块的 digest；把它和本块的 token 序列一起喂进 sha256，
    使最终 digest 依赖**之前的所有内容**。这样 hit 校验只需比对 digest，
    就可以确信整个前缀逐 token 一致，无需把 token 再比较一遍。
    """
    h = hashlib.sha256(previous.encode())
    h.update(",".join(map(str, tokens)).encode())
    return h.hexdigest()


@dataclass
class HashBlock:
    """一个 hash block 的索引条目。

    存「该端点的前缀 digest + 块尾 token 位置 + 逻辑 physical block 归属」。

    * ``digest``        —— chained hash，索引的 key（也是本类的 key）。
    * ``end_token``     —— 本块覆盖到第几个 token（含，块尾端点，必为 512 整数倍）。
    * ``physical_block``—— 该端点在页池里落到的 physical block 号。
    * ``ready``         —— 内容是否已填充（异步登记时为 False，填充后置 True）。
                           lookup 只认 ``ready=True`` 的块。
    """

    digest: str
    end_token: int
    physical_block: int
    ready: bool = True


@dataclass
class LookupResult(object):
    """一次 lookup 的返回值：命中到什么程度、命中哪些物理块。

     * ``matched_tokens`` — MLA 侧逐块匹配到的端点数（能被复用的前缀长度）。
     * ``physical_blocks`` — 命中的物理块 id 列表（**已去重**，供上层 pin/共享）。
     * ``kda_checkpoint``  — 命中的 KDA checkpoint 边界；None 表示没有命中。

    ``hit`` 是性质属性：只要匹配到 token 就算「前缀命中」。
    """

    matched_tokens: int
    physical_blocks: list[int] = field(default_factory=list)
    kda_checkpoint: int | None = None

    @property
    def hit(self) -> bool:
        return self.matched_tokens > 0


@dataclass
class PrefixCache:
    """双粒度前缀缓存的全部状态与操作。

    粒度参数：
     * ``hash_block_tokens``     — MLA 细粒度匹配的块大小（默认 512）。
     * ``physical_block_tokens`` — 物理块（页池 share 单元）大小（默认 6144，
                                   即 12 个 hash block）。
     * ``n_kda_groups``           — KDA 的 group 数。KDA 状态按 group 各自
                                   保存 checkpoint，原子性 = 所有 group 都要有。

    内部状态：
     * ``index``           — digest -> HashBlock 的倒排索引。
     * ``kda_checkpoints`` —  group -> 该边界已存 checkpoint 的边界集合。
     * ``free_list``       — 可复用的物理块 id（回收后再分配）。
     * ``_next_block``     — 单调递增的物理块分配器（free_list 为空时使用）。
    """

    hash_block_tokens: int = 512
    physical_block_tokens: int = 6144
    n_kda_groups: int = 1

    index: dict[str, HashBlock] = field(default_factory=dict)
    #: group -> 该边界已有完整 KDA checkpoint（原子性的体现：per-group 集合）
    kda_checkpoints: dict[int, set[int]] = field(default_factory=dict)
    free_list: list[int] = field(default_factory=list)
    _next_block: int = 0

    def __post_init__(self) -> None:
        # 为每个 group 建一个空集合：保证 checkpoint_available 遍历时不缺 key。
        # field 默认只有一份 dict 会被共享，所以必须在 __post_init__ 里逐个
        # setdefault，避免多个 PrefixCache 实例共享同一份 group 状态。
        for g in range(self.n_kda_groups):
            self.kda_checkpoints.setdefault(g, set())

    @property
    def hash_blocks_per_physical(self) -> int:
        """一个 physical block 装多少个 hash block（读数）。"""
        return self.physical_block_tokens // self.hash_block_tokens

    # -- 注册  -----------------------------------------------------------

    def allocate_from_free(self) -> int:
        """从 free_list 弹出一个物理块号；空则分配一个新的全局递增 id。

        ``pop(0)`` 选 FIFO 而非栈式 ``pop()``：块回收后再次分配时
        尽量复用较早释放的 id，保持分配次序稳定、易于调试。
        """
        if self.free_list:
            return self.free_list.pop(0)
        ident = self._next_block
        self._next_block += 1
        return ident

    def register(self, tokens: list[int], *, ready: bool = True) -> list[str]:
        """把一次请求的前缀全部分成 hash 块并登记进索引，返回各块 digest。

        只有完整长度 ``hash_block_tokens`` 的倍数的前缀才登记（末尾不足
        一块的尾部 token 不碰——lookup 恢复也只能恢复到块端点）。
        物理块归属：每写满 ``physical_block_tokens`` 就换一个物理块；
        但**结尾恰在物理边界**时（``end >= len(tokens)``）不切换到下一个
        物理块——末尾不必先占新页，留给后续 register 续接时再分。
        """
        digests: list[str] = []
        previous = ""  # 链式起点：空字符串代表「从头开始」
        block = self.allocate_from_free()
        for i in range(0, len(tokens) - self.hash_block_tokens + 1, self.hash_block_tokens):
            chunk = tokens[i : i + self.hash_block_tokens]
            previous = chained_hash(previous, chunk)
            end = i + self.hash_block_tokens
            if end % self.physical_block_tokens == 0 and end < len(tokens):
                # 已写满一个 physical block，且后面还有内容 -> 换到下一个物理块。
                # 「且 end < len(tokens)」是关键：若结尾恰在物理边界上，
                # 则这块仍要留在**当前**物理块里（后续不继续写，不预占新页）。
                block = self.allocate_from_free()
            self.index[previous] = HashBlock(previous, end, block, ready=ready)
            digests.append(previous)
        return digests

    def mark_ready(self, digest: str) -> None:
        """异步登记机制：digest 已登记但内容填充完成后，把标记置为 ready。

        lookup 到 ``ready=False`` 的块会立刻 break —— 保证绝不命中
        「登记过但还没把 KV/状态填进物理块」的假前缀。
        """
        if digest in self.index:
            self.index[digest].ready = True

    # -- KDA checkpoint  ------------------------------------------------

    def persist_kda_checkpoint(self, boundary: int, *, groups: list[int] | None = None) -> None:
        """在 ``boundary`` 处为指定（或全部）group 存一个 KDA checkpoint。

        KDA 快照只能在 hash block 端点保存：递推状态只在这些位置与 MLA
        缓存对齐，其他位置恢复不了。这里把边界不合法直接 raise——
        宁可启动早期报错，也不让一个错误对齐的 checkpoint 静默污染索引。
        """
        if boundary % self.hash_block_tokens != 0:
            raise ValueError(
                f"KDA checkpoints may only sit on hash-block boundaries; got {boundary}"
            )
        for g in groups if groups is not None else range(self.n_kda_groups):
            self.kda_checkpoints[g].add(boundary)

    def evict_kda_checkpoint(self, boundary: int, *, groups: list[int] | None = None) -> None:
        """把 ``boundary`` 从指定（或全部）group 的 checkpoint 集合里移除。

        指定 ``groups`` 时只逐出部分 group（把「部分缺少」的状态补全成
        完整、或让某个边界从原子可用变为不可用）；默认遍历全部 group，
        用 ``discard``（不存在即 no-op），保证逐出是幂等的，不会残留
        某个 group 有、某个 group 无的半原子状态。
        """
        targets = groups if groups is not None else range(self.n_kda_groups)
        for g in targets:
            self.kda_checkpoints[g].discard(boundary)

    def checkpoint_available(self, boundary: int) -> bool:
        """边界被每个 group 都有 checkpoint 才算可用。

        这是「原子性」的查询面：KDA 递推状态必须整组一起恢复，
        只要有一个 group 缺快照，这个边界就不可复用。
        """
        return all(boundary in group for group in self.kda_checkpoints.values())

    # -- lookup  -----------------------------------------------------------

    def lookup(self, tokens: list[int]) -> LookupResult:
        """两段式 lookup：MLA 一键逐块 + KDA 回退到有 checkpoint 的边界。

        阶段一（MLA）：
            从 t=0 开始按 ``hash_block_tokens`` 连续计算 chained hash，
            每个 digest 去 ``index`` 查。只要块在且 ``ready``，就继续；
            一旦 miss 或未就绪就立刻停止——前缀命中必须是**连续**的。
            不连续的后继 token 不属于该前缀，即使它的 hash 恰好在索引里。

        阶段二（KDA 原子性约束）：
            MLA 匹配得多深并不直接决定复用有多长——还必须「最深匹配边界
            vs 每个 group 是否有 checkpoint」。从最深匹配边界往回退，
            找到 endpoint 处 ``checkpoint_available`` 而又能覆盖前面的。
            这样命中的 ``matched_tokens`` 一定落在 hash block 端点上，
            不必是 physical block 的整数倍（6144 的倍率线）。

            同时收集「该前缀实际用到的物理块」——阶段 1 产出的候选块
            集去重，供上层执行「共享这些物理块」的动作。
        """
        previous = ""
        candidates: list[tuple[int, int]] = []  # (end_token, physical_block)

        # 阶段 1：连续哈希块（MLA 细粒度匹配）
        for i in range(0, len(tokens) - self.hash_block_tokens + 1, self.hash_block_tokens):
            chunk = tokens[i : i + self.hash_block_tokens]
            previous = chained_hash(previous, chunk)
            entry = self.index.get(previous)
            if entry is None or not entry.ready:
                break
            candidates.append((entry.end_token, entry.physical_block))

        # 阶段 2 ：从最深边界回退，找每个 KDA group 都有 checkpoint 的边界
        for end, _block in reversed(candidates):
            if self.checkpoint_available(end):
                # 去重收集命中物理块：多个 hash block 共享同一物理块时只看一次
                seen: list[int] = []
                for boundary, block in candidates:
                    if boundary <= end and block not in seen:
                        seen.append(block)
                return LookupResult(end, seen, end)
        return LookupResult(0, [], None)

    # -- accounting  -------------------------------------------------------

    def recompute_saved(self, result: LookupResult, total_tokens: int) -> int:
        """报告「本请求免重算的 token 数」。

        上限是 ``result.matched_tokens``（命中多深就省多深），但请求的总
        ``total_tokens`` 可能比命中还短——比如请求只有 300 token 却命中
        512？那么 300 才真正免于重算。``max(0, ...)`` 兜底避免无命中时
        出现负数（matched_tokens=0、total=0 的极端情况）。
        """
        return max(0, min(result.matched_tokens, total_tokens))