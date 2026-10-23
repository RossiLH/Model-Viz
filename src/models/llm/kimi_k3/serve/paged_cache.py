"""统一页池 — 混合 KDA/MLA 注意力的 paged cache（KDA-Infra.md §2）。

设计动机
--------
MLA 的 KV cache 随序列增长，需要按 token 分页（vLLM/PagedAttention 的做法）；
KDA 的递归状态 ``S(B,H,dk,dv)`` 尺寸固定，每请求一份、不随序列增长。

两种 cache 的「物理本质」完全不同，但在字节层面可以归一化：
把两种 page 放进**同一个字节宽的页池**，用**一条 free list** 统一分配/回收/引用计数。
这样上层（前缀缓存、投机解码、KCP）只需操作 ``PagePool`` 一个对象，
不必区分「这是 MLA 页还是 KDA 页」——这就是「统一页池」的含义。

``pack_heads`` 把 ``(B,H,dk,dv)`` 状态按 head 连续摊平成字节流：
每 head 的一串字节是跨节点传输的最小单元（本项目只做字节计量，
不关心元素排布细节，只保证 ``pack`` -> ``unpack`` 无损还原）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor

__all__ = ["PageKind", "Page", "PagePool", "PagedCache", "pack_heads", "unpack_heads"]


class PageKind(str, Enum):
    """页的「语义种类」。同一个页池里混存两类页，靠 kind 区分用途。"""

    MLA_KV = "mla_kv"
    KDA_STATE = "kda_state"


@dataclass
class Page:
    """页池中的一个物理页。

    属性说明：
    * ``ident``    —— 全局唯一 id，alloc 时从 free list 弹出，release 时回收。
    * ``kind``     —— 本页承载的数据类型（MLA_KV 或 KDA_STATE）。
    * ``data``     —— 一段固定字节宽的扁平 Tensor（pad 到整页）。
                      对 MLA：逐 token 填充；对 KDA：整页写一次。
    * ``refcount`` —— 引用计数。预留给「共享页」（前缀缓存/多请求端起同一页）
                      场景；=0 的时刻即释放回 free list 的时刻。
    * ``filled``   —— 已用元素数（以元素为单位，不是字节）。load 时按此切片，
                      区分「真实数据」与「pad 到页宽的填充」。
    """

    ident: int
    kind: PageKind
    data: Tensor
    refcount: int = 0
    filled: int = 0

    def is_full(self, tokens_per_page: int) -> bool:
        """判断该页是否已写满 ``tokens_per_page`` 个 token。

        这是 MLA 分页的核心判据：``PagedCache.append_mla`` 在
        当前尾页 ``is_full`` 时才从池里再分配新页。
        """
        return self.filled >= tokens_per_page


@dataclass
class PagePool:
    """单一 free list 管理两种 page kind —— 统一页池的点。

    ``free`` 是一份 id 的列表，所有类型（MLA/KDA）共用同一份分配池。
    不区分「KDA 内存」和「MLA 缓存内存」两张表——这正是与
    传统按类型分内存池的方案的本质区别。
    """

    page_bytes: int
    n_pages: int
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    pages: dict[int, Page] = field(default_factory=dict)
    free: list[int] = field(default_factory=list)
    _next: int = 0

    def __post_init__(self) -> None:
        # free 从 0..n-1 铺满：初始状态是「全部页空闲」。
        # 注意 dataclass 的 field 默认值只对新构造对象生效，
        # 因此 free 不直接取值，而在 __post_init__ 里真正初始化，
        # 避免同一个默认 list 对象被所有实例共享。
        self.free = list(range(self.n_pages))

    @property
    def elements_per_page(self) -> int:
        """每页可容纳的元素个数（按 torch 元素计量，而非字节）。

        ``finfo(dtype).bits // 8`` 把一个元素编成一个字节数，
        于是 ``page_bytes`` 就能换算成 ``elements_per_page``，
        供 alloc 时 `torch.zeros` 确定页的大小。
        """
        return self.page_bytes // (torch.finfo(self.dtype).bits // 8)

    def allocate(self, kind: PageKind) -> Page:
        """从 free list 弹出一个 id 分配为 ``kind`` 类型的新页。

        分配语义：
        1. 若 free list 空 -> 抛 ``MemoryError``（页池耗尽，调用方决定怎么处理）。
        2. 从 free list 头部弹出 id（``pop(0)``，先进先出）。
           这里用 FIFO 而非栈式 ``pop()``：id 是「先分配先归还」的
           自然顺序，便于外层把 id 当作稳定的分配次序来跟踪/调试。
           页池 n 通常几百~几千，O(n) 的 pop(0) 开销可忽略。
        3. 新页 ``refcount=1``，data 置零（避免读到同一物理页的残留数据）。
        """
        if not self.free:
            raise MemoryError("page pool exhausted")
        ident = self.free.pop(0)
        page = Page(
            ident=ident,
            kind=kind,
            data=torch.zeros(self.elements_per_page, dtype=self.dtype, device=self.device),
            refcount=1,
        )
        self.pages[ident] = page
        return page

    def release(self, ident: int) -> None:
        """对该页引用计数减一；归零则把 id 放回 free list。

        ``refcount`` 从 1 开始（分配即持有），所以第一次 release 会把它降为 0，
        页即归还。归还后 ``pages`` 里不再有这个 id，
        若调用方仍持 Page 对象去 ``data``，只能读到已释放的副本 ——
        这本身就是设计意图：缓存层负责保证用后释放。
        """
        page = self.pages.get(ident)
        if page is None:
            # 重复释放（double-free）在这里是 no-op：
            # 一次操作系统级页表的 double-free 会崩，
            # 但池层的 id 已经不在 pages 里，失败静默即可。
            return
        page.refcount -= 1
        if page.refcount <= 0:
            del self.pages[ident]
            self.free.append(ident)

    def retain(self, ident: int) -> None:
        """显式引用 +1：当页要被多个调用方共享（如前缀缓存命中时被多次引用）时使用。"""
        self.pages[ident].refcount += 1

    def utilization(self) -> float:
        """池利用率 = 已占用页数 / 总页数。占用 = 1 - 空闲占比。

        这是页池健康度的最直观指标：
        接近 1 说明页池可能过小（分配会失败）；
        长期很低说明池设得太大，浪费内存。
        """
        return 1.0 - len(self.free) / max(1, self.n_pages)


def pack_heads(state: Tensor) -> Tensor:
    """``(B, H, dk, dv)`` -> 按 head 连续的一维 flat 字节流。

    为什么不直接 flat？因为 head 是跨节点传输的最小语义单元 ——
    后续 KCP（上下文并行）按 head 拆分通信，所以这里保留 head 边界。
    ``contiguous()`` 保证 reshape 后的内存是连续的，没有 stride 跳变。
    """
    return state.reshape(-1).contiguous()


def unpack_heads(flat: Tensor, heads: int, dk: int, dv: int) -> Tensor:
    """``pack_heads`` 的逆操作：一维 flat -> ``(B, H, dk, dv)``。

     batch 从总长度反推：``numel(flat) // (H*dk*dv)``。
     注意：只切 ``batch * H * dk * dv`` 个元素，
     即使页里有 pad（未写的零），也丢弃，保证不会把 pad 值解进状态。
    """
    batch = flat.numel() // (heads * dk * dv)
    return flat[: batch * heads * dk * dv].reshape(batch, heads, dk, dv)


@dataclass
class PagedCache:
    """每请求一份的共享池视图（view over a PagePool）。

    不是独立的内存所有权，而是**对 pool 的按请求组织**：
    * ``mla_pages`` 按顺序记录本请求 MLA 占的页 id 序列（这个序列顺序对应 token 顺序）。
    * ``kda_pages`` 是 ``layer -> page id`` 映射，每层 KDA 状态占一页。
    * 请求结束时调 ``release()`` 把所有这些页归还给池，视图自己清空。

    两类页共享同一个池内存池，一个请求内可同时分页进行 —— 统一页池的完整约束。
    """

    pool: PagePool
    tokens_per_page: int = 512
    mla_pages: list[int] = field(default_factory=list)
    kda_pages: dict[int, int] = field(default_factory=dict)  # layer -> page id
    length: int = 0

    def append_mla(self, n_tokens: int) -> None:
        """接续追加 ``n_tokens`` 到 MLA 页序列。

        分页规则（PagedAttention 语义）：
        1. 若当前末尾页已满（``is_full``）或还没有任何页，则 alloc 一个新页。
        2. 把本次追加的 token 尽量填进当前尾页，最多填满，剩余的部分滚到新页。
        3. ``length`` 只是本请求的逻辑 token 计数器（展示用），与页分配解耦。

        返回值 count：如果中途遇到页池用尽会抛 ``MemoryError``，
        调用方（serving 引擎）可以把请求标记为 OOM 或触发 eviction。
        """
        remaining = n_tokens
        while remaining > 0:
            # 当前尾页若 '满' 或还没页 => 再拆一个新页（alloc 会从 free 弹 id）
            if not self.mla_pages or self.pool.pages[self.mla_pages[-1]].is_full(
                self.tokens_per_page
            ):
                self.mla_pages.append(self.pool.allocate(PageKind.MLA_KV).ident)
            page = self.pool.pages[self.mla_pages[-1]]
            # 本页还能填多少就填多少（不能超 tokens_per_page）
            take = min(remaining, self.tokens_per_page - page.filled)
            # 真实 KV 写入语义不在 Page 层，``filled`` 记录「已占用容量」便够了；
            page.filled += take
            remaining -= take
        self.length += n_tokens

    def store_kda(self, layer: int, state: Tensor) -> None:
        """把某层 KDA 的 ``(B,H,dk,dv)`` 状态写入该层的那一页。

        语义：每个 layer 最多一页 → 页 id 缓存在 ``kda_pages[layer]``。
        第二次写入同一层会复用同一页（覆盖写），不会重新 alloc。
        """
        if layer not in self.kda_pages:
            self.kda_pages[layer] = self.pool.allocate(PageKind.KDA_STATE).ident
        page = self.pool.pages[self.kda_pages[layer]]
        # pack 成 flat 后直接拷贝进页的 data（data 是整页宽，页比状态大则余 pad 零）
        flat = pack_heads(state)
        page.data[: flat.numel()] = flat
        page.filled = flat.numel()   # 记录写入的元素数，load 时据此切片

    def load_kda(self, layer: int, heads: int, dk: int, dv: int) -> Tensor:
        """读回该层 KDA 状态，还原成 ``(B,H,dk,dv)``。

        ``filled`` 记录真实数据长度，读回时只取 ``[:n]`` 去掉 pad，
        再经 ``unpack_heads`` 还原。``n`` 的取值规则：
        优先用 ``filled``（正常存储路径都写它）；若为 0（旧路径/未写），
        回退为整页宽——保证两种情况下都能还原出正确形状。
        """
        page = self.pool.pages[self.kda_pages[layer]]
        n = page.filled if page.filled > 0 else page.data.numel()
        return unpack_heads(page.data[:n], heads, dk, dv)

    def release(self) -> None:
        """请求结束：把本请求占用的所有页归还池。

        遍历两类页 id，逐个 ``pool.release``，
        然后清空自己的引用与 ``length`` —— 下次 fill 成一个完全新的 view。
        """
        for ident in self.mla_pages:
            self.pool.release(ident)
        for ident in self.kda_pages.values():
            self.pool.release(ident)
        self.mla_pages.clear()
        self.kda_pages.clear()
        self.length = 0