"""Fleet 层调度模拟 —— K3 技术报告 §5.4.3（KDA-Infra.md §7）。

设计动机（为什么需要 fleet 层）
--------------------------------
单实例 serving（前 5 个任务：页池 / 前缀缓存 / 投机回滚 / KCP / 引擎）
解决的是「单个实例内的请求效率」；Fleet 层把视角提升到「一组实例的
可预期性」。报告 §5.4.3 给出两个 fleet 级策略：

1. **cache-aware affinity scheduling**：1M 上下文下典型编码输入的 prefix
   400K、prefill increment 只有 4K，命中一次避免重算整个前缀，比 miss
   便宜几个数量级。因此把 request 路由到**持有其 prefix cache 的集群**。
   但亲和性会把 session 绑死到单集群，一旦该集群故障会中断所有绑定的
   session。一致哈希于是把每个 session 绑定 **primary + secondary** 两个
   集群：primary 正常服务，secondary 在主集群失效时接管并重新 prefill；
   各 session 的 secondary 由一致哈希均匀分散，re-fill 工作量被分散，
   而非集中一处。

2. **budget-based admission control**：生产流量混着短请求（<2K token）与
   超长请求（到 1M token），单请求长度横跨 3 个数量级。为不同请求类分配
   独立资源预算，突发长 ctx 只消耗自己那份 capacity，不拖累其他类的 TTFT。

本模块用 CPU 可跑的事件驱动仿真落地这两个策略：
- 一致哈希环（``ConsistentHashRing``）实现 primary/secondary 双绑定与
  集群增减的最小扰动（hopscotch 性质）；
- ``ClusterInstance`` 复用现有 ``PrefixCache``（命中判定真实）+ ``PagePool``，
  代价公式折算成时间；
- ``BudgetController`` 实现请求类预算隔离；
- ``FleetSimulator`` 按到达时间推进事件循环，产出命中率 / 代价 / P95 TTFT /
  负载均衡 / failover 等汇总指标。

纯 CPU、纯标准库依赖（md5 + bisect），与 `prefix_cache.py` 同级独立。
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass

from models.llm.kimi_k3.serve.paged_cache import PagePool
from models.llm.kimi_k3.serve.prefix_cache import PrefixCache

__all__ = [
    "ConsistentHashRing",
    "ClusterInstance",
    "BudgetController",
    "Request",
    "FleetSimulator",
]


# ---------------------------------------------------------------- hash ring

class ConsistentHashRing:
    """一致哈希环：把 session 稳定映射到 (primary, secondary) 两个集群。

    为什么需要虚拟节点：若直接对 cluster_id 取模，少数集群会造成严重
    不均衡（有的集群分到大半 session，有的近乎空）。给每个集群散布
    ``vnodes`` 个虚拟节点（``cluster_id:i`` 的 md5 整数位），统计上
    各集群接管比例趋于均匀，同时保证「移除/新增一个集群只扰动
    O(1/N) 的 session」—— 报告 §5.4.3 正是依赖这一性质，让 failover
    后 re-prefill 的工作量被均匀分散到全 fleet，而不是集中在一处。

    实现：``_ring`` 为按位置升序的 ``[(pos, cluster_id)]``；``route``
    用 session 的 md5 做 ``bisect`` 顺时针落入 primary，再顺移取下一个
    不同集群作为 secondary —— primary != secondary 是硬约束（仅当环上
    只有一个集群时才允许退化为同一集群）。
    """

    def __init__(self) -> None:
        self._ring: list[tuple[int, str]] = []

    @staticmethod
    def _hash(key: str) -> int:
        return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)

    def add_cluster(self, cluster_id: str, vnodes: int = 8) -> None:
        """把一个集群散布 ``vnodes`` 个虚拟节点到环上。"""
        for i in range(vnodes):
            pos = self._hash(f"{cluster_id}:{i}")
            bisect.insort(self._ring, (pos, cluster_id))

    def remove_cluster(self, cluster_id: str) -> None:
        """摘除该集群的全部虚拟节点 —— 只有指向它的 session 需要改路由。"""
        self._ring = [(p, c) for (p, c) in self._ring if c != cluster_id]

    def clusters(self) -> list[str]:
        return list(dict.fromkeys(c for _, c in self._ring))

    def route(self, session_id: str) -> tuple[str, str]:
        """返回该 session 的 (primary, secondary)。同一 session 恒定。"""
        if not self._ring:
            raise ValueError("ring is empty: add clusters first")
        key = self._hash(session_id)
        positions = [p for p, _ in self._ring]
        idx = bisect.bisect_right(positions, key) % len(self._ring)
        primary = self._ring[idx][1]
        for off in range(1, len(self._ring) + 1):
            cid = self._ring[(idx + off) % len(self._ring)][1]
            if cid != primary:
                return primary, cid
        return primary, primary  # 仅当环上只有一个集群


# ------------------------------------------------------------- request

@dataclass
class Request:
    """一条（模拟的）请求：绑定一个 session、携带前缀与长度信息。

    - ``prefix`` 是 session 的共享前缀 token 列表，用于集群内真实命中判定；
    - ``prompt_len`` 是这个请求的总长度（含前缀），决定 prefill 计算量；
    - ``arrive_time`` 是仿真时钟的到达时间，决定事件顺序。
    """
    session_id: str
    request_class: str       # "short"（<2K）或 "long"（可达 1M）
    prefix: list[int]        # session 的前缀 token
    prompt_len: int          # 请求总 token 数
    arrive_time: float = 0.0


# ------------------------------------------------------------- cluster

class ClusterInstance:
    """模拟一个 serving 集群（实例）：有自己的一整套 serving 结构。

    关键点：命中判定**不是**拍一个概率参数，而是委托现有 ``PrefixCache``
    的 lookup / recompute_saved —— 因此这里的命中率与 serve 层真实模块
    一致（chained-hash 匹配 + KDA checkpoint 原子性），这是本仿真
    「可信」的来源：代价数字不是凭空编的，而是真实缓存行为折算出来的。

    代价模型：``work / throughput``。命中时只重算 ``prompt_len - reused``
    的增量；未命中则整段重新 prefill。当 reused 接近 prompt_len 时，
    miss/hit 的代价比是数量级 —— 正是报告 §5.4.3 说的"orders of
    magnitude"。
    """

    def __init__(self, cluster_id: str, throughput: float = 1.0,
                 page_bytes: int = 16 * 16 * 16 * 8, n_pages: int = 32) -> None:
        self.cluster_id = cluster_id
        self.throughput = throughput
        self.alive = True
        self.cache = PrefixCache()
        self.pool = PagePool(page_bytes=page_bytes, n_pages=n_pages)
        self.hits = 0
        self.misses = 0

    def register(self, tokens: list[int]) -> None:
        """登记前缀（含 KDA checkpoint）—— 之后同前缀请求才能真实命中。"""
        self.cache.register(tokens)
        self.cache.persist_kda_checkpoint(len(tokens))

    def reused_tokens(self, tokens: list[int], total: int) -> int:
        """真实命中：PrefixCache.lookup + recompute_saved。"""
        res = self.cache.lookup(tokens)
        return self.cache.recompute_saved(res, total)

    def serve_cost(self, prompt_len: int, reused_tokens: int) -> float:
        """命中只重算增量，未命中整段 —— 这就是命中便宜几个数量级。"""
        work = max(0, prompt_len - reused_tokens)
        return work / self.throughput


# ------------------------------------------------------------- budget

class BudgetController:
    """预算准入控制：给不同请求类分配独立配额，互不挤占。

    报告动机：短请求 <2K 与超长 1M 混流，长度横跨 3 个数量级——若总容量
    固定且没有隔离，长请求突发会把短请求的 TTFT 全线拖垮。给每类独立
    预算后：长突发只能吃掉自己的配额，短请求配额不受影响。

    权重：每类请求占用不同份容量（短=1、长=3），预算数字是「容量单位」
    而非「请求条数」，更贴近工程里"一份容量"的心智。
    """

    def __init__(self, short_budget: int, long_budget: int,
                 short_weight: int = 1, long_weight: int = 3) -> None:
        self.budgets = {"short": short_budget, "long": long_budget}
        self.weights = {"short": short_weight, "long": long_weight}

    def admit(self, request_class: str, usage: dict[str, int]) -> bool:
        """在当前用量下，该类再加一个请求（占 weight 份）是否在预算内。"""
        w = self.weights.get(request_class)
        b = self.budgets.get(request_class)
        if w is None or b is None:
            raise KeyError(request_class)
        return usage.get(request_class, 0) + w <= b


# ------------------------------------------------------------- simulator

class FleetSimulator:
    """顶层事件驱动仿真：ring + 集群 + 预算 + 请求流 → 统计指标。

    时间模型（CPU 无法真跑 1M token，用公式折算）：
    - prefill 代价 = ``serve_cost``；TTFT = 集群当前累计负载 / throughput
      （队列近似）+ 本次 prefill 代价。
    - 路由：session 绑定的 primary 存活走 primary；失效则走预分配的
      secondary 并累加一个 re-prefill 成本（该 secondary 需要重算此前
      整个绑定前缀）。

    run() 产出每请求明细；results() 汇总命中率 / 总代价 / 分请求类 P95
    TTFT / 拒绝数 / failover / 负载标准差。``stats`` 在每次 run 后刷新
    为 results() 的同一份字典，方便直接读取。
    """

    def __init__(self, n_clusters: int = 8, short_budget: int = 8,
                 long_budget: int = 8, throughput: float = 1.0) -> None:
        self.ring = ConsistentHashRing()
        self.clusters: dict[str, ClusterInstance] = {}
        for i in range(n_clusters):
            cid = f"c{i}"
            self.ring.add_cluster(cid)
            self.clusters[cid] = ClusterInstance(cid, throughput=throughput)
        self.budget = BudgetController(
            short_budget=short_budget, long_budget=long_budget)
        # 运行期指标
        self._cluster_load: dict[str, float] = {cid: 0.0 for cid in self.clusters}
        self._session_primary: dict[str, str] = {}   # session 当前绑定的 primary
        self._class_ttft: dict[str, list[float]] = {"short": [], "long": []}
        self._rejected: dict[str, int] = {"short": 0, "long": 0}
        self._total_cost = 0.0
        self._failover = 0
        self._re_prefill_cost = 0.0
        self.n_served = 0
        self.n_hits = 0
        self.stats: dict = {}

    def add_cluster(self, cluster_id: str, throughput: float = 1.0) -> None:
        self.ring.add_cluster(cluster_id)
        self.clusters[cluster_id] = ClusterInstance(cluster_id, throughput=throughput)
        self._cluster_load[cluster_id] = 0.0

    def fail_cluster(self, cluster_id: str) -> None:
        """令某集群失效（摘出环 + ``alive=False``）。"""
        if cluster_id not in self.clusters:
            raise KeyError(cluster_id)
        self.ring.remove_cluster(cluster_id)
        self.clusters[cluster_id].alive = False

    def _usage(self) -> dict[str, int]:
        # 预算用量近似为「该类已服务请求数」（容量占用 = 请求数 × weight，
        # 而 weight 在 BudgetController 里折算）。
        return {"short": len(self._class_ttft["short"]),
                "long": len(self._class_ttft["long"])}

    def run(self, requests: list[Request]) -> list[dict]:
        """按到达时间执行一批请求，返回每次服务的明细 dict 列表。

        事件循环每一步：
          1. 预算准入（类内隔离 → 长突发不挤占短类）；
          2. 路由：session 的 primary 失效则走 secondary（failover）；
          3. 登记前缀 + 真实命中判定（复用现有 PrefixCache）；
          4. prefill 代价 + TTFT（等待 = 集群累计负载 / 吞吐）；
          5. 更新统计。
        """
        results: list[dict] = []
        for req in sorted(requests, key=lambda r: r.arrive_time):
            # 1) 预算准入
            if not self.budget.admit(req.request_class, self._usage()):
                self._rejected[req.request_class] += 1
                continue
            # 2) 路由 + failover 检测
            primary, secondary = self.ring.route(req.session_id)
            target = primary
            prev = self._session_primary.get(req.session_id)
            if prev is not None and not self.clusters[prev].alive:
                # 该 session 绑定的 primary 已失效 → 走预分配 secondary
                target = secondary if self.clusters[secondary].alive else primary
                self._failover += 1
                self._re_prefill_cost += req.prompt_len / self.clusters[target].throughput
            self._session_primary[req.session_id] = target
            cluster = self.clusters[target]
            # 3) 真实命中（先查后登记：前缀只在"这次请求之后"才对本 session
            #    的未来请求可见 —— 首次请求必然 miss，第二次才命中，这正是
            #    "命中一次避免重算整个前缀"的语义来源）
            reused = cluster.reused_tokens(req.prefix, req.prompt_len)
            cost = cluster.serve_cost(req.prompt_len, reused)
            # 服务完毕再登记前缀，让后续同前缀请求能命中
            cluster.register(req.prefix)
            # 4) TTFT = 队列等待（集群累计负载）+ 本次 prefill
            wait = self._cluster_load[target] / cluster.throughput
            ttft = wait + cost
            # 5) 更新统计
            self._cluster_load[target] += cost
            self._class_ttft[req.request_class].append(ttft)
            self._total_cost += cost
            self.n_served += 1
            if reused > 0:
                self.n_hits += 1
                cluster.hits += 1
            else:
                cluster.misses += 1
            results.append({
                "session_id": req.session_id,
                "reused_tokens": reused,
                "ttft": ttft,
                "cluster": target,
            })
        self.stats = self.results()
        return results

    # -- 指标 -------------------------------------------------------------

    def hit_rate(self) -> float:
        return self.n_hits / max(1, self.n_served)

    def _load_std(self) -> float:
        loads = list(self._cluster_load.values())
        if not loads:
            return 0.0
        mean = sum(loads) / len(loads)
        var = sum((x - mean) ** 2 for x in loads) / len(loads)
        return var ** 0.5

    def _p95(self, request_class: str) -> float:
        vals = sorted(self._class_ttft[request_class])
        if not vals:
            return 0.0
        return vals[int(0.95 * len(vals)) - 1]

    def results(self) -> dict:
        """汇总指标（run 后自动写入 ``self.stats``）。"""
        return {
            "hit_rate": self.hit_rate(),
            "total_cost": self._total_cost,
            "avg_ttft": (sum(t for ts in self._class_ttft.values() for t in ts)
                         / max(1, self.n_served)),
            "p95_ttft_by_class": {
                cls: self._p95(cls) for cls in ("short", "long")
            },
            "rejected_by_class": dict(self._rejected),
            "failover_count": self._failover,
            "re_prefill_cost": self._re_prefill_cost,
            "load_std": self._load_std(),
        }
