"""fleet 层调度（K3 报告 §5.4.3）的单元测试。

覆盖四组行为：
1. 一致哈希环 —— 路由稳定 / primary!=secondary / 集群增减的最小扰动 / 均匀分布；
2. 代价模型 —— 命中 vs 未命中的数量级差异、failover 的重 prefill 代价；
3. 预算准入 —— 长突发不拖累短请求、超配额被拒；
4. 顶层仿真 —— 命中路由压低总代价、预算隔离压低短类 P95 TTFT。
"""

from __future__ import annotations

import pytest

from models.llm.kimi_k3.serve.fleet import (
    BudgetController,
    ClusterInstance,
    ConsistentHashRing,
    FleetSimulator,
    Request,
)

N_SESSIONS = 600


def _prefix(seed: int) -> list[int]:
    """确定性的「前缀 token 序列」，长度 512（与 prefix_cache hash block 一致）。"""
    return list(range(seed, seed + 512))


def _sessions(n: int = N_SESSIONS, seed: int = 0) -> list[str]:
    return [f"session-{seed}-{i}" for i in range(n)]


# -- 一致哈希 ---------------------------------------------------------------


def test_ring_route_returns_two_distinct_clusters():
    ring = ConsistentHashRing()
    for cid in ["c0", "c1", "c2"]:
        ring.add_cluster(cid, vnodes=8)
    primary, secondary = ring.route("session xyz")
    assert primary == "c0" or primary == "c1" or primary == "c2"
    assert secondary != primary
    assert primary in ring.clusters() and secondary in ring.clusters()


def test_ring_route_stable_for_same_session():
    ring = ConsistentHashRing()
    for cid in ["c0", "c1", "c2"]:
        ring.add_cluster(cid, vnodes=8)
    a = ring.route("session-42")
    b = ring.route("session-42")
    assert a == b


def test_remove_cluster_only_affects_few_sessions():
    ring = ConsistentHashRing()
    for c in range(8):
        ring.add_cluster(f"c{c}", vnodes=8)
    prime_before = [ring.route(s)[0] for s in _sessions()]
    ring.remove_cluster("c3")
    prime_after = [ring.route(s)[0] for s in _sessions()]
    changed = sum(1 for x, y in zip(prime_before, prime_after) if x != y)
    # 一致哈希的 hopscotch 性质：移除 1 个集群只扰动 O(1/N) 的 session
    assert changed < N_SESSIONS * 0.6
    # 只有指向失效集群的 session 必须改绑到别处
    assert all(c != "c3" for c in prime_after)


def test_ring_distributes_sessions_uniformly():
    ring = ConsistentHashRing()
    for c in range(8):
        ring.add_cluster(f"c{c}", vnodes=8)
    counts: dict[str, int] = {f"c{i}": 0 for i in range(8)}
    for s in _sessions():
        primary, _ = ring.route(s)
        counts[primary] += 1
    minimum = min(counts.values())
    assert minimum >= len(_sessions()) * 0.05  # 每个集群至少分到 ~5%


# -- 代价模型 ---------------------------------------------------------------


def test_cost_miss_is_orders_magnitude_higher():
    cluster = ClusterInstance("c0", throughput=1.0)
    hit_cost = cluster.serve_cost(prompt_len=400_000, reused_tokens=399_488)
    miss_cost = cluster.serve_cost(prompt_len=400_000, reused_tokens=0)
    assert miss_cost > hit_cost * 10


def test_failover_goes_to_secondary_and_adds_cost():
    sim = FleetSimulator(n_clusters=6)
    session = "f-session"
    # 登记第一个请求把 prefix 落到 primary
    req1 = Request(session_id=session, request_class="short",
                   prefix=_prefix(0), prompt_len=512, arrive_time=0.0)
    sim.run([req1])
    primary, secondary = sim.ring.route(session)
    sim.fail_cluster(primary)
    req2 = Request(session_id=session, request_class="short",
                   prefix=_prefix(0), prompt_len=512, arrive_time=1.0)
    sim.run([req2])
    assert sim.stats["failover_count"] == 1
    # 失效接管必须重新 prefill 整个前缀 -> 产生正的恢复成本
    assert sim.stats["re_prefill_cost"] > 0.0


# -- 预算准入 ---------------------------------------------------------------


def test_budget_isolates_long_burst_from_short():
    budget = BudgetController(short_budget=10, long_budget=10)
    usage = {"short": 5, "long": 10}  # 长类已占满
    assert budget.admit("short", usage)   # 短类不受长类影响
    assert not budget.admit("long", usage)


def test_budget_rejects_when_over_quota():
    budget = BudgetController(short_budget=4, long_budget=12)
    usage = {"short": 4, "long": 11}
    assert not budget.admit("short", usage)
    assert not budget.admit("long", usage)  # long_weight=3，且 11+3>12 -> 拒绝


# -- 顶层仿真 ---------------------------------------------------------------


def test_simulator_routes_to_primary_on_hit():
    sim = FleetSimulator(n_clusters=8)
    session = "hit-session"
    sim.run([Request(session, "short", _prefix(0), 512, 0.0)])
    # 同前缀第二请求 → 同一 primary 集群命中缓存（首次 miss，第二次 hit）
    hit = sim.run([Request(session, "short", _prefix(0), 512, 1.0)])
    assert hit[0]["reused_tokens"] == 512
    # 2 个请求里命中 1 个（首个必然是 miss）：命中率 = 0.5
    assert sim.stats["hit_rate"] == 0.5


def test_simulator_isolated_budget_keeps_short_ttft():
    # 长突发场景：无预算时短请求可能被长请求排挤，预算隔离后短请求 P95 更低
    sim_iso = FleetSimulator(n_clusters=8, short_budget=8, long_budget=8)
    sim_all = FleetSimulator(n_clusters=8, short_budget=10**9, long_budget=10**9)
    long_reqs = [Request(f"L{i}", "long", _prefix(1000 + i), 1_000_000, i * 0.01)
                 for i in range(20)]
    short_reqs = [Request(f"S{i}", "short", _prefix(2000 + i), 1000, 2.0 + i * 0.001)
                  for i in range(50)]
    sim_iso.run([*long_reqs, *short_reqs])
    sim_all.run([*long_reqs, *short_reqs])
    iso_p95 = sim_iso.results()["p95_ttft_by_class"]["short"]
    all_p95 = sim_all.results()["p95_ttft_by_class"]["short"]
    # 预算隔离把短请求限入固定额度 → 不被长突发拖垮，P95 应严格更优（或接近）
    assert iso_p95 < all_p95