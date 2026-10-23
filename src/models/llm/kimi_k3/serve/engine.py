"""收官引擎：多请求 under shared prefix 的 before/after 汇总。

设计动机（为什么要有这个模块）
-----------------------------
前 4 个任务分别实现了独立子系统（paged_cache / prefix_cache / decode / kcp），
本模块是**收官汇总层**：把「统一页池 + 前缀缓存 + KDA 上下文并行」串起来，
回答工程上一个很朴素的问题——**第二个请求复用了同一个前缀时，到底省了什么**：

* **before（无前缀缓存）**：第二个请求从头 re-compute——MLA 侧重新 prefill
  全部 512 token（再占一页），KDA 侧重新走完整递推（重算 checkpoint）。
* **after（有前缀缓存）**：MLA 的 KV 页逐块命中直接复用，KDA 从 checkpoint
  快照处恢复 O(1) 状态，免去全部重算。省下的正是 ``recompute_saved`` 报告的
  token 数（本例 512 vs 0）。

第 1 请求：prefill → PagedCache 存 KDA page → register + persist checkpoint；
第 2 请求：prefix lookup 命中 → 复用 MLA page + KDA checkpoint；否则全重算。
``run_once`` 是可导入的纯函数（无 IO、无随机），实验脚本与集成测试共用——
测试替实验保证数值契约，实验替文档给出可视化证据。
"""

from __future__ import annotations

import torch

from models.llm.kimi_k3.parallel.kcp import KDAContextParallel
from models.llm.kimi_k3.serve.paged_cache import PagePool, PagedCache
from models.llm.kimi_k3.serve.prefix_cache import PrefixCache


def run_once(hash_present: bool = True) -> dict:
    """模拟「两请求共享/不共享前缀」各跑一遍，返回复用统计的纯函数。

    Args:
        hash_present: ``True`` 表示第二请求的前缀与首请求完全一致（命中）；
                      ``False`` 表示换了一段全新前缀（chaired hash 必然 miss）。

    Returns:
        ``dict``，含四项指标：
            * ``reused_tokens`` —— 第二请求免重算的 token 数（512 vs 0，本实验核心）。
            * ``pages``          —— 首请求在统一页池里实际占用的页数
                                    （MLA 页 + KDA 页，验证"一请求一份状态"）。
            * ``comm_bytes``     —— KCP 每层跨 8 个 rank 交换 (M, Stilde) 的
                                    总字节数，与序列长度无关（KDA 稳定性佐证）。
            * ``pool_utilization``—— 页池利用率，展示统一页池的健康度读数。

    为什么用固定的 3 head / 16x16 / 8 rank：与前面 4 个任务的实验配置保持一致，
    保证跨任务数值可横向对比；float64 与 kda_ref / kcp 全部默认精度一致。
    """
    # 统一页池：页宽 = 一层 KDA 状态的字节（3 head * 16 * 16 * float64）。
    # n_pages=32 足够容纳首请求的 MLA 页 + KDA 页，还能留出利用率余量。
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=32)
    # 首请求的请求级视图：prefill 512 token 占满一页 MLA + 单层 KDA 一页。
    view1 = PagedCache(pool, tokens_per_page=512)
    view1.append_mla(512)                       # 首请求 prefill 512 token
    view1.store_kda(0, torch.zeros(1, 3, 16, 16, dtype=torch.float64))  # 层0 KDA 状态

    # 前缀缓存：首请求结束，登记前缀并把每 KDA group 的 checkpoint 落盘。
    cache = PrefixCache()
    base = list(range(512))
    cache.register(base)
    cache.persist_kda_checkpoint(512)

    # 第二请求：命中分支用同一前缀（能复用 512），miss 分支换成全新前缀（复用 0）。
    if hash_present:
        tokens2 = base                     # 第二请求相同前缀 -> 命中
    else:
        tokens2 = list(range(512, 1024))   # 第二请求不同前缀 -> miss

    # lookup 给出双粒度命中（MLA 块 + KDA checkpoint），recompute_saved 折算成
    # 「本次真正免于重算的 token 数」——这就是 before/after 里 after 的收益。
    hit = cache.lookup(tokens2)
    reused = cache.recompute_saved(hit, 512)
    return {
        "reused_tokens": reused,
        "pages": len(view1.mla_pages) + len(view1.kda_pages),
        "comm_bytes": KDAContextParallel.communication_volume(
            batch=1, heads=3, dk=16, dv=16, world=8
        ),
        "pool_utilization": pool.utilization(),
    }