import torch

from models.llm.kimi_k3.serve.paged_cache import PagePool, PagedCache
from models.llm.kimi_k3.serve.prefix_cache import PrefixCache
from models.llm.kimi_k3.serve.engine import run_once


def test_second_request_reuses_shared_prefix() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=16)
    cache = PrefixCache()
    # 首请求 512 token 独占 prefill
    tokens = list(range(512))
    cache.register(tokens)
    cache.persist_kda_checkpoint(512)

    # 第二请求同一前缀
    hit = cache.lookup(tokens)
    assert hit.hit and hit.matched_tokens == 512
    assert cache.recompute_saved(hit, 512) == 512


def test_engine_returns_hit_stats() -> None:
    stats = run_once(hash_present=True)
    assert stats["reused_tokens"] == 512
    assert stats["pages"] >= 1
    assert stats["comm_bytes"] > 0


def test_engine_miss_returns_zero() -> None:
    stats = run_once(hash_present=False)
    assert stats["reused_tokens"] == 0
    assert stats["pages"] >= 1