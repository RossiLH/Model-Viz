import pytest

from models.llm.kimi_k3.serve.prefix_cache import PrefixCache, chained_hash


def test_chained_hash_is_deterministic():
    assert chained_hash("", [1, 2, 3]) == chained_hash("", [1, 2, 3])
    assert chained_hash("", [1, 2, 3]) != chained_hash("", [1, 2, 4])


def test_register_then_lookup_at_same_prefix_hits():
    cache = PrefixCache()
    tokens = list(range(512))
    cache.register(tokens)
    cache.persist_kda_checkpoint(512)
    res = cache.lookup(tokens)
    assert res.hit
    assert res.matched_tokens == 512
    assert res.kda_checkpoint == 512


def test_lookup_without_checkpoint_misses():
    cache = PrefixCache()
    tokens = list(range(512))
    cache.register(tokens)
    res = cache.lookup(tokens)
    assert not res.hit
    assert res.matched_tokens == 0


def test_two_stage_uses_longest_checkpoint_boundary():
    cache = PrefixCache()
    tokens = list(range(1024))
    cache.register(tokens)
    cache.persist_kda_checkpoint(512)     # 只在这个边界存
    res = cache.lookup(tokens)
    assert res.matched_tokens == 512      # 不会超过没有 checkpoint 的 1024
    assert res.kda_checkpoint == 512


def test_checkpoint_atomic_across_groups():
    cache = PrefixCache(n_kda_groups=2)
    tokens = list(range(512))
    cache.register(tokens)
    cache.persist_kda_checkpoint(512)
    assert cache.checkpoint_available(512)
    # 原子性核心语义：仅一个 group 有 checkpoint 时，另一个缺失 → 视为不可用
    cache.evict_kda_checkpoint(512, groups=[0])
    assert not cache.checkpoint_available(512)
    cache.persist_kda_checkpoint(512, groups=[0])
    assert cache.checkpoint_available(512)
    cache.evict_kda_checkpoint(512)
    assert not cache.checkpoint_available(512)


def test_recompute_saved_caps_at_request_length():
    cache = PrefixCache()
    tokens = list(range(512))
    cache.register(tokens)
    cache.persist_kda_checkpoint(512)
    res = cache.lookup(tokens)
    assert cache.recompute_saved(res, total_tokens=512) == 512
    assert cache.recompute_saved(res, total_tokens=300) == 300