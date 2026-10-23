import torch
import pytest

from models.llm.kimi_k3.serve.paged_cache import PageKind, PagePool, PagedCache, pack_heads, unpack_heads
from models.llm.kimi_k3.serve.test_helpers import make_state


def test_single_free_list_shared_across_kinds() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=4)
    kda = pool.allocate(PageKind.KDA_STATE)
    mla = pool.allocate(PageKind.MLA_KV)
    assert kda.kind == PageKind.KDA_STATE
    assert mla.kind == PageKind.MLA_KV
    assert pool.utilization() == 0.5
    pool.release(kda.ident)
    assert kda.ident in pool.free


def test_store_and_load_roundtrip() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=4)
    view = PagedCache(pool)
    st = make_state()
    view.store_kda(0, st)
    got = view.load_kda(0, heads=3, dk=16, dv=16)
    assert torch.allclose(got, st)


def test_append_mla_increments_pages_across_boundary() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=16)
    view = PagedCache(pool, tokens_per_page=512)
    view.append_mla(600)
    assert len(view.mla_pages) == 2
    assert view.length == 600


def test_pack_unpack_roundtrip() -> None:
    state = make_state(2, 3, 16, 16)
    flat = pack_heads(state)
    assert flat.numel() == 2 * 3 * 16 * 16
    got = unpack_heads(flat, heads=3, dk=16, dv=16)
    assert torch.allclose(got, state)


def test_pool_exhaustion_raises() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=2)
    pool.allocate(PageKind.KDA_STATE)
    pool.allocate(PageKind.KDA_STATE)
    with pytest.raises(MemoryError):
        pool.allocate(PageKind.KDA_STATE)


def test_release_returns_page_to_free() -> None:
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=3)
    p = pool.allocate(PageKind.MLA_KV)
    pool.release(p.ident)
    assert p.ident in pool.free
