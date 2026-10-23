import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
from models.llm.core.standard.mha import StandardMHA
from models.llm.core.standard.moe import StandardMoE
from models.llm.core.standard.residual import StandardResidual


def test_standard_mha_shape():
    mha = StandardMHA(dim=128, n_heads=4, n_kv_heads=2)
    x = torch.randn(2, 8, 128)
    cache = mha.init_cache(2)
    y = mha(x, cache)
    assert y.shape == (2, 8, 128)


def test_standard_mha_cache_growth():
    mha = StandardMHA(dim=64, n_heads=4, n_kv_heads=2)
    cache = mha.init_cache(2)
    x1 = torch.randn(2, 4, 64)
    mha(x1, cache)
    assert cache["k"].shape[2] == 4
    x2 = torch.randn(2, 3, 64)
    mha(x2, cache)
    assert cache["k"].shape[2] == 7


def test_standard_moe_shape():
    moe = StandardMoE(dim=64, n_experts=4, top_k=2, hidden_dim=128)
    x = torch.randn(2, 4, 64)
    y = moe(x)
    assert y.shape == (2, 4, 64)


def test_standard_residual():
    res = StandardResidual()
    x = torch.randn(2, 4, 64)
    out = res(x, [x * 2])
    assert torch.allclose(out, x + x * 2)