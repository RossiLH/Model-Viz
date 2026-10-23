import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))
import torch
from models.llm.core.standard.mha import StandardMHA
from models.llm.kimi_k3.modules.kda import KimiDeltaAttention
from models.llm.kimi_k3.config import KDAConfig


def test_kda_vs_standard_output_shape():
    dim = 64
    standard = StandardMHA(dim=dim, n_heads=4, n_kv_heads=2)
    kda_cfg = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8)
    kda = KimiDeltaAttention(dim=dim, cfg=kda_cfg)

    x = torch.randn(2, 4, dim)
    cache = standard.init_cache(2)

    out_std = standard(x, cache)
    out_kda, _ = kda(x)

    assert out_std.shape == (2, 4, dim)
    assert out_kda.shape == (2, 4, dim)


def test_kv_cache_size_comparison():
    dim = 64
    standard = StandardMHA(dim=dim, n_heads=4, n_kv_heads=2)
    kda_cfg = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8)
    kda = KimiDeltaAttention(dim=dim, cfg=kda_cfg)

    batch = 2
    cache = standard.init_cache(batch)
    seq_lens = [16, 32, 64]
    for sl in seq_lens:
        x = torch.randn(batch, sl, dim)
        standard(x, cache)
        kv_params = cache["k"].numel() + cache["v"].numel()
        kda_params = 4 * 16 * 16
        assert kv_params > kda_params, f"At seq_len={sl}, standard KV cache should be larger than KDA fixed state"