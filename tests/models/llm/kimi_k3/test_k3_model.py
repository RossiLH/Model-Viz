import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
from models.llm.kimi_k3.config import Kimi3Config, Kimi3ModelConfig, KDAConfig, MoEConfig, AttnResConfig
from models.llm.kimi_k3.model import Kimi3Block, Kimi3Model, Kimi3ForCausalLM


def test_kimi3_config():
    cfg = Kimi3Config(model=Kimi3ModelConfig(dim=64, n_layers=2, n_heads=4, vocab_size=100))
    assert cfg.model.kda.num_heads == 96
    assert cfg.model.moe.n_routed == 896


def test_kimi3_block():
    cfg = Kimi3Config(model=Kimi3ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=100,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8, conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0, alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8, expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0, situ_beta_up=1.0, rmsnorm_before_up=True, qb_enabled=False, qb_bins=10, qb_ema=0.9),
        n_dense_layers=1,
    ))
    block = Kimi3Block(cfg, layer_idx=1, use_kda=True)
    x = torch.randn(2, 4, 64)
    y = block(x)
    assert y.shape == (2, 4, 64)


def test_kimi3_model():
    cfg = Kimi3Config(model=Kimi3ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=100,
        n_dense_layers=1,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8, conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0, alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8, expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0, situ_beta_up=1.0, rmsnorm_before_up=True, qb_enabled=False, qb_bins=10, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=False),
    ))
    model = Kimi3Model(cfg)
    tokens = torch.randint(0, 100, (2, 8))
    out = model(tokens)
    assert out.shape == (2, 8, 64)


def test_kimi3_causal_lm():
    cfg = Kimi3Config(model=Kimi3ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=100,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8, conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0, alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8, expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0, situ_beta_up=1.0, rmsnorm_before_up=True, qb_enabled=False, qb_bins=10, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=False),
    ))
    model = Kimi3ForCausalLM(cfg)
    tokens = torch.randint(0, 100, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 100)