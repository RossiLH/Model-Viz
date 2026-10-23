import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import math
import pytest
import torch
from models.llm.kimi_k3.config import (
    Kimi3Config, Kimi3ModelConfig, KDAConfig, MoEConfig, AttnResConfig,
    ConfigError,
)
from models.llm.kimi_k3.model import Kimi3Block, Kimi3Model, Kimi3ForCausalLM, doc_masks


def _tiny_cfg(n_layers=2, n_dense=1, attn_res_enabled=False):
    return Kimi3Config(model=Kimi3ModelConfig(
        dim=64, n_layers=n_layers, n_heads=4, vocab_size=100, n_dense_layers=n_dense,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                       conv_kernel=3, chunk_size=8, tile_size=8, g_min=-5.0,
                       alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8, expert_hidden=16,
                       shared_hidden=32, situ_beta_gate=1.0, situ_beta_up=1.0,
                       rmsnorm_before_up=True, qb_enabled=False, qb_bins=100, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=attn_res_enabled),
    ))


# --- Config tests -----------------------------------------------------------

def test_config_validate_accepts_valid():
    cfg = _tiny_cfg()
    cfg.validate()


def test_config_rejects_negative_dim():
    cfg = Kimi3Config(model=Kimi3ModelConfig(dim=-1))
    with pytest.raises(ConfigError, match="dim.*positive"):
        cfg.validate()


def test_config_rejects_topk_above_expert_count():
    import pytest
    cfg = Kimi3Config(model=Kimi3ModelConfig(moe=MoEConfig(n_routed=8, top_k=16)))
    with pytest.raises(ConfigError, match="top_k"):
        cfg.validate()


def test_config_rejects_alpha_init_below_decay_floor():
    cfg = Kimi3Config(model=Kimi3ModelConfig(kda=KDAConfig(alpha_init=math.exp(-6))))
    with pytest.raises(ConfigError, match="decay floor"):
        cfg.validate()


def test_config_rejects_misaligned_chunk_tile():
    cfg = Kimi3Config(model=Kimi3ModelConfig(kda=KDAConfig(chunk_size=48, tile_size=32)))
    with pytest.raises(ConfigError, match="multiple of tile_size"):
        cfg.validate()


def test_config_rejects_bad_alpha_init():
    cfg = Kimi3Config(model=Kimi3ModelConfig(kda=KDAConfig(alpha_init=0.0)))
    with pytest.raises(ConfigError, match="alpha_init"):
        cfg.validate()


def test_config_rejects_negative_g_min():
    import pytest
    cfg = Kimi3Config(model=Kimi3ModelConfig(kda=KDAConfig(g_min=1.0)))
    with pytest.raises(ConfigError, match="g_min"):
        cfg.validate()


def test_config_layer_schedule():
    cfg = Kimi3Config(model=Kimi3ModelConfig(n_layers=9))
    sched = cfg.model.layer_schedule()
    assert len(sched) == 9
    assert sched[:4] == ["kda", "kda", "kda", "mla"]
    assert sched[-1] == "mla"


def test_config_n_kda_layers():
    cfg = Kimi3Config(model=Kimi3ModelConfig(n_layers=9))
    assert cfg.model.n_kda_layers == 6
    assert cfg.model.n_mla_layers == 3


def test_config_n_moe_layers():
    cfg = Kimi3Config(model=Kimi3ModelConfig(n_layers=10, n_dense_layers=1))
    assert cfg.model.n_moe_layers == 9


def test_config_n_attnres_blocks():
    cfg = Kimi3Config(model=Kimi3ModelConfig(n_layers=10, attn_res=AttnResConfig(block_size=3)))
    assert cfg.model.n_attnres_blocks == 4


def test_config_n_sublayers():
    cfg = Kimi3Config(model=Kimi3ModelConfig(n_layers=10))
    assert cfg.model.n_sublayers == 20


# --- Model tests -------------------------------------------------------------

def test_kimi3_config():
    cfg = _tiny_cfg()
    assert cfg.model.kda.num_heads == 4
    assert cfg.model.moe.n_routed == 4


def test_kimi3_block():
    cfg = _tiny_cfg()
    block = Kimi3Block(cfg, layer_idx=1, use_kda=True)
    x = torch.randn(2, 4, 64)
    y = block(x)
    assert y.shape == (2, 4, 64)


def test_kimi3_model():
    cfg = _tiny_cfg(attn_res_enabled=False)
    model = Kimi3Model(cfg)
    tokens = torch.randint(0, 100, (2, 8))
    out = model(tokens)
    assert out.shape == (2, 8, 64)


def test_kimi3_causal_lm():
    cfg = _tiny_cfg(attn_res_enabled=False)
    model = Kimi3ForCausalLM(cfg)
    tokens = torch.randint(0, 100, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 100)


def test_doc_masks():
    doc_ids = torch.tensor([[0, 0, 1, 1, 1]])
    reset, attn = doc_masks(doc_ids, 5, torch.device("cpu"))
    assert reset is not None
    assert reset.tolist() == [[True, False, True, False, False]]


def test_no_positional_state_anywhere():
    cfg = _tiny_cfg(attn_res_enabled=False)
    model = Kimi3ForCausalLM(cfg)
    names = {n.lower() for n, _ in model.named_parameters()}
    for banned in ("rope", "inv_freq", "position_emb", "alibi"):
        assert not any(banned in n for n in names), f"found {banned}"


def test_generation_with_cache_matches_full_forward():
    cfg = _tiny_cfg(attn_res_enabled=False)
    torch.manual_seed(0)
    model = Kimi3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        full = model(ids)
        caches = model.init_caches(1)
        steps = [model(ids[:, i:i+1], caches=caches) for i in range(8)]
    incremental = torch.cat(steps, dim=1)
    assert torch.allclose(incremental, full, atol=2e-3, rtol=2e-3)


def test_generate_produces_new_tokens():
    cfg = _tiny_cfg(attn_res_enabled=False)
    torch.manual_seed(0)
    model = Kimi3ForCausalLM(cfg).eval()
    prompt = torch.randint(0, 100, (1, 8))
    out = model.generate(prompt, max_new_tokens=5, greedy=True)
    assert out.shape == (1, 13)
    assert torch.equal(out[:, :8], prompt)


def test_every_parameter_receives_gradient():
    cfg = _tiny_cfg(attn_res_enabled=False)
    model = Kimi3ForCausalLM(cfg)
    model.train()
    ids = torch.randint(0, 100, (1, 8))
    logits = model(ids)
    logits.square().mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient: {missing}"


def test_seeded_runs_are_deterministic():
    cfg = _tiny_cfg(attn_res_enabled=False)
    outs = []
    for _ in range(2):
        torch.manual_seed(1234)
        model = Kimi3ForCausalLM(cfg).eval()
        ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            outs.append(model(ids))
    assert torch.equal(outs[0], outs[1])


def test_init_caches_returns_correct_length():
    cfg = _tiny_cfg(n_layers=3, attn_res_enabled=False)
    model = Kimi3ForCausalLM(cfg)
    caches = model.init_caches(2)
    assert len(caches) == 3


def test_model_with_attn_res():
    cfg = _tiny_cfg(n_layers=2, attn_res_enabled=True)
    model = Kimi3Model(cfg)
    tokens = torch.randint(0, 100, (2, 4))
    out = model(tokens)
    assert out.shape == (2, 4, 64)