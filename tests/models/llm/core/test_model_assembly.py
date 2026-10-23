import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
import torch
from models.llm.core.config import LLMConfig, ModelConfig
from models.llm.core.model import SimpleBlock, GenericLLM
from models.llm.core.standard.mha import StandardMHA
from models.llm.core.standard.moe import StandardMoE
from models.llm.core.standard.residual import StandardResidual
from models.llm.core.sequence import sample_top_k, sample_top_p


def test_simple_block():
    dim = 64
    attn = StandardMHA(dim, n_heads=4, n_kv_heads=2)
    ffn = torch.nn.Linear(dim, dim)
    block = SimpleBlock(attn, ffn, dim)
    x = torch.randn(2, 4, dim)
    y = block(x)
    assert y.shape == (2, 4, dim)


def test_generic_llm():
    cfg = LLMConfig(model=ModelConfig(dim=64, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=100))
    blocks = []
    for _ in range(cfg.model.n_layers):
        attn = StandardMHA(cfg.model.dim, cfg.model.n_heads, cfg.model.n_kv_heads)
        ffn = torch.nn.Linear(cfg.model.dim, cfg.model.dim)
        blocks.append(SimpleBlock(attn, ffn, cfg.model.dim))
    model = GenericLLM(cfg, blocks)
    tokens = torch.randint(0, 100, (2, 8))
    out = model(tokens)
    assert out.shape == (2, 8, 64)


def test_sample_top_k():
    logits = torch.randn(2, 4, 100)
    probs = sample_top_k(logits, 10)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(-1), torch.ones(2, 4))


def test_sample_top_p():
    logits = torch.randn(2, 100)
    probs = sample_top_p(logits, 0.95)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(-1), torch.ones(2))