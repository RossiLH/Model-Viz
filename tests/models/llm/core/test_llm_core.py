import torch
from models.llm.core.interfaces import AttnModule, MoEModule, ResidualModule, Block
from models.llm.core.config import LLMConfig, ModelConfig
from models.llm.core.norms import RMSNorm, HeadRMSNorm, l2_normalize


def test_llm_config():
    cfg = LLMConfig(name="test", model=ModelConfig(dim=128, n_layers=4))
    assert cfg.model.dim == 128
    assert cfg.model.n_layers == 4


def test_rms_norm():
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.allclose(y.pow(2).mean(-1).sqrt().mean(), torch.tensor(1.0), atol=0.02)


def test_head_rms_norm():
    norm = HeadRMSNorm(4, 16)
    x = torch.randn(2, 4, 16)
    y = norm(x)
    assert y.shape == x.shape


def test_l2_normalize():
    x = torch.randn(2, 8, 32)
    y = l2_normalize(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_interfaces_abstract():
    import pytest
    with pytest.raises(TypeError):
        AttnModule()
    with pytest.raises(TypeError):
        MoEModule()
    with pytest.raises(TypeError):
        ResidualModule()
    with pytest.raises(TypeError):
        Block()