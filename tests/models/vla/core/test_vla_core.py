import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))
import torch
from models.vla.core.config import VLAConfig, VisionEncoderConfig, ActionDecoderConfig
from models.vla.core.interfaces import VisionEncoder, ActionDecoder, VLAInference
from models.vla.core.norms import AdaRMSNorm


def test_vla_config():
    cfg = VLAConfig(vision=VisionEncoderConfig(dim=32), action=ActionDecoderConfig(width=32))
    assert cfg.vision.dim == 32
    assert cfg.action.width == 32


def test_adarms_norm():
    norm = AdaRMSNorm(32, cond_dim=64)
    x = torch.randn(2, 8, 32)
    cond = torch.randn(2, 64)
    y = norm(x, cond)
    assert y.shape == (2, 8, 32)


def test_adarms_norm_standard():
    norm = AdaRMSNorm(32)
    x = torch.randn(2, 8, 32)
    y = norm(x)
    assert y.shape == (2, 8, 32)


def test_adarms_norm_pooled_cond_is_token_blind():
    """默认（per_token_cond=False）对 3-D 条件做均值池化 → 逐 token 信息丢失

    这不是 bug，而是 π0/π0.5 的正确语义（一次去噪只有一个标量 timestep，
    序列里每个 token 的 cond 本来相同）。这里把该行为**钉死为基线**，
    以便与 π0.7 的 per_token_cond 路径做 before/after 对照。

    构造：把两个 token 的条件向量互换。均值不变 → 输出必须完全不变。
    """
    torch.manual_seed(0)
    norm = AdaRMSNorm(8, cond_dim=8, per_token_cond=False)
    torch.nn.init.normal_(norm.cond_proj.weight, std=0.5)  # 打破零初始化，让调制真实生效
    x = torch.randn(1, 4, 8)
    cond = torch.zeros(1, 4, 8)
    cond[0, 0] = 5.0
    cond[0, 3] = -5.0
    swapped = cond.clone()
    swapped[0, 0], swapped[0, 3] = cond[0, 3].clone(), cond[0, 0].clone()
    # 均值池化把两者压成同一个向量，故输出逐位相同
    assert torch.allclose(norm(x, cond), norm(x, swapped))


def test_adarms_norm_per_token_cond_is_position_sensitive():
    """per_token_cond=True 时逐 token 独立调制（训练期 RTC 的硬性前提）

    论文 2512.05964（训练期 RTC）改造第 1 步原话：
        "simply allow the scale, shift, and gate to differ between tokens.
         This does not change the number of learnable parameters."

    因此本测试同时断言两件事：
    1. 交换两个 token 的条件后输出**必须改变**（位置敏感）；
    2. 参数量与池化版**完全相同**（论文明确说不增参）。
    """
    torch.manual_seed(0)
    pooled = AdaRMSNorm(8, cond_dim=8, per_token_cond=False)
    per_tok = AdaRMSNorm(8, cond_dim=8, per_token_cond=True)
    torch.nn.init.normal_(per_tok.cond_proj.weight, std=0.5)

    x = torch.randn(1, 4, 8)
    cond = torch.zeros(1, 4, 8)
    cond[0, 0] = 5.0
    cond[0, 3] = -5.0
    swapped = cond.clone()
    swapped[0, 0], swapped[0, 3] = cond[0, 3].clone(), cond[0, 0].clone()
    assert not torch.allclose(per_tok(x, cond), per_tok(x, swapped))

    # 不增参：这是论文的明确断言，必须守住
    n_pooled = sum(p.numel() for p in pooled.parameters())
    n_per_tok = sum(p.numel() for p in per_tok.parameters())
    assert n_pooled == n_per_tok


def test_adarms_norm_per_token_cond_matches_pooled_on_uniform_cond():
    """条件在序列轴恒定时，两条路径必须数值等价（退化一致性）

    这保证 per_token_cond 是纯粹的**能力扩展**而非行为改写：
    π0/π0.5 那种"整段共享一个 timestep"的场景下，开关不影响结果。
    """
    torch.manual_seed(0)
    norm = AdaRMSNorm(8, cond_dim=8, per_token_cond=True)
    torch.nn.init.normal_(norm.cond_proj.weight, std=0.5)
    x = torch.randn(2, 5, 8)
    row = torch.randn(2, 8)
    out_2d = norm(x, row)                              # (B, cond_dim) 广播路径
    out_3d = norm(x, row.unsqueeze(1).expand(2, 5, 8))  # 序列轴恒定的 3-D 路径
    assert torch.allclose(out_2d, out_3d, atol=1e-6)


def test_interfaces_abstract():
    import pytest
    with pytest.raises(TypeError): VisionEncoder()
    with pytest.raises(TypeError): ActionDecoder()


def test_time_embedding():
    from models.vla.core.embedding import SinusoidalTimeEmbedding
    emb = SinusoidalTimeEmbedding(64)
    t = torch.tensor([0.5, 1.0])
    out = emb(t)
    assert out.shape == (2, 64)


def test_action_embedding():
    from models.vla.core.embedding import ActionEmbedding
    emb = ActionEmbedding(32, 64)
    x = torch.randn(2, 50, 32)
    out = emb(x)
    assert out.shape == (2, 50, 64)


def test_flow_matching_scheduler():
    from models.vla.core.diffusion import flow_matching_scheduler
    timesteps, dt = flow_matching_scheduler(10)
    assert len(timesteps) == 10
    assert timesteps[0] == 1.0
    assert dt < 0


def test_standard_vla_attention():
    from models.vla.core.standard.attention import StandardVLAAttention
    attn = StandardVLAAttention(dim=64, n_heads=4, n_kv_heads=1, head_dim=16)
    x = torch.randn(2, 8, 64)
    y = attn(x)
    assert y.shape == (2, 8, 64)


def test_standard_vla_ffn():
    from models.vla.core.standard.mlp import StandardVLAFFN
    ffn = StandardVLAFFN(dim=64, mlp_dim=128)
    x = torch.randn(2, 8, 64)
    y = ffn(x)
    assert y.shape == (2, 8, 64)