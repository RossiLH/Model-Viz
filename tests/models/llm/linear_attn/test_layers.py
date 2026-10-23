import torch
from models.llm.linear_attn.layers import (
    GatedLinearAttentionLayer, GatedDeltaNetLayer,
)
from models.llm.linear_attn.variants import gla_step, gated_delta_net_step


def _manual_forward(layer, gate_fn, x, n_heads, head_dim, step_fn):
    """手动复刻 layer.forward 的逐 token 递推，验证封装与算子路径一致。"""
    B, T, _ = x.shape
    q = layer.q_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = layer.k_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    v = layer.v_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    S = torch.zeros(B, n_heads, head_dim, head_dim, device=x.device)
    outputs = []
    for t in range(T):
        out, S = step_fn(
            S, q[:, :, t], k[:, :, t], v[:, :, t], *gate_fn(x, t, n_heads, head_dim)
        )[0:2]
        outputs.append(out)
    o = torch.stack(outputs, dim=2).transpose(1, 2).reshape(B, T, n_heads * head_dim)
    return layer.o_proj(o)


def test_gla_layer_shape():
    B, T, D, H, Dk = 2, 5, 16, 4, 8
    layer = GatedLinearAttentionLayer(D, H, Dk)
    out = layer(torch.randn(B, T, D))
    assert out.shape == (B, T, D)


def test_gated_delta_net_layer_shape():
    B, T, D, H, Dk = 2, 5, 16, 4, 8
    layer = GatedDeltaNetLayer(D, H, Dk)
    out = layer(torch.randn(B, T, D))
    assert out.shape == (B, T, D)


def test_gla_layer_equals_manual_loop():
    torch.manual_seed(0)
    B, T, D, H, Dk = 2, 6, 16, 4, 8
    layer = GatedLinearAttentionLayer(D, H, Dk)
    x = torch.randn(B, T, D)
    out = layer(x)

    def gate_fn(x, t, n_heads, head_dim):
        g = layer.gate_proj(x).view(x.shape[0], x.shape[1], n_heads, head_dim)
        return (g.transpose(1, 2)[:, :, t],)  # gla_step 的 gate 参数

    manual = _manual_forward(layer, gate_fn, x, H, Dk, gla_step)
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-5)


def test_gated_delta_net_layer_equals_manual_loop():
    torch.manual_seed(1)
    B, T, D, H, Dk = 2, 6, 16, 4, 8
    layer = GatedDeltaNetLayer(D, H, Dk, gate_rank=6)
    x = torch.randn(B, T, D)
    out = layer(x)

    def gate_fn(x, t, n_heads, head_dim):
        g = layer.gate_up(layer.gate_down(x)).view(
            x.shape[0], x.shape[1], n_heads, head_dim
        ).transpose(1, 2)[:, :, t]
        b = layer.beta_proj(x).transpose(1, 2)[:, :, t]
        return g, b  # gate_logit, beta_logit

    manual = _manual_forward(layer, gate_fn, x, H, Dk, gated_delta_net_step)
    assert torch.allclose(out, manual, atol=1e-6, rtol=1e-5)


def test_gate_is_learnable_data_dependent_projection():
    """门控投影 W_gate 可训练、且随输入 x 变化（data-dependent）。"""
    torch.manual_seed(2)
    D, H, Dk = 16, 4, 8
    layer = GatedLinearAttentionLayer(D, H, Dk)
    x1 = torch.randn(2, 3, D)
    x2 = torch.randn(2, 3, D)
    gate1 = layer.gate_proj(x1)
    gate2 = layer.gate_proj(x2)
    # 不同输入产生不同 gate（不是常数）
    assert not torch.allclose(gate1.detach(), gate2.detach())
    # W_gate 参与反向传播
    loss = layer(x2).sum()
    loss.backward()
    assert layer.gate_proj.weight.grad is not None
    assert layer.gate_proj.weight.grad.abs().sum() > 0