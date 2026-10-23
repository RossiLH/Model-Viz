"""探针 01：MQA 注意力基底实测（主参考 mini + 次参考 nano 对照）。

要验证什么
----------
DeepSeek-V4 的注意力基底与标准 MHA / MLA 都不同，本脚本逐项实测并对照两个仓库：

  1. Multi-Query Attention：Q 有 H 个头，但 K/V 共享（num_key_value_heads=1）。
     —— 大幅削减 KV cache，是 V4 能在 128 层规模保持显存可控的基础。
  2. Q 的 LoRA 式低秩投影：q_a_proj(隐层->低秩) -> RMSNorm -> q_b_proj(低秩->全宽)。
     —— 用极少的参数重建 Q，隐含"低秩化注意力查询"思想。
  3. Partial RoPE：只对 head_dim 的后 rope_dim 维做旋转，前部保留无位置编码的
     "语义信息"，后部编码相对位置。比全量 RoPE 更灵活。
  4. Attention Sink：在 softmax 前额外拼接一个可学习的"注意力槽"列，
     吸收模型对无关上下文的过度注意力，稳定长序列训练（StreamingLLM 思想）。
  5. Grouped Output Projection：把 H 个头分组，组内共享一个低秩输出矩阵，
     再拼回 —— 输出侧的低秩压缩（o_lora_rank / o_groups）。

关键分歧（写入 D4 比对表）：
  - mini 的 K/V 是"分离"的（K 加 RoPE，V 不加）；
    nano 的 K=V 共享同一张量，因此输出端必须"逆 RoPE"把旋转抵消掉。
    本脚本分别实测两种路径，给出数值证据。

用法
----
    python scripts\\deepseek_v4\\01_probe_mqa_attention.py

退出码
------
0 = 全部断言通过
"""
from __future__ import annotations

import torch

from _probe_common import (
    add_mini_to_path,
    add_nano_to_path,
    banner,
    count_params,
    kv,
    section,
    shape_of,
    table,
)

add_mini_to_path()
add_nano_to_path()


def probe_mini_mqa_shapes() -> None:
    """问题 1/2/3/4/5：主参考 mini 的 HCA 基底模块结构（用 HCA 配置逼近 MQA）。"""
    section("1. 主参考 mini：MQA 基底的投影结构")
    from src.deepseek_hca_attention import HCAAttention
    from src.hca_components import HCAConfig

    cfg = HCAConfig(
        d_model=64,
        n_heads=4,
        head_dim=16,
        window_size=8,
        compression_factor=4,
        max_seq_len=128,
        rotary_dim=8,          # 只对后 8 维做 RoPE（partial）
        output_projection_groups=2,
    )
    attn = HCAAttention(cfg)

    kv("q_proj", f"{shape_of(attn.q_proj.weight)}  d_model -> H*head_dim (全宽 Q)")
    kv("kv_proj", f"{shape_of(attn.kv_proj.weight)}  d_model -> head_dim (共享 K/V，MQA)")
    kv("z_proj", f"{shape_of(attn.z_proj.weight)}  d_model -> head_dim (压缩打分)")
    kv("sink_k/sink_v", f"{shape_of(attn.sink_k)} / {shape_of(attn.sink_v)}  可学习注意力槽")
    if attn.use_grouped_output_projection:
        kv("分组输出", f"{attn.out_proj.num_groups} 组 x 每组 {attn.out_proj.heads_per_group} 头")
        kv("每组的参数量", f"{attn.out_proj.group_projs[0].weight.numel()}  (=group_dim^2 低秩)")
    kv("总参数量", count_params(attn))

    # 断言 MQA：K/V 只有 head_dim（无头数因子）
    assert attn.kv_proj.out_features == cfg.head_dim, "MQA 要求 K/V 仅 head_dim 宽"
    print()
    print("  MQA 判据：K/V 投影输出宽 = head_dim（而非 H*head_dim）=> 通过")


def probe_partial_rope() -> None:
    """问题 3：Partial RoPE —— 只有部分维度被旋转。"""
    section("2. Partial RoPE 实测（用 mini 的 RotaryEmbedding）")
    from src.deepseek_hca_attention import HCAAttention
    from src.hca_components import HCAConfig

    cfg = HCAConfig(
        d_model=64, n_heads=4, head_dim=16,
        window_size=8, compression_factor=4, max_seq_len=128,
        rotary_dim=8, use_grouped_output_projection=False,
    )
    attn = HCAAttention(cfg)
    torch.manual_seed(0)

    q = torch.randn(2, 6, 4, 16)
    pos = torch.arange(6)
    q_rot = attn.rope(q, position_ids=pos)

    # 前 (head_dim - rotary_dim) 维应不变（NoPE），后 rotary_dim 维应被旋转
    nope = q[..., : cfg.head_dim - cfg.rotary_dim]
    nope_rot = q_rot[..., : cfg.head_dim - cfg.rotary_dim]
    rope_part = q[..., cfg.head_dim - cfg.rotary_dim:]
    rope_part_rot = q_rot[..., cfg.head_dim - cfg.rotary_dim:]

    kv("NoPE 部分 (前8维) 是否不变", f"max diff = {(nope - nope_rot).abs().max().item():.3e}")
    kv("RoPE 部分 (后8维) 是否变化", f"max diff = {(rope_part - rope_part_rot).abs().max().item():.3e}")
    assert torch.allclose(nope, nope_rot, atol=1e-6), "NoPE 部分不应被旋转"
    assert not torch.allclose(rope_part, rope_part_rot, atol=1e-6), "RoPE 部分应被旋转"


def probe_nano_inverse_rope() -> None:
    """关键分歧：nano 用 V=K + 输出逆 RoPE。"""
    section("3. 关键分歧：nano 的 V=K 与逆 RoPE")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import apply_partial_rope, rope_cos_sin

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    rope_dim = cfg.qk_rope_head_dim
    kv("qk_rope_head_dim (partial_rotary_factor=%.1f)" % cfg.partial_rotary_factor, rope_dim)
    kv("head_dim", cfg.head_dim)

    x = torch.randn(2, 3, cfg.num_attention_heads, cfg.head_dim)
    pos = torch.arange(3)
    cos, sin = rope_cos_sin(pos, rope_dim, cfg.rope_theta)

    x_rot = apply_partial_rope(x, cos, sin)
    x_back = apply_partial_rope(x_rot, cos, -sin)  # 逆旋转用 -sin

    kv("前向旋转后", f"max diff vs 原始 = {(x_rot - x).abs().max().item():.3e} (应 >0)")
    kv("逆旋转还原", f"max diff vs 原始 = {(x_back - x).abs().max().item():.3e} (应 ~0)")
    assert torch.allclose(x_back, x, atol=1e-5), "逆 RoPE 应能还原原向量"

    print()
    print("  说明：nano 里 K 和 V 是同一张量（local_kv 直接当 V 用），")
    print("        因此必须：加 RoPE -> 注意力 -> 对输出做逆 RoPE，")
    print("        才能让『位置编码』只影响注意力权重、而不污染输出的语义向量。")
    print("        mini 则把 K/V 分开（K 加 RoPE、V 不加），无需逆旋转 ——")
    print("        这是两种等价的『RoPE 只参与注意力、不污染值』的实现路线。")


def probe_nano_grouped_linear() -> None:
    """问题 5：nano 的 GroupedLinear（分组低秩输出投影）。"""
    section("4. nano 的 GroupedLinear（o_a_proj: 全宽 -> o_groups 组低秩）")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import DeepSeekV4Attention

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    attn = DeepSeekV4Attention(cfg, layer_type="sliding_attention")

    kv("o_a_proj (GroupedLinear)", f"{shape_of(attn.o_a_proj.weight)}")
    kv("o_b_proj", f"{shape_of(attn.o_b_proj.weight)}")
    kv("o_groups", cfg.o_groups)
    kv("o_lora_rank", cfg.o_lora_rank)
    kv("attention_width (H*head_dim)", cfg.attention_width)

    # 全模型前向：确认 MQA + 逆RoPE + 分组输出的形状闭合
    # 注意：nano 的 rope_cos_sin 需要 position_ids 的 batch 维度与 x 一致
    #       （即 [B, T] 二维），这里用 batch=1 的二维 position_ids。
    x = torch.randn(1, 8, cfg.hidden_size)
    pos = torch.arange(8).unsqueeze(0)  # [1, T]
    out = attn(x, pos)
    kv("attention 前向输出", shape_of(out))
    assert out.shape == x.shape, "注意力输出应与输入同形"
    kv("注意力参数量", count_params(attn))


def main() -> int:
    banner("探针 01：MQA 注意力基底（mini 主参考 + nano 对照）")
    probe_mini_mqa_shapes()
    probe_partial_rope()
    probe_nano_inverse_rope()
    probe_nano_grouped_linear()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
