"""探针 06：MoE 路由机制实测（三方对比重点）。

要验证什么
----------
DeepSeek-V4 的 MoE 有三处与常规 MoE 不同的设计，本脚本逐一实测：

  1. 打分函数 scoring_func = "sqrtsoftplus"，即 s = sqrt(softplus(logits))。
     常规做法是 softmax（归一化竞争）或 sigmoid（独立打分）。
     为什么用 sqrt(softplus)？—— 它是无上界、单调、且在负半轴平滑趋 0 的，
     既保留"专家亲和度可以任意大"的表达力，又不像 exp 那样爆炸。
     本脚本对三种打分函数做数值对照，看它们的取值范围与梯度行为差异。

  2. 负载均衡：DeepSeek 的正统做法是 aux-loss-free（noaux_tc），
     即不加辅助损失，而是给每个专家一个可动态调整的偏置 e_score_correction_bias，
     根据实际负载反馈更新。而主参考 mini 用的是显式 MSE 均衡损失。
     —— 这是 D4 比对表的一个关键分歧点，本脚本量化两种机制的行为差异。

  3. Hash 路由：前若干层（n_hash_layers=3）用 token-id 确定性哈希路由，
     不走学习路由。为什么？—— 前几层特征还很"词表化"，学习路由容易塌缩到
     少数专家；用哈希强制均匀铺开，是一种零成本的冷启动均衡。
     本脚本实测哈希路由的负载分布均匀性。

用法
----
    python scripts\\deepseek_v4\\06_probe_moe.py

退出码
------
0 = 全部断言通过
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

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

from src.deepseek_moe import DeepSeekMoE, DeepSeekMoEConfig  # noqa: E402


def probe_scoring_functions() -> None:
    """问题 1：三种打分函数的数值行为对照。"""
    section("1. 打分函数对照：softmax vs sigmoid vs sqrt(softplus)")
    torch.manual_seed(0)
    logits = torch.tensor([[-4.0, -1.0, 0.0, 0.5, 2.0, 6.0]])

    softmax_s = F.softmax(logits, dim=-1)
    sigmoid_s = torch.sigmoid(logits)
    sqrtsp_s = torch.sqrt(F.softplus(logits) + 1e-9)

    rows = []
    for i, lg in enumerate(logits[0].tolist()):
        rows.append([
            f"{lg:+.1f}",
            f"{softmax_s[0, i]:.4f}",
            f"{sigmoid_s[0, i]:.4f}",
            f"{sqrtsp_s[0, i]:.4f}",
        ])
    table(["logit", "softmax", "sigmoid", "sqrt(softplus)"], rows)

    print()
    kv("softmax 和", f"{softmax_s.sum():.4f}   <- 归一化，专家间零和竞争")
    kv("sigmoid 上界", "1.0      <- 有界，饱和后梯度消失")
    kv("sqrt(softplus) 上界", "无界     <- 强亲和可无限表达，且负半轴平滑趋 0")

    # 梯度行为：大 logit 处 sigmoid 饱和，sqrt(softplus) 仍有梯度
    section("1b. 大 logit 处的梯度（饱和性对比）")
    rows = []
    for lg in (2.0, 6.0, 12.0):
        x1 = torch.tensor([lg], requires_grad=True)
        torch.sigmoid(x1).backward()
        x2 = torch.tensor([lg], requires_grad=True)
        torch.sqrt(F.softplus(x2) + 1e-9).backward()
        rows.append([f"{lg:+.1f}", f"{x1.grad.item():.3e}", f"{x2.grad.item():.3e}"])
    table(["logit", "d sigmoid", "d sqrt(softplus)"], rows)
    print("  推论：logit 越大 sigmoid 梯度越趋 0（饱和），")
    print("        sqrt(softplus) 梯度趋于 1/(2*sqrt(x))，衰减但不消失。")


def probe_mini_moe_routing() -> None:
    """问题 2：主参考 mini 的学习路由 + 显式均衡损失。"""
    section("2. 主参考 mini：学习路由 + 显式 MSE 均衡损失")
    cfg = DeepSeekMoEConfig(
        d_model=32,
        num_experts=8,
        top_k=2,
        router_type="learned",
        router_score_fn="sqrt_softplus",
        balance_loss_weight=0.01,
    )
    moe = DeepSeekMoE(cfg)
    torch.manual_seed(0)
    x = torch.randn(2, 16, 32)

    out = moe(x)
    # forward 返回值可能是 tensor 或 (tensor, aux)，两种都兼容
    if isinstance(out, tuple):
        y, aux = out[0], out[1]
    else:
        y, aux = out, {}

    kv("输入 x", shape_of(x))
    kv("输出 y", shape_of(y))
    kv("参数量", count_params(moe))
    kv("router_score_fn", cfg.router_score_fn)
    kv("均衡机制", "显式 MSE 损失 (balance_loss_weight=0.01)")
    if isinstance(aux, dict) and aux:
        print()
        print("  aux 统计键：")
        for k in sorted(aux.keys()):
            kv("  " + k, shape_of(aux[k]), width=32)
    assert y.shape == x.shape, "MoE 输出应与输入同形"


def probe_hash_routing() -> None:
    """问题 3：哈希路由的负载均匀性。"""
    section("3. Hash 路由的负载分布（前 n 层用，冷启动均衡）")
    cfg = DeepSeekMoEConfig(
        d_model=32,
        num_experts=8,
        top_k=2,
        router_type="hash",
        hash_routing_stride=1,
    )
    moe = DeepSeekMoE(cfg)
    torch.manual_seed(0)
    x = torch.randn(4, 64, 32)
    # 哈希路由依赖 token id，故必须传 input_ids
    input_ids = torch.randint(0, 500, (4, 64))

    out = moe(x, input_ids=input_ids)
    y = out[0] if isinstance(out, tuple) else out
    kv("输出", shape_of(y))

    # 直接调用内部哈希函数，统计专家负载
    _, _, _, topk_indices, _ = moe._hash_routing(
        input_ids=input_ids, B=4, T=64, device=x.device, dtype=x.dtype
    )
    counts = torch.bincount(topk_indices.reshape(-1), minlength=cfg.num_experts).float()
    frac = counts / counts.sum()
    ideal = 1.0 / cfg.num_experts

    rows = [[f"E{i}", int(counts[i]), f"{frac[i]:.4f}"] for i in range(cfg.num_experts)]
    table(["专家", "命中次数", "占比"], rows)
    print()
    kv("理想占比", f"{ideal:.4f}")
    kv("最大偏离理想", f"{(frac - ideal).abs().max().item():.4f}")
    kv("负载标准差", f"{frac.std().item():.4f}")
    print("  推论：token-id 取模哈希在 id 均匀分布时天然接近均衡，")
    print("        代价是完全不考虑语义亲和（纯静态映射）。")


def probe_nano_aux_loss_free() -> None:
    """对照：次参考 nano 的 aux-loss-free 偏置更新（DeepSeek 正统做法）。"""
    section("4. 次参考 nano：aux-loss-free 偏置更新（正统 noaux_tc）")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import DeepSeekV4MoE

    cfg = DeepSeekV4Config()
    # nano 用 layer_type 字符串区分路由方式："moe" = 学习路由，"hash_moe" = 哈希路由
    moe = DeepSeekV4MoE(cfg, layer_type="moe")
    kv("n_routed_experts", cfg.n_routed_experts)
    kv("num_experts_per_tok", cfg.num_experts_per_tok)
    kv("scoring_func", cfg.scoring_func)
    kv("router_bias_update_speed", cfg.router_bias_update_speed)

    bias_before = moe.e_score_correction_bias.clone()
    # 构造极不均衡的负载：所有路由都命中专家 0
    # update_balance_bias 接收 topk_idx（而非 counts），内部自行 bincount
    topk_idx = torch.zeros(10, 10, cfg.num_experts_per_tok, dtype=torch.long)
    moe.update_balance_bias(topk_idx)
    bias_after = moe.e_score_correction_bias

    delta = bias_after - bias_before
    rows = []
    for i in range(min(6, cfg.n_routed_experts)):
        rows.append([f"E{i}", f"{bias_before[i]:+.5f}", f"{bias_after[i]:+.5f}", f"{delta[i]:+.5f}"])
    table(["专家", "更新前偏置", "更新后偏置", "变化"], rows)
    print()
    kv("过载专家 E0 偏置变化", f"{delta[0]:+.5f}   <- 应为负（抑制）")
    kv("空闲专家 E1 偏置变化", f"{delta[1]:+.5f}   <- 应为正（鼓励）")
    assert delta[0] < 0, "过载专家的偏置应被下调"
    assert delta[1] > 0, "空闲专家的偏置应被上调"
    print("  推论：偏置只作用于 top-k 的『选择』，不进入最终权重，")
    print("        因此不引入梯度偏差 —— 这正是 aux-loss-free 的核心优势。")


def main() -> int:
    banner("探针 06：MoE 路由机制（mini 主参考 + nano 对照）")
    probe_scoring_functions()
    probe_mini_moe_routing()
    probe_hash_routing()
    probe_nano_aux_loss_free()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
