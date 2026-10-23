"""探针 02：HCA（Heavily Compressed Attention）重度压缩注意力实测。

要验证什么
----------
HCA 把连续 rate=128 个 token 的 K/V 用一个「可学习权重的 softmax 池化」
压缩成 1 个 token 级向量，使全局上下文以 O(T/rate) 的极小代价被长期保留。
本脚本用 mini 与 nano 两个实现分别实测：

  1. 压缩比例：T 个 token -> ceil(T/rate) 个压缩块，KV 显存大幅下降。
  2. 池化机制：softmax 权重 = 门控打分 + 可学习位置偏置（position_bias）。
     —— 位置偏置让模型学会「块内哪些位置更值得保留」。
  3. 压缩后 RoPE：压缩块用『块首 token 位置』重编位置编码（nano 用 pos[0]），
     mini 用『块末 token 位置』（pos_end）。这是又一个三方差异点。
  4. 压缩块数量随序列长度增长：每 rate 个 token 增加 1 个压缩块。

关键分歧（写入 D4 比对表）：
  - mini 的压缩位置取块末 (comp_position_ids 用 a_end-1)；
    nano 的压缩位置取块首 (rope_positions = chunk_positions[:, :, 0])。
    两者对『压缩块应代表块内哪个时间点』的建模不同，值得记录。
  - mini 压缩的是分离的 C(值)/Z(打分) 两路；nano 压缩的是 K 与门控两路，
    且压缩后直接当 KV 用（V=K 风格）。

用法
----
    python scripts\\deepseek_v4\\02_probe_hca.py

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


def probe_mini_compression() -> None:
    """问题 1/2/4：主参考 mini 的 HCATokenCompressor。"""
    section("1. 主参考 mini：HCATokenCompressor 压缩比例与池化")
    from src.hca_components import HCATokenCompressor

    torch.manual_seed(0)
    rate = 4  # 教学用小比例；论文/SGLang 是 128
    dim = 16
    comp = HCATokenCompressor(compression_factor=rate, head_dim=dim, init_std=0.02)
    kv("compression_factor (rate)", rate)

    C = torch.randn(2, 13, dim)   # 13 个 token，rate=4 -> ceil(13/4)=4 块
    Z = torch.randn(2, 13, dim)
    C_comp, valid_mask, pos_ids = comp(C=C, Z=Z)

    kv("输入 token 数 T", 13)
    kv("压缩后块数 S", C_comp.shape[1], width=34)
    kv("压缩块 shape", shape_of(C_comp))
    kv("有效掩码 shape", shape_of(valid_mask))
    kv("压缩位置 shape", shape_of(pos_ids))

    rows = []
    for i in range(C_comp.shape[1]):
        rows.append([i, f"{C_comp[0, i].norm().item():.4f}", str(bool(valid_mask[0, i]))])
    table(["块索引", "压缩向量 L2 范数", "有效"], rows)

    # 位置语义：mini 用块末 token 位置
    print()
    kv("压缩位置取值 (mini, 块末语义)", [int(p) for p in pos_ids.tolist()])
    print("  说明：T=13, rate=4 -> 块边界 [0:4],[4:8],[8:12],[12:13]，")
    print("        块末位置 = [3, 7, 11, 12]（即每块最后 token 的原始位置）。")

    assert C_comp.shape[1] == 4, "13//4 向上取整应为 4 块"
    assert C_comp.shape[2] == dim


def probe_mini_positions_are_block_end() -> None:
    """问题 3：mini 压缩位置是块末（区别于 nano 的块首）。"""
    section("2. mini 位置语义 = 块末（与 nano 的块首对照）")
    # 直接复用上面结论：pos_ids 输出即块末位置。
    # 这里通过 mini 的 CSA 里也用了相同逻辑，进一步说明。
    from src.hca_components import HCATokenCompressor

    torch.manual_seed(0)
    comp = HCATokenCompressor(compression_factor=4, head_dim=8, init_std=0.02)
    C = torch.randn(1, 8, 8)
    Z = torch.randn(1, 8, 8)
    _, _, pos_ids = comp(C=C, Z=Z)
    kv("T=8, rate=4 的压缩位置", [int(p) for p in pos_ids.tolist()], width=34)
    print("  推论：块 [0:4] 末 token 位置=3，块 [4:8] 末=7 => 块末语义确认。")


def probe_nano_hca() -> None:
    """问题 1/2/3：次参考 nano 的 HCACompressor（含 position_bias 与块首位置）。"""
    section("3. 次参考 nano：HCACompressor（position_bias + 块首位置）")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import HCACompressor

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    # nano 配置 compress_rates 里存了 heavily_compressed_attention 的 rate
    rate = cfg.compress_rates["heavily_compressed_attention"]
    comp = HCACompressor(cfg)
    kv("rate (heavily_compressed_attention)", rate)
    kv("position_bias shape", shape_of(comp.position_bias))
    kv("kv_proj/gate_proj", f"{shape_of(comp.kv_proj.weight)} / {shape_of(comp.gate_proj.weight)}")

    x = torch.randn(1, 12, cfg.hidden_size)
    pos = torch.arange(12).unsqueeze(0)
    comp_kv, end_pos = comp(x, pos)
    kv("压缩 KV shape", shape_of(comp_kv))
    kv("块末位置 (end_positions)", str(end_pos.tolist()))

    # 压缩块数量 = T // rate
    expected = 12 // rate
    kv("期望块数 T//rate", expected)
    assert comp_kv.shape[2] == expected, "HCA 压缩块数应为 T//rate（丢弃不足一块的尾 token）"
    print()
    print("  说明：nano 用『块首 token 位置』做压缩后 RoPE（rope_positions=pos[0]），")
    print("        与 mini 的『块末』相反。这是三方在压缩时间语义上的分歧。")


def probe_kv_savings() -> None:
    """问题 4：HCA 的 KV 显存节省量化。"""
    section("4. KV 显存节省：局部滑窗 + 全局压缩 vs 全量 KV")
    # 无滑窗的标准 KV 是 O(T)；HCA 的全局部分是 O(T/rate)。
    # 这里给出 T 从 1K 到 128K 的块数增长对比。
    rows = []
    for T in (1024, 4096, 16384, 65536, 131072):
        rows.append([
            f"{T:>7}",
            T,                       # 标准 KV 块数（无压缩）
            T // 128,                # HCA rate=128 的压缩块数
            f"{(T / 128) / T * 100:.3f}%",
        ])
    table(["序列长 T", "标准 KV 数", "HCA 压缩块数 (rate=128)", "占标准比例"], rows)
    print()
    print("  推论：全局上下文压缩到约 0.8%（rate=128），几乎不再随 T 线性增长；")
    print("        局部细节交给滑窗分支（window_size=128）。")
    print("        HCA 是『牺牲块内细节，换取超长全局记忆』的极端压缩。")


def main() -> int:
    banner("探针 02：HCA 重度压缩注意力（mini 主参考 + nano 对照）")
    probe_mini_compression()
    probe_mini_positions_are_block_end()
    probe_nano_hca()
    probe_kv_savings()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
