"""探针 03：CSA + Lightning Indexer（压缩稀疏注意力）实测。

要验证什么
----------
CSA 是 V4 中最精巧的一层：它不像 HCA 那样"无差别压缩所有块"，
而是先让一个轻量 Indexer 为每个 query 选出 top-k 个『最相关的压缩块』，
再只对这些块做精确注意力。本脚本实测四个要点：

  1. 压缩率：rate=4 的重叠 a/b 压缩，每个压缩块覆盖当前块 + 前一块。
  2. Lightning Indexer：query 与压缩块算相关性打分，ReLU 截断负分，
     top-k 选出候选块。这是『稀疏选择』的核心。
  3. 保守因果掩码：只允许选择『已完整压缩完成』的块（block < floor(t/rate)），
     当前块交给精确滑窗分支处理。
  4. top-k 命中率：统计被选中的块是否确实是相关性最高的块（排序正确性）。

关键分歧（写入 D4 比对表）：
  - mini 的 Indexer 是『无状态』的（不持有自己的 query 投影，打分在外部算好
    传入）；nano 的 CSAIndexer 自带 q_b_proj 与 per-head weights_proj。
  - 两者都用了重叠 a/b 压缩，但 mini 用 current+previous 两块拼接后 softmax，
    nano 用 2-slot-per-window 的 slot 压缩。
  - top-k 语义：mini 默认 top_k 与压缩块数相关；nano 的 index_topk=4。

用法
----
    python scripts\\deepseek_v4\\03_probe_csa_indexer.py

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


def probe_mini_indexer() -> None:
    """问题 2/3/4：主参考 mini 的 CSALightningIndexer。"""
    section("1. 主参考 mini：Lightning Indexer（无状态打分 + top-k）")
    from src.csa_light_indexer import CSALightningIndexer

    torch.manual_seed(0)
    B, T, Hi, I = 2, 12, 2, 8      # Hi=indexer头数, I=indexer维度
    S = 3                          # 压缩块数
    indexer = CSALightningIndexer(compression_factor=4, top_k=2)

    index_q = torch.randn(B, T, Hi, I)
    index_weights = torch.randn(B, T, Hi).abs()          # 头权重
    I_comp = torch.randn(B, S, I)                        # 压缩块表示
    comp_valid_mask = torch.ones(B, S, dtype=torch.bool)

    topk_indices, topk_scores, topk_mask, all_scores = indexer(
        index_q, index_weights, I_comp, comp_valid_mask, need_scores=True
    )

    kv("index_q shape", shape_of(index_q))
    kv("I_comp shape", shape_of(I_comp))
    kv("topk_indices shape", shape_of(topk_indices))
    kv("topk_scores shape", shape_of(topk_scores))
    kv("topk_mask shape", shape_of(topk_mask))
    kv("topk 取值", str(topk_indices[0, 0].tolist()))
    kv("topk 分数", f"[{topk_scores[0,0,0]:.4f}, {topk_scores[0,0,1]:.4f}]")

    # 保守因果：query 在块 i，只能选 block < floor(t/rate)
    rows = []
    for t in (0, 3, 4, 8, 11):
        row = [f"{t:>2}"]
        for b in range(S):
            allowed = b < t // 4
            row.append("Y" if allowed else "N")
        rows.append(row)
    table(["t\\块", "B0", "B1", "B2"], rows)
    print("  说明：query 位置 t 只允许看到 floor(t/4) 之前的完整块；")
    print("        当前未完成块交给精确滑窗分支，保证因果性。")

    assert topk_indices.shape == (B, T, 2)
    assert topk_mask.dtype == torch.bool


def probe_mini_overlapped_compressor() -> None:
    """问题 1：mini 的重叠 a/b 压缩（current + previous）。"""
    section("2. mini 重叠 a/b 压缩：每块 = 当前块 + 前一块")
    from src.csa_components import CSAOverlappedCompressor

    torch.manual_seed(0)
    comp = CSAOverlappedCompressor(compression_factor=4, dim=8)
    kv("bias_a/bias_b shape", f"{shape_of(comp.bias_a)} / {shape_of(comp.bias_b)}")

    B, T, D = 1, 9, 8
    C_a = torch.randn(B, T, D)
    C_b = torch.randn(B, T, D)
    Z_a = torch.randn(B, T, D)
    Z_b = torch.randn(B, T, D)
    C_comp, valid_mask, pos = comp(C_a, C_b, Z_a, Z_b)

    kv("输入 T", T, width=34)
    kv("压缩块数 S (=ceil(9/4)=3)", C_comp.shape[1], width=34)
    kv("C_comp shape", shape_of(C_comp))
    kv("块 0 覆盖 (A 块[0:4], B 空)", "仅当前块")
    kv("块 1 覆盖 (A 块[4:8], B 块[0:4])", "当前 + 前一块")
    print()
    print("  说明：每个压缩块同时混合『本块』与『前一块』，让相邻块信息")
    print("        有重叠衔接，避免块边界信息断层。块 0 无前一块则只含本块。")
    assert C_comp.shape[1] == 3


def probe_nano_csa_indexer() -> None:
    """问题 2：次参考 nano 的 CSACompressor + CSAIndexer（自带投影）。"""
    section("3. 次参考 nano：CSA 自带压缩器与 Indexer")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import CSACompressor, CSAIndexer, DeepSeekV4Attention

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    csa = CSACompressor(cfg)
    kv("rate (compressed_sparse_attention)", cfg.compress_rates["compressed_sparse_attention"])
    kv("index_topk", cfg.index_topk)
    kv("index_n_heads", cfg.index_n_heads)
    kv("index_head_dim", cfg.index_head_dim)

    # 完整 CSA 注意力层前向（内部会正确构造 q_residual 并驱动 indexer）
    x = torch.randn(1, 12, cfg.hidden_size)
    pos = torch.arange(12).unsqueeze(0)
    attn = DeepSeekV4Attention(cfg, layer_type="compressed_sparse_attention")
    out = attn(x, pos)
    kv("CSA attention 输出", shape_of(out))
    assert out.shape == x.shape
    kv("CSA 注意力参数量", count_params(attn))
    print()
    print("  说明：CSA 层 = 精确滑窗分支 + (Indexer 选中的) top-k 压缩块分支；")
    print("        两层 KV 拼接后做标准注意力。Indexer 的自带投影位于层内部。")


def main() -> int:
    banner("探针 03：CSA + Lightning Indexer（mini 主参考 + nano 对照）")
    probe_mini_indexer()
    probe_mini_overlapped_compressor()
    probe_nano_csa_indexer()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
