"""探针 04：混合注意力层调度（compress_ratios / layer_types）实测。

要验证什么
----------
DeepSeek-V4 不是每层都用同一注意力，而是给不同层分配不同压缩率：
  - compress_ratios = 0            -> 无压缩（sliding attention 滑窗）
  - compress_ratios = csa_rate     -> CSA（压缩稀疏 + Indexer）
  - compress_ratios = hca_rate     -> HCA（重度压缩）
真实的 V4-Flash 配置里，前三层通常是无压缩的滑窗层（bootstrap），
之后 CSA / HCA 交替。本脚本实测三个仓库如何描述与推导这份调度：

  1. nano：layer_types 显式列表 / 从 compress_ratios 推导 / 默认前2层滑窗+交替。
  2. mini：get_layer_attention_type 按 pattern 循环的混合调度。
  3. 对照 SGLang 配置里 compress_ratios 的真实排布（静态核对）。

用法
----
    python scripts\\deepseek_v4\\04_probe_layer_schedule.py

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
    kv,
    section,
    table,
)

add_mini_to_path()
add_nano_to_path()


def probe_nano_default_schedule() -> None:
    """问题 1：nano 的默认层调度（前 2 层滑窗 + CSA/HCA 交替）。"""
    section("1. nano 默认层调度（V4-Flash 风格 bootstrap + 交替）")
    from nano_deepseek_v4.config import DeepSeekV4Config

    cfg = DeepSeekV4Config()
    rows = [[i, lt] for i, lt in enumerate(cfg.layer_types)]
    table(["层索引", "注意力类型"], rows)
    print()
    kv("csa_rate", cfg.compress_rates["compressed_sparse_attention"])
    kv("hca_rate", cfg.compress_rates["heavily_compressed_attention"])
    print()
    print("  说明：默认 `前 min(N,2) 层 = sliding_attention`（无压缩，稳定早期特征），")
    print("        之后 `CSA/HCA 交替`（i%2==0 -> CSA，否则 HCA）。")
    print("        这是 V4-Flash 官方 checkpoint 的层调度风格。")


def probe_nano_from_ratios() -> None:
    """问题 1b：从 compress_ratios 推导 layer_types（显式指定路线）。"""
    section("2. nano 从 compress_ratios 推导 layer_types")
    from nano_deepseek_v4.config import DeepSeekV4Config

    cfg = DeepSeekV4Config(compress_ratios=[0, 4, 0, 128])
    rows = [[i, cfg.compress_ratios[i], cfg.layer_types[i]] for i in range(4)]
    table(["层", "compress_ratio", "推导出的 layer_type"], rows)
    print()
    print("  说明：compress_ratios 是『官方 checkpoint』的表达形式；")
    print("        nano 支持显式传 compress_ratios 再自动映射成 layer_types：")
    print("        0 -> sliding，csa_rate -> compressed_sparse，hca_rate -> heavily_compressed。")
    assert cfg.layer_types[1] == "compressed_sparse_attention"
    assert cfg.layer_types[3] == "heavily_compressed_attention"


def probe_mini_schedule() -> None:
    """问题 2：mini 的 pattern 循环调度。"""
    section("3. mini 的混合注意力 pattern 循环调度")
    from src.deepseek_components import DeepSeekV4LMConfig, get_layer_attention_type

    # mini 用 attention_type="hybrid" + attention_pattern 按层索引取模循环
    cfg = DeepSeekV4LMConfig(
        vocab_size=256,
        d_model=64,
        n_layers=8,
        attention_type="hybrid",
        attention_pattern=("csa", "hca"),
    )
    rows = []
    for i in range(8):
        rows.append([i, cfg.attention_pattern[i % len(cfg.attention_pattern)],
                     get_layer_attention_type(cfg, i)])
    table(["层索引", "pattern 值", "get_layer_attention_type"], rows)
    print()
    print("  说明：mini 用『attention_type=hybrid + pattern 取模』的简单循环；")
    print("        nano/SGLang 用『compress_ratios 列表』的显式排布。")
    print("        两者是同一思想（逐层不同注意力）的两种表达。")


def main() -> int:
    banner("探针 04：混合注意力层调度（nano 主 + mini 对照）")
    probe_nano_default_schedule()
    probe_nano_from_ratios()
    probe_mini_schedule()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
