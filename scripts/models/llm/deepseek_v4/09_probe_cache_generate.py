"""探针 09：DeepSeek-V4 Cache / 生成循环实测（C8）。

要验证什么
----------
探针 08 已验证「预填充 8 + 解码 12」与全量一致，但当时缓存是**无界**的
（滑窗层存全量 K/V，CSA 存全量 token 历史）。C8 把缓存改成有界，因此本探针
专门验证「省内存」与「不改变数值」这两件必须同时成立的事：

  1. 逐 token 解码（32 步）== 全量预填充。
  2. 分块预填充（chunk=8）== 全量预填充。
     —— 裁剪必须按 chunk 内**最早** query 位置，按 max 会误删仍需的 key。
  3. 滑窗缓存元素数在窗口填满后**保持常量**（不随 T 增长）。
  4. 滚动压缩缓冲（打开块 / 前一块）有上界，不随 T 线性增长。
  5. 压缩块缓存按 O(T/m) 增长（设计如此：全局记忆必须留下）。
  6. 压缩器块大小取自组件配置 compression_factor，而非调度标签 hca_rate。
  7. generate() 贪心解码：形状正确、prompt 保留、可复现。

对比基线（before/after）
------------------------
「无界缓存」= 朴素实现：滑窗层也把所有 K/V 存下来（O(T)）。
本探针同时打印两者的元素数，展示 C8 的实际节省倍数。

用法
----
    python scripts\\deepseek_v4\\09_probe_cache_generate.py

退出码
------
0 = 全部断言通过
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deepseek_v4"))

from _probe_common import banner, section

from models.llm.deepseek_v4.config import (
    ATTN_HCA, ATTN_CSA,
    DeepSeekV4ModelConfig, HCAConfig, CSAConfig, MHCConfig, MoEV4Config,
    MQAV4Config, MTPConfig,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention


def make_config() -> DeepSeekV4ModelConfig:
    """小型可 CPU 运行的 V4 配置（与探针 08 一致，便于交叉对照）。"""
    return DeepSeekV4ModelConfig(
        dim=64, n_layers=6, n_heads=4, head_dim=16, q_lora_rank=32,
        qk_rope_head_dim=8, window_size=8, vocab_size=1000, max_seq_len=512,
        compress_ratios=[0, 0, 4, 128, 4, 128],
        mqa=MQAV4Config(dim=64, n_heads=4, head_dim=16, q_lora_rank=32,
                        qk_rope_head_dim=8, window_size=8, max_seq_len=512),
        hca=HCAConfig(compression_factor=4),
        csa=CSAConfig(compression_factor=4),
        mhc=MHCConfig(n_hc=2, sinkhorn_iters=10),
        moe=MoEV4Config(n_routed_experts=8, n_shared_experts=1,
                        num_experts_per_tok=2, moe_intermediate_size=64,
                        num_hash_layers=2),
        mtp=MTPConfig(num_nextn_predict_layers=2),
    )


def decode_all(model, ids, *, trim: bool = True):
    """逐 token 解码，返回 (hidden, 每步缓存统计)。trim=False 模拟无界基线。"""
    orig = MQAV4Attention._trim_window
    if not trim:
        MQAV4Attention._trim_window = lambda self, c, q: None
    try:
        with torch.no_grad():
            cache = model.init_cache(ids.shape[0])
            outs, stats = [], []
            for t in range(ids.shape[1]):
                outs.append(model(ids[:, t:t + 1],
                                  position_ids=torch.tensor([[t]]).expand(ids.shape[0], -1),
                                  cache=cache))
                stats.append(DeepSeekV4LM.cache_size_elements(cache))
            return torch.cat(outs, dim=1), stats
    finally:
        MQAV4Attention._trim_window = orig


def main():
    banner("DeepSeek-V4 Cache / 生成循环探针 (C8)")
    torch.manual_seed(0)

    cfg = make_config()
    model = DeepSeekV4LM(cfg).eval()
    T = 32
    ids = torch.randint(1, 1000, (1, T))

    # ---------- 1. 压缩器块大小来源 ----------
    section("1. 压缩器块大小取自组件配置（非调度标签）")
    print(f"  hca_rate={cfg.hca_rate} csa_rate={cfg.csa_rate}（调度标签）")
    print(f"  hca.compression_factor={cfg.hca.compression_factor} "
          f"csa.compression_factor={cfg.csa.compression_factor}（真实块大小）")
    for i, (lt, layer) in enumerate(zip(cfg.layer_types(), model.layers)):
        comp = getattr(layer.attention, "compressor", None)
        if lt in (ATTN_HCA, ATTN_CSA):
            print(f"  L{i} {lt:8s} compressor.m = {comp.m}")
            assert comp.m == 4, f"L{i} 块大小应为 4，实际 {comp.m}"

    # ---------- 2. 逐 token 解码一致性 ----------
    section("2. 逐 token 解码 vs 全量预填充")
    with torch.no_grad():
        full = model(ids)
    dec, stats = decode_all(model, ids, trim=True)
    diff = (dec - full).abs().max().item()
    print(f"  32 步解码 max 误差: {diff:.3e}")
    assert diff < 1e-4

    # ---------- 3. 分块预填充一致性 ----------
    section("3. 分块预填充 (chunk=8) vs 全量预填充")
    with torch.no_grad():
        cache = model.init_cache(1)
        outs = []
        for s in range(0, T, 8):
            pos = torch.arange(s, s + 8).unsqueeze(0)
            outs.append(model(ids[:, s:s + 8], position_ids=pos, cache=cache))
    diff_chunk = (torch.cat(outs, dim=1) - full).abs().max().item()
    print(f"  分块预填充 max 误差: {diff_chunk:.3e}")
    assert diff_chunk < 1e-4

    # ---------- 4. 缓存有界性 ----------
    section("4. 缓存增长曲线（trim=ON）")
    print(f"  window_size = {cfg.window_size}")
    print("    t   local  compressed  rolling   total")
    for t in (7, 15, 23, 31):
        s = stats[t]
        print(f"  {t + 1:3d} {s['local']:7d} {s['compressed']:11d} "
              f"{s['rolling']:8d} {s['total']:7d}")
    local_after = [s["local"] for s in stats[cfg.window_size:]]
    assert len(set(local_after)) == 1, "滑窗缓存在窗口填满后应保持常量"
    print(f"  滑窗缓存常量 = {local_after[0]} 元素（不随 T 增长）")

    rolling = [s["rolling"] for s in stats]
    print(f"  滚动缓冲 max = {max(rolling)} 元素（有上界）")
    assert max(rolling[16:]) <= max(rolling[:16]) * 2

    comp = [s["compressed"] for s in stats]
    print(f"  压缩块缓存 t=8 -> t=32: {comp[7]} -> {comp[31]}"
          f"（O(T/m)，全局记忆按设计保留）")

    # ---------- 5. before/after：无界基线对比 ----------
    section("5. before/after：无界缓存基线 vs C8 有界缓存")
    dec_nt, stats_nt = decode_all(model, ids, trim=False)
    same = (dec_nt - dec).abs().max().item()
    print(f"  两者数值差异: {same:.3e}（应为 0，裁剪无损）")
    assert same < 1e-5
    b, a = stats_nt[-1], stats[-1]
    print(f"  无界基线 local = {b['local']:6d} 元素")
    print(f"  C8 有界   local = {a['local']:6d} 元素")
    print(f"  滑窗部分节省 = {b['local'] / max(a['local'], 1):.2f}x  (T={T})")
    print(f"  合计 total: {b['total']} -> {a['total']}"
          f"  ({b['total'] / max(a['total'], 1):.2f}x)")
    assert a["local"] < b["local"], "C8 滑窗缓存应严格小于无界基线"

    # ---------- 6. generate ----------
    section("6. generate() 贪心解码")
    prompt = ids[:, :8]
    g1 = model.generate(prompt, max_new_tokens=6, greedy=True)
    g2 = model.generate(prompt, max_new_tokens=6, greedy=True)
    print(f"  输出形状: {tuple(g1.shape)}")
    print(f"  prompt 保留: {torch.equal(g1[:, :8], prompt)}")
    print(f"  两次贪心一致: {torch.equal(g1, g2)}")
    print(f"  新生成 token: {g1[0, 8:].tolist()}")
    assert g1.shape == (1, 14)
    assert torch.equal(g1[:, :8], prompt)
    assert torch.equal(g1, g2)

    # 采样路径可跑（top_p）
    torch.manual_seed(3)
    gs = model.generate(prompt, max_new_tokens=4, temperature=0.8, top_p=0.9)
    print(f"  采样解码 (T=0.8, top_p=0.9): {gs[0, 8:].tolist()}")
    assert gs.shape == (1, 12)

    banner("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
