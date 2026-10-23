"""DeepSeekV4 分析文档补充实测证据采集（CPU）。

只输出文档需要引用的数字，全部标注测量条件。
"""
import sys
import time

import torch

sys.path.insert(0, "src")

from models.llm.deepseek_v4.config import (
    CSAConfig,
    DeepSeekV4ModelConfig,
    HCAConfig,
    IndexerConfig,
    MQAV4Config,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM

torch.manual_seed(0)
SEP = "=" * 72


def nparam(m):
    return sum(p.numel() for p in m.parameters())


def make_cfg(**kw):
    """dim=128, H=4, Dh=32：HCA/CSA 只带压缩参数，形状取自 mqa 组件配置。"""
    return DeepSeekV4ModelConfig(
        vocab_size=256, dim=128, n_layers=4, n_heads=4, head_dim=32,
        max_seq_len=64, window_size=8,
        mqa=MQAV4Config(dim=128, n_heads=4, head_dim=32, window_size=8,
                        qk_rope_head_dim=16),
        hca=HCAConfig(compression_factor=4),
        csa=CSAConfig(compression_factor=4,
                      indexer=IndexerConfig(index_n_heads=2, index_head_dim=16,
                                            index_topk=4)),
        **kw,
    )


print(SEP)
print("E-E  全模型 before/after 前向 + 参数量（CPU 真实运行）")
print(SEP)
for tag, kw in (("标准基线(残差+滑窗MQA)", dict(use_mhc=False, use_moe=False,
                                              compress_ratios=[0, 0, 0, 0])),
                ("V4 创新(mHC+CSA/HCA)", dict(use_mhc=True, use_moe=False,
                                              compress_ratios=[0, 4, 128, 4]))):
    c = make_cfg(**kw)
    m = DeepSeekV4LM(c).eval()
    ids = torch.randint(0, 256, (1, 24))
    # 预热
    with torch.no_grad():
        m(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(ids)
    dt = (time.perf_counter() - t0) * 1000
    logits = out[0] if isinstance(out, tuple) else out
    print(f"  {tag:22s} params={nparam(m):>9,}  logits={tuple(logits.shape)}  "
          f"fwd={dt:6.2f}ms")
print("  测量条件：torch 2.13 CPU, dim=128, n_layers=4, batch=1, seq=24")
print()

print(SEP)
print("E-E2  同配置下 mHC 带来的参数量（n_hc=4 vs 退化标准残差）")
print(SEP)
c_on = make_cfg(use_mhc=True, use_moe=False, compress_ratios=[0, 0, 0, 0])
c_off = make_cfg(use_mhc=False, use_moe=False, compress_ratios=[0, 0, 0, 0])
m_on, m_off = DeepSeekV4LM(c_on), DeepSeekV4LM(c_off)
print(f"  use_mhc=True  total params = {nparam(m_on):,}")
print(f"  use_mhc=False total params = {nparam(m_off):,}")
print(f"  mHC 增加      delta        = {nparam(m_on) - nparam(m_off):,}")
print()

print(SEP)
print("E-F  Sinkhorn 迭代收敛（双随机误差, 与探针05/图对齐）")
print(SEP)
from models.llm.deepseek_v4.modules.mhc import sinkhorn

logits = torch.randn(4, 4) * 2
for it in (1, 2, 4, 8, 12, 20, 32):
    M = sinkhorn(logits, n_iters=it)
    re = (M.sum(-1) - 1).abs().max().item()
    ce = (M.sum(-2) - 1).abs().max().item()
    print(f"  iters={it:>3}  行和误差={re:.3e}  列和误差={ce:.3e}")
print("  测量条件：4x4 随机 logits(std=2), float32, CPU")

print()
print(SEP)
print("E-B  逆 RoPE 两种语义对照（论文 2.3.3 vs 本项目实现, 平移不变性）")
print(SEP)
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention

cfg = MQAV4Config(dim=512, n_heads=8, head_dim=64, q_lora_rank=64,
                  window_size=16, qk_rope_head_dim=16)
mqa = MQAV4Attention(cfg)
x = torch.randn(1, 12, 512)


def variant(kind, pos):
    B, T, _ = x.shape
    q = mqa.q_b_proj(mqa.q_a_norm(mqa.q_a_proj(x)))
    q = q.view(B, T, mqa.n_heads, mqa.head_dim).transpose(1, 2)
    kv = mqa.kv_norm(mqa.kv_proj(x)).unsqueeze(1)
    p = pos.unsqueeze(0).expand(B, -1)
    cos, sin = mqa._rope_cos_sin(int(p.max()) + 1, x.device)
    c, s = cos[p[0]], sin[p[0]]
    q = mqa._apply_partial_rope(q, c, s)
    k = mqa._apply_partial_rope(kv, c, s)
    v = kv if kind == "repo" else k
    mask = mqa._local_mask(p, p, None)
    ctx = mqa._core_attention(q, k.expand(B, mqa.n_heads, -1, mqa.head_dim),
                              v.expand(B, mqa.n_heads, -1, mqa.head_dim), mask)
    if kind == "paper":
        ctx = mqa._apply_partial_rope(ctx, c, s, negate_sin=True)
    return mqa.o_b_proj(mqa.o_a_proj(ctx.transpose(1, 2).reshape(B, T, mqa.attention_width)))


p1, p2 = torch.arange(12), torch.arange(12) + 1000
with torch.no_grad():
    r1, r2 = variant("repo", p1), variant("repo", p2)
    a1, a2 = variant("paper", p1), variant("paper", p2)
    print(f"  本项目(V 不旋转, 无逆RoPE) 平移不变性 = {(r1 - r2).abs().max():.3e}")
    print(f"  论文(V==K 旋转, 输出逆RoPE) 平移不变性 = {(a1 - a2).abs().max():.3e}")
    print(f"  两种语义输出差异 max|.| = {(r1 - a1).abs().max():.3e}")
    print(f"  两种语义输出相对差异 = {((r1 - a1).norm() / a1.norm()) * 100:.1f}%")
    print("  注：相对差异随配置变化（dim=64 曾测约 51%，dim=512 此脚本约 19~25%），"
          "恒非零。两者都保持平移不变，属实现选择差异")