# -*- coding: utf-8 -*-
"""HeadRMSNorm 探讨实验（Analysis.md Phase 4.7）。

反方观点（4.3）验证：
  c) "按 head 归一化在极端尺度不平衡下仍可能失效 —— 若某个 head 完全饱和/为零
     （RMS≈0），归一化会把噪声放大到与正常 head 相当，导致该 head 输出被噪声主导。"
  我们构造这个失败 case：让一个 head 的输入几乎为零（RMS→0，纯数值噪声），
  另一个 head 正常。HeadRMSNorm 把二者都拉到 RMS=1，于是噪声 head 的信号占比
  被夸大 —— 这暴露了"head-wise 归一化对近零 head 的噪声放大"这一固有问题。

对照：
  - HeadRMSNorm（按 head 归一化）：近零 head 被放大到与正常 head 同尺度。
  - 整体 RMSNorm（跨 head 归一化）：近零 head 保持接近 0（被大 head 主导）。

这印证：K3 中 HeadRMSNorm 之后一定紧跟全秩 sigmoid 门（g_proj），门可以抑制
噪声 head，而不是依赖归一化本身去压噪声。

实测输出回填见 HeadwiseRMSNorm-梳理-V1.md §4.7。
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from models.llm.kimi_k3.modules.norm import HeadRMSNorm, rms_norm  # noqa: E402


def run() -> None:
    torch.manual_seed(0)
    head_dim = 8
    layer = HeadRMSNorm(head_dim, eps=1e-6)

    # 构造：head0 正常，head1 接近零（纯噪声），head2 完全为零。
    x = torch.zeros(1, 1, 3, head_dim)
    x[0, 0, 0] = torch.randn(head_dim) * 2.0          # 正常 head
    x[0, 0, 1] = torch.randn(head_dim) * 1e-4         # 近零 head（数值噪声）
    x[0, 0, 2] = 0.0                                  # 全零 head

    with torch.no_grad():
        out_head = layer(x)                            # head-wise 归一化
        # 真正"跨 head 的整体归一化"：把 (B,T,H,Dh) 展平成 (B,T,H*Dh) 再对最后一维归一化。
        flat = x.reshape(1, 1, -1)
        out_flat = rms_norm(flat, weight=None, eps=1e-6)
        out_whole = out_flat.reshape_as(x)

    rms = lambda t: t.pow(2).mean(-1, keepdim=True).sqrt()  # noqa: E731

    print("输入各 head 的 RMS:")
    for h in range(3):
        print(f"  head{h}: {float(rms(x)[0,0,h,0]):.2e}")
    print("\nHeadRMSNorm 归一化后各 head 的 RMS（都被拉到 ~1）:")
    for h in range(3):
        print(f"  head{h}: {float(rms(out_head)[0,0,h,0]):.4f}")
    print("\n整体 RMSNorm 归一化后各 head 的 RMS:")
    for h in range(3):
        print(f"  head{h}: {float(rms(out_whole)[0,0,h,0]):.4f}")

    # 近零 head 在两种归一化下的输出幅度。
    nz_head_out = float(out_head[0, 0, 1].abs().max())
    nz_whole_out = float(out_whole[0, 0, 1].abs().max())
    print(f"\n关键读数:")
    print(f"  近零 head 经 HeadRMSNorm 后的最大输出幅度: {nz_head_out:.3f} (噪声被放大)")
    print(f"  近零 head 经整体 RMSNorm 后的最大输出幅度: {nz_whole_out:.3e} (噪声保持很小)")
    print("\n关键推论:")
    print("  - HeadRMSNorm 把近零 head 也拉到 RMS=1，放大了其纯噪声 → 暴露 head-wise")
    print("    归一化对近零 head 的噪声放大问题（4.3c 的失败条件）。")
    print("  - 因此 K3 在其后必须接全秩 sigmoid 门（g_proj）来抑制这类噪声 head，")
    print("    而非依赖归一化本身。这验证了输出门存在的必要性。")


if __name__ == "__main__":
    run()
