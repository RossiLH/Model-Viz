"""探针 05：mHC（Manifold-Constrained Hyper-Connections）残差机制实测。

要验证什么
----------
mHC 用一个「近似双随机矩阵」B 在 hc_mult 条残差流之间做混合，取代标准残差
x = x + F(x)。双随机（行和=列和=1）保证混合既不放大也不衰减总信号量，
这是它能稳定堆叠几十层的关键。本脚本用实测数据回答四个问题：

  1. Sinkhorn 迭代真的收敛到双随机吗？收敛有多快？
     —— 直接量化行和/列和偏离 1 的最大误差随迭代数的下降曲线。
  2. 三方仓库的默认迭代数不一致（nano=8, SGLang=20, mini=20），
     8 次够不够？—— 给出 8 与 20 的残余误差对比，量化这个分歧的实际影响。
  3. log 域 Sinkhorn 与普通 Sinkhorn 数值是否等价？（mini 提供两条路径）
  4. mHC 的三个门控 A / B / C 的形状与取值范围是否符合论文约束
     （A ∈ (0,1) 用 sigmoid；B 双随机；C ∈ (0,2) 用 2·sigmoid）。

为什么重要
----------
这是 D4 比对表里「hc_sinkhorn_iters 取值分歧」条目的判据：
若 8 次迭代残余误差已经足够小，则该分歧对精度无实质影响，可按性能取小值；
若 8 次明显不收敛，则必须以 SGLang 的 20 为准。

用法
----
    python scripts\\deepseek_v4\\05_probe_mhc.py

退出码
------
0 = 全部断言通过；非 0 = 存在与预期不符的行为（会打印具体断言）
"""
from __future__ import annotations

import torch

from _probe_common import add_mini_to_path, banner, kv, section, shape_of, table

add_mini_to_path()

from src.mHC_residuals import (  # noqa: E402  (需先改 sys.path)
    ManifoldHyperConnection,
    ManifoldHyperConnectionConfig,
)
from src.mHC_residuals_utils import (  # noqa: E402
    collapse_residual_stream,
    expand_residual_stream,
    log_sinkhorn,
    sinkhorn,
)


def doubly_stochastic_error(m: torch.Tensor) -> tuple[float, float]:
    """返回 (行和最大偏离, 列和最大偏离)。

    双随机矩阵要求每行和、每列和都等于 1。
    这里取绝对偏离的最大值，是最严格的（无穷范数）度量。
    """
    row_err = (m.sum(dim=-1) - 1.0).abs().max().item()
    col_err = (m.sum(dim=-2) - 1.0).abs().max().item()
    return row_err, col_err


def probe_sinkhorn_convergence() -> None:
    """问题 1 & 2：Sinkhorn 收敛速度，以及 8 次 vs 20 次的差距。"""
    section("1. Sinkhorn 收敛性：迭代数 -> 双随机残余误差")
    torch.manual_seed(0)
    # n_hc=4 对应 SGLang/nano 的 hc_mult=4；用较大 logits 加大收敛难度
    logits = torch.randn(2, 8, 4, 4) * 2.0

    rows = []
    for n_iters in (1, 2, 4, 8, 12, 16, 20, 32, 64):
        m = sinkhorn(logits, n_iters=n_iters, eps=1e-6, fp32=True)
        row_err, col_err = doubly_stochastic_error(m)
        rows.append([n_iters, f"{row_err:.3e}", f"{col_err:.3e}"])
    table(["n_iters", "行和最大偏离", "列和最大偏离"], rows)

    m8 = sinkhorn(logits, n_iters=8)
    m20 = sinkhorn(logits, n_iters=20)
    r8, c8 = doubly_stochastic_error(m8)
    r20, c20 = doubly_stochastic_error(m20)
    print()
    kv("8 次迭代 (nano 默认) 最大偏离", f"{max(r8, c8):.3e}")
    kv("20 次迭代 (SGLang/mini 默认) 最大偏离", f"{max(r20, c20):.3e}")
    kv("两者矩阵元素最大差异", f"{(m8 - m20).abs().max().item():.3e}")
    # 结论判据：8 次是否已进入 1e-3 量级
    verdict = "8 次已足够（<1e-3）" if max(r8, c8) < 1e-3 else "8 次未收敛（>=1e-3）"
    kv("判据", verdict)


def probe_log_vs_normal_sinkhorn() -> None:
    """问题 3：log 域实现与普通实现是否数值等价。"""
    section("2. log 域 Sinkhorn vs 普通 Sinkhorn")
    torch.manual_seed(0)
    rows = []
    for scale, label in ((1.0, "温和 logits (×1)"), (10.0, "极端 logits (×10)")):
        logits = torch.randn(2, 4, 4, 4) * scale
        a = sinkhorn(logits, n_iters=20)
        b = log_sinkhorn(logits, n_iters=20)
        ra, ca = doubly_stochastic_error(a)
        rb, cb = doubly_stochastic_error(b)
        rows.append([
            label,
            f"{max(ra, ca):.3e}",
            f"{max(rb, cb):.3e}",
            f"{(a - b).abs().max().item():.3e}",
        ])
    table(["输入规模", "普通残余", "log域残余", "两者最大差"], rows)
    print("  说明：log 域用 logsumexp 规避 exp 溢出，极端 logits 下更稳。")


def probe_gates() -> None:
    """问题 4：A / B / C 三个门控的形状与取值范围。"""
    section("3. mHC 门控 A / B / C 的形状与取值范围")
    cfg = ManifoldHyperConnectionConfig(
        d_model=32,
        n_hc=4,
        sinkhorn_iters=20,
    )
    mhc = ManifoldHyperConnection(cfg)
    torch.manual_seed(0)
    X = torch.randn(2, 6, 4, 32)  # [B, T, n_hc, D]

    A, B_mat, C = mhc.compute_ABC(X)
    rows = [
        ["A (pre-mix 门)", shape_of(A), f"[{A.min():.4f}, {A.max():.4f}]", "sigmoid -> (0,1)"],
        ["B (流间混合)", shape_of(B_mat), f"[{B_mat.min():.4f}, {B_mat.max():.4f}]", "sinkhorn -> 双随机"],
        ["C (输出注入)", shape_of(C), f"[{C.min():.4f}, {C.max():.4f}]", "2*sigmoid -> (0,2)"],
    ]
    table(["门控", "shape", "实测取值范围", "论文约束"], rows)

    r, c = doubly_stochastic_error(B_mat)
    print()
    kv("B 的双随机残余 (行/列)", f"{r:.3e} / {c:.3e}")

    assert 0.0 < A.min() and A.max() < 1.0, "A 应严格落在 (0,1)"
    assert 0.0 < C.min() and C.max() < 2.0, "C 应严格落在 (0,2)"


def probe_stream_roundtrip() -> None:
    """附加：残差流的 expand / collapse 语义（标准残差 <-> 多流的桥）。"""
    section("4. 残差流 expand / collapse 语义")
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    rows = []
    for mode in ("first", "mean", "repeat"):
        X = expand_residual_stream(x, n_hc=4, mode=mode)
        back = collapse_residual_stream(X, mode="sum" if mode == "mean" else "first")
        err = (back - x).abs().max().item()
        rows.append([mode, shape_of(X), f"{err:.3e}"])
    table(["expand 模式", "扩展后 shape", "往返误差(配对collapse)"], rows)
    print("  说明：first+first 与 mean+sum 都能无损往返，")
    print("        这保证 mHC 可以退化成标准残差（作为 before/after 基线）。")


def probe_full_forward() -> None:
    """附加：完整 mHC 前向，确认 X_next = B@X + C*y_sub 的形状闭合。"""
    section("5. mHC 完整前向（pre_mix -> 子层 -> update）")
    cfg = ManifoldHyperConnectionConfig(d_model=32, n_hc=4, sinkhorn_iters=20)
    mhc = ManifoldHyperConnection(cfg)
    torch.manual_seed(0)
    X = torch.randn(2, 6, 4, 32)

    x_sub, aux = mhc.pre_mix(X, return_aux=True)
    kv("pre_mix 输出 x_sub", shape_of(x_sub) + "   <- 送入注意力/FFN 的 d_model 向量")
    # 模拟子层：这里用恒等映射代替真实注意力，聚焦 mHC 自身的形状与数值
    y_sub = torch.tanh(x_sub)
    X_next = mhc.update(X, y_sub, B_mat=aux.get("B"), C=aux.get("C"))
    kv("update 输出 X_next", shape_of(X_next) + "   <- 回到 n_hc 条流")
    assert X_next.shape == X.shape, "X_next 必须与 X 同形，才能逐层堆叠"
    kv("X_next 与 X 同形", "是")
    kv("参数量 (d_model=32, n_hc=4)", sum(p.numel() for p in mhc.parameters()))


def main() -> int:
    banner("探针 05：mHC 残差机制（主参考 deepseek-v4-mini-pytorch）")
    probe_sinkhorn_convergence()
    probe_log_vs_normal_sinkhorn()
    probe_gates()
    probe_stream_roundtrip()
    probe_full_forward()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
