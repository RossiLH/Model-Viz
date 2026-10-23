# -*- coding: utf-8 -*-
"""Stable LatentMoE 教学演示 —— K3 稀疏前馈 + 分位数负载均衡（论文 §2.3, Eqs. 11-14）。

运行:
    python experiments/moe/demo_stable_latent_moe.py
产出:
    experiments/moe/demo_stable_latent_moe.png   (2x2 四联图)
    experiments/moe/demo_stable_latent_moe.json  (关键数值, 供文档引用)

四联图(全部在 CPU 微型配置上运行, 可复现):
  (a) Quantile Balancing 前后负载柱状图 —— 一次 update_bias 把"热专家/冷专家"
      的载荷拉平, 对应论文 §2.3.3 "一步到达目标负载, 无需学习率", 兼附
      max_min_ratio / cv 两个健康指标;
  (b) 分位数估计精确性: update_bias_exact(torch.quantile) vs 直方图分箱估计
      (附录 D), 两者偏置差 ≤ 一个 bin 宽 —— 直方图方法只有微小近似损失;
  (c) SwiGLU vs SiTU-GLU 激活幅值 vs 输入尺度: SwuGLU 发散, SiTU-GLU 被钳在
      beta1*beta2 = 100 内(§2.3.2, Eq. 17);
  (d) up-projection 前的 RMSNorm: 把路由分支的 down 权重放大若干倍,
      无 latent_norm 的输出幅值随之放大, 有 latent_norm 的保持不变
      (对应测试 test_rmsnorm_stable_before_up_projection_removes_scale_sensitivity)。

一次运行产出 PNG + JSON, 并打印各子图关键读数; 文档 Phase 4.8 直接用这些数值
回填。所有代码 CPU 可运行, 无外部依赖(仅 torch/matplotlib)。
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font  # noqa: E402  (experiments/attnres/plot_utils.py)

setup_cjk_font()

from models.llm.kimi_k3.config import MoEConfig  # noqa: E402
from models.llm.kimi_k3.modules.moe import StableLatentMoE  # noqa: E402
from models.llm.kimi_k3.modules.router_qb import QuantileBalancingRouter  # noqa: E402
from models.llm.kimi_k3.modules.situ_glu import SiTUGLU, SwiGLU  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def make_cfg(**kw) -> MoEConfig:
    """微型 MoE 配置(与 tests/math/test_moe.py 使用同一套 shape 约束)。"""
    base = dict(
        n_routed=8, n_shared=2, top_k=2, latent_dim=16,
        expert_hidden=24, shared_hidden=24, qb_bins=256,
    )
    base.update(kw)
    return MoEConfig(**base)


def make_imbalanced_router(dim, n_experts, top_k, hot, bins=256, seed=0):
    """构造一个"随机路由就天然失衡"的 Router: 放大热门专家权重行。

    复现论文 §2.3.3 想解决的 hot-expert 问题：在没有任何 bias 干预前,
    对任意 token 分数的 sigmoid() 对热门专家的值都更大 -> Top-k 路由天然
    集中在 hot 子集。这就是 QB 要矫治的载荷再平衡场景。
    """
    torch.manual_seed(seed)
    r = QuantileBalancingRouter(dim=dim, n_experts=n_experts, top_k=top_k,
                                enabled=True, bins=bins)
    with torch.no_grad():
        for e in hot:
            r.weight.data[e] *= 6.0
    return r


def collect_routing(router, n_batches, batch, dim, seed=0):
    """让 router 连续 n_batches 个随机 batch 做训练态前向, 拼接 scores/cutoffs。

    training 模式会把 histogram/load 累加进 buffer, 这正是 demo (a) 要观测的。
    """
    router.train()
    torch.manual_seed(seed)
    scores_all, cutoffs_all = [], []
    for _ in range(n_batches):
        x = torch.randn(batch, dim)
        out = router(x)
        scores_all.append(out.scores.detach())
        cutoffs_all.append(out.cutoff.detach())
    return torch.cat(scores_all, 0), torch.cat(cutoffs_all, 0)


def demo_quantile_balancing(dim=32, n_experts=8, top_k=2, n_batches=40,
                            batch=200, hot=(0, 1), bins=256):
    """(a)+(b) 分位数均衡的度量与估计精度。

    返回值里同时给出:
      - 均衡前/后的 load 分布与 max/min、cv 指标
      - exact 偏置(torch.quantile 真值)与直方图分箱估计的对比及差值
    """
    r = make_imbalanced_router(dim, n_experts, top_k, list(hot), bins=bins, seed=0)

    # --- 修复前: 收集 loads ---
    scores, cutoffs = collect_routing(r, n_batches, batch, dim, seed=1)
    load_before = [int(v) for v in r.load.tolist()]
    stats_before = r.load_balance_stats()

    # --- exact 偏置: torch.quantile 真值(Eq. 14 在全量坐标上直接做) ---
    r_exact = make_imbalanced_router(dim, n_experts, top_k, list(hot), bins=bins, seed=0)
    exact_bias = r_exact.update_bias_exact(scores, cutoffs)
    exact_arr = [float(v) for v in exact_bias.tolist()]

    # --- histogram 偏置: 同一批数据按 batch 切块喂进直方图(附录 D) ---
    r_hist = make_imbalanced_router(dim, n_experts, top_k, list(hot), bins=bins, seed=0)
    r_hist.histogram.zero_()
    r_hist.token_count.zero_()
    for i in range(0, scores.shape[0], batch):
        s = scores[i:i + batch]
        c = cutoffs[i:i + batch]
        top_idx = torch.topk(s + r_hist.bias, top_k, dim=-1).indices
        r_hist._accumulate(s.detach(), c.detach(), top_idx.detach())
    hist_bias = r_hist.update_bias()
    hist_arr = [float(v) for v in hist_bias.tolist()]

    # --- 修复后: 把 exact bias 写回原始 router, 重新收集(均衡路由) ---
    with torch.no_grad():
        r.bias.copy_(exact_bias)
    r.histogram.zero_()
    r.token_count.zero_()
    r.load.zero_()
    _, _ = collect_routing(r, n_batches, batch, dim, seed=2)
    load_after = [int(v) for v in r.load.tolist()]
    stats_after = r.load_balance_stats()

    err = max(abs(a - b) for a, b in zip(exact_arr, hist_arr))
    return {
        "bias_exact": exact_arr,
        "bias_hist": hist_arr,
        "max_abs_bias_diff": err,
        "load_before": load_before,
        "load_after": load_after,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }


def demo_activation_bound():
    """(c) SwiGLU vs SiTU-GLU 激活幅值随输入尺度的关系。"""
    torch.manual_seed(0)
    situ = SiTUGLU(32, 64, beta_gate=4.0, beta_up=25.0)
    swiglu = SwiGLU(32, 64)
    z = torch.randn(128, 32)
    scales = [0.5, 1.0, 2.0, 4.0, 8.0]
    rows = []
    for sc in scales:
        x = z * sc
        with torch.no_grad():
            rows.append({
                "scale": sc,
                "swiglu": float(swiglu(x).abs().max()),
                "situ": float(situ(x).abs().max()),
            })
    return rows


def demo_rmsnorm_scale_scaling():
    """(d) 放大路由 down 权后, 无/有 latent_norm 的输出幅值比。"""
    torch.manual_seed(0)
    dim, n_exp, n_shared, top_k, latent = 16, 4, 1, 2, 8
    factors = [1.0, 10.0, 50.0]
    rows = []
    for factor in factors:
        cur = {}
        for use_norm in (False, True):
            cfg = make_cfg(n_routed=n_exp, n_shared=n_shared, top_k=top_k,
                           latent_dim=latent)
            moe = StableLatentMoE(dim, cfg).eval()
            with torch.no_grad():
                for p in moe.parameters():
                    p.normal_(std=0.2)
            x = torch.randn(2, 8, dim)
            flat = x.reshape(-1, dim)
            with torch.no_grad():
                lat = moe.down_proj(flat)
                route = moe.router(flat)
                rows_idx = torch.arange(len(flat), device=x.device).repeat_interleave(top_k)
                eo = moe.experts.forward_indexed(lat[rows_idx], route.indices.reshape(-1))
                u = torch.zeros_like(lat)
                u.index_add_(0, rows_idx, eo)
                base = float(moe.up_proj(moe.latent_norm(u) if use_norm else u).abs().max())
                # 放大 down 权 factor 倍, 重走路由分支
                moe.experts.down.mul_(factor)
                lat2 = moe.down_proj(flat)
                route2 = moe.router(flat)
                rows_idx2 = torch.arange(len(flat), device=x.device).repeat_interleave(top_k)
                eo2 = moe.experts.forward_indexed(lat2[rows_idx2], route2.indices.reshape(-1))
                u2 = torch.zeros_like(lat)
                u2.index_add_(0, rows_idx2, eo2)
                scaled = float(moe.up_proj(moe.latent_norm(u2) if use_norm else u2).abs().max())
            cur["ratio_with_norm" if use_norm else "ratio_no_norm"] = scaled / base
        cur["factor"] = factor
        rows.append(cur)
    return rows


def plot_demo_four(
    dim=32,
    n_experts=8,
    top_k=2,
    n_batches=40,
    batch=200,
    hot=(0, 1),
    bins=256,
):
    """计算 + 绘制 Stable LatentMoE 四联图，返回 (fig, text)。

    函数可被 gradio demo 等直接复用；绘图数据来自 demo_quantile_balancing /
    demo_activation_bound / demo_rmsnorm_scale_scaling 三个纯计算函数。
    """
    torch.set_num_threads(2)
    torch.manual_seed(0)

    qb = demo_quantile_balancing(
        dim=dim, n_experts=n_experts, top_k=top_k,
        n_batches=n_batches, batch=batch, hot=hot, bins=bins,
    )
    act = demo_activation_bound()
    rms = demo_rmsnorm_scale_scaling()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) QB 前后负载柱状
    ax = axes[0][0]
    idx = range(len(qb["load_before"]))
    ax.bar([i - 0.2 for i in idx], qb["load_before"], width=0.4, label="before QB")
    ax.bar([i + 0.2 for i in idx], qb["load_after"], width=0.4, label="after QB")
    ax.set_title("(a) Quantile Balancing 前后负载\n"
                 "max/min: %.1f -> %.1f" % (
                     qb["stats_before"]["max_min_ratio"],
                     qb["stats_after"]["max_min_ratio"]))
    ax.set_xlabel("expert id")
    ax.set_ylabel("token 被路由次数")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) exact vs histogram 分位数估计
    ax = axes[0][1]
    ax.plot(range(len(qb["bias_exact"])), qb["bias_exact"], "o-", label="exact 分位数")
    ax.plot(range(len(qb["bias_hist"])), qb["bias_hist"], "s--", label="histogram 估计")
    ax.set_title("(b) 分位数估计: exact vs histogram\nmax diff = %.4f" % qb["max_abs_bias_diff"])
    ax.set_xlabel("expert id")
    ax.set_ylabel("b 偏置值")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) 激活有界性
    ax = axes[1][0]
    ax.semilogx([r["scale"] for r in act], [r["swiglu"] for r in act],
                "o-", label="SwiGLU")
    ax.semilogx([r["scale"] for r in act], [r["situ"] for r in act],
                "s-", label="SiTU-GLU")
    ax.axhline(4.0 * 25.0, color="red", ls="--", alpha=0.6, label="beta1*beta2 = 100")
    ax.set_title("(c) SwiGLU vs SiTU-GLU 激活幅值 vs 输入 scale")
    ax.set_xlabel("输入 scale")
    ax.set_ylabel("max|activation|")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) up 前 RMSNorm 尺度不敏感性
    ax = axes[1][1]
    ax.plot([r["factor"] for r in rms], [r["ratio_no_norm"] for r in rms],
            "o-", label="无 latent_norm")
    ax.plot([r["factor"] for r in rms], [r["ratio_with_norm"] for r in rms],
            "s-", label="有 latent_norm")
    ax.set_title("(d) up 前 RMSNorm 的尺度不敏感性\n(放大 routed down 权重)")
    ax.set_xlabel("down 权放大倍数")
    ax.set_ylabel("输出幅值比 (放大/原始)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    text = (
        "== (a) Quantile Balancing 负载均衡 ==\n"
        "  load before: %s\n"
        "  load after : %s\n"
        "  max_min_ratio: %.3f -> %.3f (cv %.3f -> %.3f)\n"
        "== (b) 分位数估计误差 ==\n"
        "  exact: %s\n"
        "  hist : %s  (max diff = %.4f)\n"
        "== (c) 激活有界性 (最大输入 scale=%g) ==\n"
        "  SwiGLU max = %.1f,  SiTU-GLU max = %.1f\n"
        "== (d) RMSNorm 放大 50x 前后比 ==\n"
        "  无 norm = %.2f x,  有 norm = %.2f x"
        % (
            qb["load_before"], qb["load_after"],
            qb["stats_before"]["max_min_ratio"], qb["stats_after"]["max_min_ratio"],
            qb["stats_before"]["cv"], qb["stats_after"]["cv"],
            [round(v, 4) for v in qb["bias_exact"]],
            [round(v, 4) for v in qb["bias_hist"]], qb["max_abs_bias_diff"],
            act[-1]["scale"], act[-1]["swiglu"], act[-1]["situ"],
            rms[-1]["ratio_no_norm"], rms[-1]["ratio_with_norm"],
        )
    )
    return fig, text


def main() -> None:
    torch.set_num_threads(4)
    torch.manual_seed(0)

    hot = (0, 1)
    qb = demo_quantile_balancing(dim=32, n_experts=8, top_k=2, n_batches=40,
                                 batch=200, hot=hot)
    act = demo_activation_bound()
    rms = demo_rmsnorm_scale_scaling()

    fig, text = plot_demo_four()

    png = OUT / "demo_stable_latent_moe.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("saved", png)

    print()
    print(text)

    payload = {
        "config": {"n_experts": 8, "top_k": 2, "hot": list(hot),
                   "n_batches": 40, "batch": 200},
        "quantile_balancing": qb,
        "activation": act,
        "rmsnorm": rms,
    }
    with open(OUT / "demo_stable_latent_moe.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved", OUT / "demo_stable_latent_moe.json")


if __name__ == "__main__":
    main()