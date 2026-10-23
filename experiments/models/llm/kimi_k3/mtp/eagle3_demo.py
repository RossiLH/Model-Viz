# -*- coding: utf-8 -*-
"""EAGLE-3 投机解码端到端 demo（K3 报告 §4.1.4；Analysis.md Phase 4.8/5）。

在 CPU 小模型上跑真实 ``Kimi3ForCausalLM × Eagle3Draft`` 的投机解码循环，
输出**实测**（不再是纯理论）指标：

1. 实测接受率 vs 理论；
2. 每目标前向期望 token 数（``tokens_per_forward``）；
3. Recompute vs CacheRollback 两路径**逐 token 一致** —— 回滚正确性的数值证明；
4. 生成 ``plot_e5_measured_speedup.png``：把实测点叠在理论曲线上。

核心函数：
- ``run_spec_decode()`` — 跑完整投机解码实验，返回 (agree, stats_re, stats_ls, draft_len)
- ``fig_measured_speedup()`` — 返回 (fig, text)，实测点叠在理论曲线上

CPU 运行（torch 1.11）。用法：``python experiments/mtp/eagle3_demo.py``
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[5]  # Model-Viz/
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.config import (  # noqa: E402
    AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, MTPConfig,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM  # noqa: E402
from models.llm.kimi_k3.mtp.eagle3 import (  # noqa: E402
    Eagle3Draft, speedup_from_acceptance,
)
from models.llm.kimi_k3.mtp.speculative import (  # noqa: E402
    SpeculativeDecoder,
)

OUT = HERE

plt.rcParams["figure.facecolor"] = "#17181c"
plt.rcParams["axes.facecolor"] = "#17181c"
plt.rcParams["axes.edgecolor"] = "#888899"
plt.rcParams["axes.labelcolor"] = "#dddddd"
plt.rcParams["xtick.color"] = "#bbbbbb"
plt.rcParams["ytick.color"] = "#bbbbbb"
plt.rcParams["text.color"] = "#e8e8e8"
plt.rcParams["axes.titlecolor"] = "#ffffff"
plt.rcParams["legend.facecolor"] = "#22232a"


def small_cfg(**kw) -> Kimi3Config:
    base = dict(
        dim=64, n_layers=2, n_heads=4, vocab_size=100, n_dense_layers=1,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                      conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0,
                      alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                      expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0,
                      situ_beta_up=1.0, rmsnorm_before_up=True,
                      qb_enabled=False, qb_bins=10, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=False),
    )
    base.update(kw)
    return Kimi3Config(model=Kimi3ModelConfig(**base))


# =====================================================================
# 核心实验：真实投机解码
# =====================================================================

def run_spec_decode(
    seed: int = 1234,
    prompt_len: int = 8,
    max_new_tokens: int = 24,
    draft_len: int = 3,
):
    """跑完整投机解码实验，返回 (agree, stats_re, stats_ls, draft_len)。

    - ``agree``: bool，Recompute 与 CacheRollback 两路径 token 流是否一致
    - ``stats_re``: SpecDecodeStats，greedy 路径统计
    - ``stats_ls``: SpecDecodeStats，lossless 路径统计
    """
    torch.manual_seed(seed)
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"))
    model = Kimi3ForCausalLM(cfg)
    model.eval()
    draft = Eagle3Draft(model.mtp, cfg.model.dim, model.lm_head)

    prompt = torch.randint(0, cfg.model.vocab_size, (1, prompt_len))

    # 路径1：greedy（确定性，用于验证 Recompute vs Rollback 逐 token 一致）
    dec_re = SpeculativeDecoder(model, draft, greedy=True,
                                generator=torch.Generator().manual_seed(seed))
    dec_rb = SpeculativeDecoder(model, draft, greedy=True,
                                generator=torch.Generator().manual_seed(seed))
    out_re, stats_re = dec_re.decode_recompute(prompt, max_new_tokens=max_new_tokens,
                                               draft_len=draft_len)
    out_rb, stats_rb = dec_rb.decode_rollback(prompt, max_new_tokens=max_new_tokens,
                                              draft_len=draft_len)
    agree = torch.equal(out_re, out_rb)

    # 路径2：lossless（真实 EAGLE-3 用例）
    dec_ls = SpeculativeDecoder(model, draft, greedy=False,
                                generator=torch.Generator().manual_seed(seed))
    out_ls, stats_ls = dec_ls.decode_recompute(prompt, max_new_tokens=max_new_tokens,
                                               draft_len=draft_len)

    return agree, stats_re, stats_ls, draft_len


# =====================================================================
# 可视化：实测加速点 vs 理论曲线
# =====================================================================

def fig_measured_speedup(draft_len: int, stats, r_theo: float) -> tuple[plt.Figure, str]:
    """把实测点叠在理论曲线上，返回 (fig, text)。"""
    rates = torch.linspace(0.1, 0.98, 45).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    for L, c in zip([1, 2, 4, 7], ["#f2924c", "#f2c94c", "#238636", "#1565c0"]):
        vals = [speedup_from_acceptance(r, L) for r in rates]
        ax.plot(rates, vals, lw=2, label=f"理论 L={L}", color=c)
    ax.scatter([stats.measured_acceptance], [stats.tokens_per_forward],
               color="#ff5252", zorder=6, s=200, marker="*",
               edgecolors="white", linewidths=1.0,
               label=f"实测 L={draft_len} (r={stats.measured_acceptance:.3f})")
    # 箭头标注框：直接在图上给出 tokens/forward 数值
    ax.annotate(f"tokens/forward = {stats.tokens_per_forward:.2f}",
                xy=(stats.measured_acceptance, stats.tokens_per_forward),
                xytext=(min(stats.measured_acceptance + 0.08, 0.72),
                        stats.tokens_per_forward + 0.8),
                arrowprops=dict(arrowstyle="->", color="#ff5252"), fontsize=10,
                color="#ffd6d6",
                bbox=dict(boxstyle="round,pad=0.35", fc="#331414",
                          ec="#ff5252", alpha=0.92))
    ax.axhline(1.0, color="#888899", ls="--", lw=1, label="无投机基线=1.0")
    ax.set_xlabel("逐 token 接受率 r")
    ax.set_ylabel("每个目标前向期望 token 数")
    ax.set_title("EAGLE-3 实测加速点 vs 理论（red * 为 lossless 实测）")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    text = (
        f"实测接受率 = {stats.measured_acceptance:.4f}，"
        f"tokens/forward = {stats.tokens_per_forward:.3f}（基线 1.0）\n"
        f"理论 E[tokens/target](r={r_theo:.4f}, L={draft_len}) = "
        f"{speedup_from_acceptance(r_theo, draft_len):.3f}\n"
        "实测点与理论接近，但受小模型随机性、草稿质量影响有偏差。"
    )
    return fig, text


# =====================================================================
# 主入口（向后兼容）
# =====================================================================

def main() -> None:
    seed = 1234
    agree, stats_re, stats_ls, draft_len = run_spec_decode(seed=seed)
    r_theo = stats_ls.measured_acceptance

    print("=" * 68)
    print("EAGLE-3 投机解码实测（CPU 小模型）")
    print("=" * 68)
    print(f"两路径 token 流一致（greedy, Recompute vs CacheRollback）: {agree}")
    print("-" * 68)
    print("【greedy 确定性路径】")
    print(f"  实测接受率={stats_re.measured_acceptance:.4f}  "
          f"tokens/forward={stats_re.tokens_per_forward:.3f}")
    print("【lossless 路径】")
    print(f"  实测接受率={stats_ls.measured_acceptance:.4f}  "
          f"tokens/forward={stats_ls.tokens_per_forward:.3f}")
    print(f"  理论 E[tokens/target]（r={r_theo:.4f}, L={draft_len}）"
          f" = {speedup_from_acceptance(r_theo, draft_len):.3f}")
    print(f"  （无投机基线 = 1.0 token/forward）")
    print("=" * 68)

    fig, _ = fig_measured_speedup(draft_len, stats_ls, r_theo)
    fig.savefig(OUT / "plot_e5_measured_speedup.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_e5_measured_speedup.png")

    return agree, stats_re, stats_ls


if __name__ == "__main__":
    main()
