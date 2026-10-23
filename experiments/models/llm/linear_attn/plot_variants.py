# -*- coding: utf-8 -*-
"""生成对比图: 6 种线性注意力变体的状态检索误差随时间演化。

对应 Analysis.md Phase 4.8 / 5.1。
验证点:
- 纯加性: 误差随 token 数递增(混叠)
- Delta Rule 类变体: 能收敛/修正到最新值
- KDA: 带衰减+门控
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font
setup_cjk_font()

from models.llm.linear_attn.variants import (
    linear_attn_step, delta_net_step, gla_step,
    gated_delta_net_step, kimi_linear_step,
    elu_plus_one,
)

OUT = pathlib.Path(__file__).resolve().parent


def retrieval_error_over_time() -> None:
    """构造 '重复相同 key、交替 value' 序列, 测每种变体对最新值的检索误差。"""
    B, H, Dk, Dv, T = 1, 1, 8, 4, 40
    torch.manual_seed(0)

    # 目标 key (固定, 模拟重复出现的概念)
    k = torch.randn(B, H, Dk)
    k = k / k.norm(dim=-1, keepdim=True)

    # 目标 value 随时间交替 (最新值才是"正确答案")
    target_v = torch.zeros(B, H, Dv)
    errors = {"Linear(纯加性)": [], "DeltaNet": [], "GatedDeltaNet": [], "KDA v1": []}

    S = {name: torch.zeros(B, H, Dk, Dv) for name in errors}

    for t in range(T):
        # 每 5 步换一次目标 value
        v_true = torch.zeros(B, H, Dv)
        v_true[..., 0] = 1.0 if (t // 5) % 2 == 0 else -1.0

        q = torch.randn(B, H, Dk)
        gate = torch.randn(B, H, Dk) * 0.5
        beta_logit = torch.randn(B, H)
        A_log = torch.ones(H) * 0.1

        # 各变体单步
        _, S["Linear(纯加性)"] = linear_attn_step(S["Linear(纯加性)"], q, k, v_true)
        _, S["DeltaNet"], _, _ = delta_net_step(S["DeltaNet"], q, k, v_true)
        _, S["GatedDeltaNet"], _, _ = gated_delta_net_step(
            S["GatedDeltaNet"], q, k, v_true, gate, beta_logit)
        _, S["KDA v1"], _, _ = kimi_linear_step(
            S["KDA v1"], q, k, v_true, gate, beta_logit, A_log)

        # 检索误差: 用目标 key 查询, 与最新正确值比较
        for name in errors:
            S_cur = S[name]
            retrieved = torch.einsum('bhd,bhdv->bhv', k, S_cur)
            err = (retrieved[..., 0] - v_true[..., 0]).abs().item()
            errors[name].append(err)

    # 绘图
    plt.figure(figsize=(8, 5))
    for name, errs in errors.items():
        plt.plot(range(T), errs, label=name, linewidth=2)
    plt.axvline(19.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(20, plt.ylim()[0], '目标值切换', fontsize=8, color='gray')
    plt.xlabel('Token 步数')
    plt.ylabel('检索误差 |S^T k - v_true|')
    plt.title('各变体对最新值的检索误差演化')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_variants.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_variants.png")
    for name, errs in errors.items():
        print(f"  {name:>16}: 末步误差 = {errs[-1]:.3f}")


if __name__ == "__main__":
    retrieval_error_over_time()
