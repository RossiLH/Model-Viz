# -*- coding: utf-8 -*-
"""PI0.5 可视化：prefix KV cache 复用 vs 每步重算 —— 推理耗时对照。

【展示特色】prefix KV cache 是 π0.5 推理实时性的关键工程点：同一份 prefix 语义
             在去噪各步完全不变，因此可以「只前向一次、逐层缓存 K/V、之后每步
             复用」，从而省掉 (num_steps-1) 次 prefix 前向计算。
【展示思路】用「有缓存 vs 朴素每步重算」两种推理策略做 before/after 对照：
             相同模型、相同输入、相同去噪步数，只差 prefix 是否被缓存。
【如何展示】真实跑小型可 CPU 运行的 Pi05FaithfulPolicyModel，对去噪步数
             num_steps∈{3,5,8,10,15,20} 各重复计时取中位数，画两条
             耗时曲线 + 加速比曲线（真实计时，非手工拟合）。

生成一张图 plot_kvcache.png（两个子图）：
   1. 左：耗时 vs 去噪步数 —— 朴素（每步 prefix+suffix 整段重算）vs PI0.5
      （prefix 只前向一次 + KV 缓存复用，之后每步只对 suffix 增量前向）。
   2. 右：加速比 = 朴素耗时 / 缓存耗时 —— 随步数增长，前缀重算被摊薄。

数据来源：真实跑本项目的 Pi05FaithfulPolicyModel 前向（torch.no_grad + 计时）。
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW  # noqa: E402
from models.vla.core.diffusion import flow_matching_scheduler  # noqa: E402

setup_fonts()


def _bench(steps: int, num_reps: int = 5) -> tuple:
    """对给定去噪步数，返回 (t_cache_ms, t_naive_ms) 中位数（真实计时）。"""
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))

    x_t0 = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    timesteps, dt = flow_matching_scheduler(steps)

    times_c, times_n = [], []
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok, pmask = pd["prefix_tokens"], pd["prefix_mask"]
        pl = ptok.shape[1]
        slen = cfg.action_horizon
        suffix_mask = torch.ones(1, slen, dtype=torch.bool)
        # suffix-only 掩码（B, slen, slen），缓存 prefix 自动全可见
        smask = suffix_mask.unsqueeze(2) & suffix_mask.unsqueeze(1)

        for _ in range(num_reps):
            # A：复用 prefix KV cache —— 第 0 步的 1 次 prefix 前向也计入耗时
            #     （对应文档"绿线第 0 步应有向上跳变"，公式 t_cache = T_p + N*T_s）
            t0 = time.perf_counter()
            kv = model.transformer.prefix_forward(ptok)
            x_t = x_t0.clone()
            for t in timesteps:
                ts = torch.full((1,), float(t))
                sd = model.embed_suffix(x_t, ts)
                _, sout, _ = model.transformer(
                    None, sd["suffix_tokens"], sd["timestep_cond"], smask, kv_cache=kv)
                v_t = model.action_out_proj(sout)
                x_t = x_t + dt * v_t
            times_c.append(time.perf_counter() - t0)

            # B：朴素 —— 每步重新拼接 prefix+suffix 整段重算
            t0 = time.perf_counter()
            x_t = x_t0.clone()
            for t in timesteps:
                ts = torch.full((1,), float(t))
                sd = model.embed_suffix(x_t, ts)
                stok = sd["suffix_tokens"]
                from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask
                mask = make_attn_mask(pmask, suffix_mask, pl, slen)
                pout, sout, _ = model.transformer(ptok, stok, sd["timestep_cond"], mask)
                v_t = model.action_out_proj(sout)
                x_t = x_t + dt * v_t
            times_n.append(time.perf_counter() - t0)

    return (np.median(times_c) * 1e3, np.median(times_n) * 1e3)


def main() -> None:
    step_list = [3, 5, 8, 10, 15, 20]
    tc, tn = [], []
    for s in step_list:
        a, b = _bench(s)
        tc.append(a); tn.append(b)
        print(f"  steps={s:2d}: cache={a:7.2f}ms  naive={b:7.2f}ms  speedup={b / a:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.plot(step_list, tn, "o-", color=C_STD, lw=2, label="朴素：每步 prefix+suffix 整段重算")
    ax.plot(step_list, tc, "s-", color=C_NEW, lw=2, label="PI0.5：prefix KV cache 复用（只前向一次）")
    ax.set_xlabel("去噪步数 num_steps")
    ax.set_ylabel("总耗时 [ms]（真实计时取中位数）")
    ax.set_title("推理效率：prefix 只前向一次 + KV 缓存复用 vs 每步整段重算")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ratio = [n / c for n, c in zip(tn, tc)]
    ax2.plot(step_list, ratio, "^-", color=C_NEW, lw=2)
    ax2.axhline(1.0, color=C_STD, lw=1, ls="--")
    ax2.set_xlabel("去噪步数 num_steps")
    ax2.set_ylabel("加速比 = 朴素耗时 / 缓存耗时")
    ax2.set_title(f"加速比 {min(ratio):.2f}x ~ {max(ratio):.2f}x（随步数增长，前缀重算被摊薄）")
    ax2.grid(alpha=0.3)

    fig.suptitle("prefix KV cache 复用：同一语义因掩码 block_ps=0 而缓存安全（一次前向，多步复用）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "plot_kvcache.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_kvcache.png")
    print(f"  speedup 范围=[{min(ratio):.2f}x, {max(ratio):.2f}x]")


if __name__ == "__main__":
    main()