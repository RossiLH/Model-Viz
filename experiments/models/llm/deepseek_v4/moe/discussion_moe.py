# -*- coding: utf-8 -*-
"""MoE counter-argument experiments (discussion/debate).

4 counter-experiments, each returns (fig, text):
  1. fig_unbounded_weight_risk — unbounded weight risk?
  2. fig_bias_hurts_semantics — bias hurts semantic routing?
  3. fig_hash_ignores_semantics — hash ignores semantics cost?
  4. fig_topk_small_risk — top-k too small risk?
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from _shared import OUT, setup_fonts  # noqa: E402
from models.llm.deepseek_v4.config import MoEV4Config  # noqa: E402
from models.llm.deepseek_v4.modules.moe import DeepSeekV4MoE  # noqa: E402

setup_fonts()


# ============================================================
#  Counter 1: unbounded weight risk
# ============================================================

def fig_unbounded_weight_risk() -> tuple:
    """Counter: will sqrt(softplus) unbounded scores cause runaway weights?

    Experiment: logits from 0 to 30, compare 'raw top-k score sum'
    vs 'normalized top-k weight sum'.
    Conclusion: after norm_topk_prob, weight sum恒为1 (= scaling_factor).
    Unboundedness only affects selection sharpness, not output magnitude.
    """
    n_experts = 8
    topk = 2
    logit_maxes = list(range(0, 31, 2))

    raw_sums = []
    normed_sums = []
    for lm in logit_maxes:
        # 100 tokens: one expert has logit=lm, others are random small
        torch.manual_seed(lm)
        logits = torch.randn(100, n_experts) * 0.5
        logits[:, 0] = lm  # E0 has the large logit
        scores = torch.sqrt(F.softplus(logits))
        topk_vals, _ = scores.topk(topk, dim=-1)
        raw_sums.append(float(topk_vals.sum(dim=-1).mean()))
        # with normalization
        normed = topk_vals / topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        normed_sums.append(float(normed.sum(dim=-1).mean()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(logit_maxes, raw_sums, 'o-', color='#c62828', linewidth=2,
            markersize=6, label='raw top-k score sum (unbounded)')
    ax.plot(logit_maxes, normed_sums, 's-', color='#238636', linewidth=2,
            markersize=6, label='normed top-k weight sum (always = 1)')
    ax.set_xlabel('max router logit')
    ax.set_ylabel('weight sum per token')
    ax.set_title('unbounded score risk: normalization eliminates it')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()

    text = (
        f'raw sum at logit=0: {raw_sums[0]:.3f}\n'
        f'raw sum at logit=30: {raw_sums[-1]:.3f} (grows ~sqrt(exp(x)/2))\n'
        f'normed sum at logit=0: {normed_sums[0]:.3f}\n'
        f'normed sum at logit=30: {normed_sums[-1]:.3f} (always 1.0)\n'
        'Conclusion: sqrt(softplus) is unbounded, but with norm_topk_prob,'
        '\nthe final weight sum is always 1.0 (times scaling_factor).'
        '\nUnboundedness only makes selection sharper, not output larger.'
    )
    return fig, text


# ============================================================
#  Counter 2: bias hurts semantics
# ============================================================

def fig_bias_hurts_semantics(e0_advantage: float = 3.0) -> tuple:
    """Counter: bias pushes tokens to bad experts, sacrificing semantics.

    Experiment: E0 is clearly best (+e0_advantage). Gradually increase
    E3 bias. Measure 'E0 eviction rate' (tokens where E0 was in top-2
    originally but got pushed out by biased E3) and 'score gap of
    evicted tokens'.
    Conclusion: with small bias, only marginal tokens are evicted;
    semantic loss is controllable.
    """
    torch.manual_seed(42)
    n_tokens = 500
    n_experts = 8

    logits = torch.randn(n_tokens, n_experts) * 0.5
    logits[:, 0] += e0_advantage
    scores = torch.sqrt(F.softplus(logits))

    # baseline: E0 in top-k?
    base_topk_idx = scores.topk(2, dim=-1).indices
    e0_in_base = (base_topk_idx == 0).any(dim=-1)

    bias_values = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    evict_rates = []
    evict_gaps = []

    for b in bias_values:
        selection = scores.clone()
        selection[:, 3] += b
        topk_idx = selection.topk(2, dim=-1).indices
        e0_in_new = (topk_idx == 0).any(dim=-1)
        # evicted: E0 was in top-k originally but not anymore
        evicted = e0_in_base & (~e0_in_new)
        evict_rates.append(float(evicted.float().mean()))
        # score gap: how much better was E0 than E3 for evicted tokens
        if evicted.any():
            gap = scores[evicted, 0] - scores[evicted, 3]
            evict_gaps.append(float(gap.mean()))
        else:
            evict_gaps.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(bias_values, [r * 100 for r in evict_rates],
             'o-', color='#c62828', linewidth=2, markersize=7)
    ax1.set_xlabel('E3 bias magnitude')
    ax1.set_ylabel('E0 eviction rate (%)')
    ax1.set_title('left: small bias -> few tokens evicted')
    ax1.grid(alpha=0.3)

    # only plot gaps where eviction happened
    valid_bias = [b for b, r in zip(bias_values, evict_rates) if r > 0]
    valid_gaps = [g for g, r in zip(evict_gaps, evict_rates) if r > 0]
    ax2.plot(valid_bias, valid_gaps,
             's-', color='#ef8c00', linewidth=2, markersize=7)
    ax2.set_xlabel('E3 bias magnitude')
    ax2.set_ylabel('avg score gap (E0 - E3) of evicted')
    ax2.set_title('right: evicted tokens are mostly marginal')
    ax2.grid(alpha=0.3)

    fig.suptitle(f'bias semantic cost (E0 advantage = {e0_advantage})')
    fig.tight_layout()

    def _g(i):
        return evict_gaps[i] if evict_rates[i] > 0 else float('nan')
    text = (
        f'E0 advantage: {e0_advantage}\\n'
        f'bias=0.5: evict rate = {evict_rates[2]*100:.1f}%, '
        f'avg gap = {_g(2):.3f}\\n'
        f'bias=2.0: evict rate = {evict_rates[5]*100:.1f}%, '
        f'avg gap = {_g(5):.3f}\\n'
        f'bias=5.0: evict rate = {evict_rates[-1]*100:.1f}%, '
        f'avg gap = {_g(-1):.3f}\\n'
        'Conclusion: with reasonable bias, only marginal tokens get evicted.'
        '\\nSemantic loss is small and controllable; aux-loss-free bias'
        '\\nself-stabilizes at small values (unlike large manual bias here).'
    )
    return fig, text


# ============================================================
#  Counter 3: hash ignores semantics
# ============================================================

def fig_hash_ignores_semantics() -> tuple:
    """Counter: hash routing completely ignores semantics.

    Experiment: two semantic clusters of tokens (200 each).
    Compare hash routing vs learned routing 'expert purity'
    (fraction of majority class per expert, averaged).
    Conclusion: hash purity ~ random; learned routing has higher purity,
    but hash has perfect load balance.
    """
    torch.manual_seed(42)
    n_per_cluster = 200
    n_experts = 8
    topk = 2

    # Two semantic clusters: cluster A and cluster B
    # Cluster A: experts 0-3 have higher scores
    # Cluster B: experts 4-7 have higher scores
    logits_a = torch.randn(n_per_cluster, n_experts) * 0.5
    logits_a[:, :4] += 2.0  # cluster A prefers E0-E3
    logits_b = torch.randn(n_per_cluster, n_experts) * 0.5
    logits_b[:, 4:] += 2.0  # cluster B prefers E4-E7

    all_logits = torch.cat([logits_a, logits_b], dim=0)
    labels = torch.cat([torch.zeros(n_per_cluster), torch.ones(n_per_cluster)])

    # learned routing: top-k by score
    scores = torch.sqrt(F.softplus(all_logits))
    learned_topk = scores.topk(topk, dim=-1).indices

    # hash routing: deterministic by token index (simulating hash by id)
    token_ids = torch.arange(len(all_logits))
    hash_topk = torch.zeros(len(all_logits), topk, dtype=torch.long)
    for r in range(topk):
        hash_topk[:, r] = (token_ids + r * 3) % n_experts

    def compute_purity(topk_idx, labels, n_experts):
        # For each expert, compute fraction of majority label among tokens assigned to it
        expert_counts = []
        for e in range(n_experts):
            mask = (topk_idx == e).any(dim=-1)
            if mask.sum() == 0:
                expert_counts.append(0.0)
                continue
            cluster_0 = (labels[mask] == 0).float().mean()
            cluster_1 = (labels[mask] == 1).float().mean()
            expert_counts.append(float(max(cluster_0, cluster_1)))
        return sum(expert_counts) / len(expert_counts)

    hash_purity = compute_purity(hash_topk, labels, n_experts)
    learned_purity = compute_purity(learned_topk, labels, n_experts)

    # Load balance
    def load_balance(topk_idx, n_experts):
        counts = torch.bincount(topk_idx.reshape(-1), minlength=n_experts).float()
        return float(counts.std() / counts.mean())

    hash_cv = load_balance(hash_topk, n_experts)
    learned_cv = load_balance(learned_topk, n_experts)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    methods = ['hash routing', 'learned routing']
    purities = [hash_purity, learned_purity]
    colors = ['#1565c0', '#6a1b9a']
    bars = ax1.bar(methods, purities, color=colors, alpha=0.85)
    for bar, p in zip(bars, purities):
        ax1.text(bar.get_x() + bar.get_width()/2, p + 0.01,
                 f'{p:.3f}', ha='center', fontsize=11)
    ax1.set_ylabel('expert purity (avg majority fraction)')
    ax1.set_title('left: semantic purity')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(alpha=0.3, axis='y')

    cvs = [hash_cv, learned_cv]
    bars2 = ax2.bar(methods, cvs, color=colors, alpha=0.85)
    for bar, c in zip(bars2, cvs):
        ax2.text(bar.get_x() + bar.get_width()/2, c + 0.01,
                 f'{c:.3f}', ha='center', fontsize=11)
    ax2.set_ylabel('load CV (std/mean, lower=better)')
    ax2.set_title('right: load balance')
    ax2.grid(alpha=0.3, axis='y')

    fig.suptitle('hash vs learned: tradeoff between purity and balance')
    fig.tight_layout()

    text = (
        f'hash purity: {hash_purity:.3f} (close to random 0.5)\n'
        f'learned purity: {learned_purity:.3f}\n'
        f'hash load CV: {hash_cv:.3f} (perfect balance)\n'
        f'learned load CV: {learned_cv:.3f}\n'
        'Conclusion: hash routing sacrifices semantic purity (first layers'
        '\nfeature are still token-id-like anyway) for perfect balance.'
        '\nLearned routing has better purity but may be imbalanced at init.'
    )
    return fig, text


# ============================================================
#  Counter 4: top-k too small risk
# ============================================================

def fig_topk_small_risk() -> tuple:
    """Counter: top-k=2 is too small, routing errors are costly.

    Experiment: true expert quality has noise (router isn't perfect).
    Reduce top-k from 8 to 1, measure 'fraction of tokens where the
    true best expert is among the selected top-k'.
    Conclusion: smaller top-k -> higher chance of missing the best expert.
    top-k=2 is a tradeoff between efficiency and safety.
    """
    torch.manual_seed(42)
    n_tokens = 2000
    n_experts = 8

    # true expert quality (unknown to router)
    true_quality = torch.randn(n_tokens, n_experts) * 1.0
    # router observes noisy version of quality
    router_noise = torch.randn(n_tokens, n_experts) * 0.8
    router_scores = true_quality + router_noise

    best_expert = true_quality.argmax(dim=-1)

    topk_values = list(range(1, 9))
    hit_rates = []
    for k in topk_values:
        topk_idx = router_scores.topk(k, dim=-1).indices
        hits = (topk_idx == best_expert.unsqueeze(-1)).any(dim=-1)
        hit_rates.append(float(hits.float().mean()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(topk_values, [r * 100 for r in hit_rates],
            'o-', color='#1565c0', linewidth=2, markersize=8)
    for k, r in zip(topk_values, hit_rates):
        ax.annotate(f'{r*100:.1f}%', (k, r * 100),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=9)
    ax.axvline(2, color='#c62828', ls='--', alpha=0.8, label='DS-V4 top-k=2')
    ax.set_xlabel('top-k')
    ax.set_ylabel('true best expert included (%)')
    ax.set_title('top-k vs best-expert hit rate (noisy router estimate)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xticks(topk_values)
    plt.tight_layout()

    text = (
        f'top-k=1: hit rate = {hit_rates[0]*100:.1f}%\n'
        f'top-k=2: hit rate = {hit_rates[1]*100:.1f}%\n'
        f'top-k=4: hit rate = {hit_rates[3]*100:.1f}%\n'
        f'top-k=8: hit rate = {hit_rates[7]*100:.1f}% (all)\n'
        'Conclusion: smaller top-k means higher chance of missing the true best expert.'
        '\ntop-k=2 is an efficiency-safety tradeoff: saves compute vs top-k=6/8,'
        '\nbut trained routers are much better than random noise — real gap is smaller.'
    )
    return fig, text


# ============================================================
#  main
# ============================================================

def main() -> None:
    for name, fn in [
        ("plot_moe_counter_unbounded.png", fig_unbounded_weight_risk),
        ("plot_moe_counter_bias_semantics.png", fig_bias_hurts_semantics),
        ("plot_moe_counter_hash_semantics.png", fig_hash_ignores_semantics),
        ("plot_moe_counter_topk_small.png", fig_topk_small_risk),
    ]:
        fig, text = fn()
        fig.savefig(OUT / name, dpi=120)
        plt.close(fig)
        print(f"saved {name}")
        print(text)
        print()


if __name__ == "__main__":
    main()
