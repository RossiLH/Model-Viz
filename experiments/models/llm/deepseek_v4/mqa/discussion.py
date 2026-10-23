# -*- coding: utf-8 -*-
"""反方辩手实验：MQA / partial RoPE 的两个失败 Case。

反方 a：MQA 共享 K/V 迫使所有头从同一份 key/value 里"各问各的"，
    当头的语义差异很大时（语法头 vs 语义头 / 颜色头 vs 形状头），
    共享 K/V 就是瓶颈——头越多，分摊越薄。

反方 b：partial RoPE 只旋转最后 rope_head_dim 维，其余 nope 维完全不携带
    位置信息。当任务严重依赖"按位置区分"（多个同名候选出现在不同位置）时，
    rope 维太少就表达不足。

每段都用真实模型代码（MQA V4Attention / StandardMHA），CPU 可跑，
实测数字回填到文档中。返回 (fig, text)，供 Gradio 与文档复用。
"""
from __future__ import annotations

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR
for _ in range(5):
    _ROOT = _ROOT.parent
_SRC = str(_ROOT / "src")
_EXP = str(_ROOT / "experiments")
_DSV4 = str(_ROOT / "experiments" / "deepseek_v4")
for _p in (_SRC, _EXP, _DSV4, str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.core.standard.mha import StandardMHA  # noqa: E402
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention  # noqa: E402
from models.llm.deepseek_v4.config import MQAV4Config  # noqa: E402


# ============================================================
# 实验一：MQA 多视图检索失败 Case
# ============================================================

class MultiViewRetrievalTask:
    """多视图关联检索：每个 token 有 H 个 key-view，每个头负责一个视图。

    序列含 N 个"物品"，每个物品的第 h 个视图对应独立的值向量 v_h。
    query 用第 h 个头去检索"第 h 视图匹配的物品"，读出其 v_h。
    多视图的意义：每个头需要完全不同的 key 语义，共享 K/V 会互相打架。
    """

    def __init__(self, n_views: int, n_items: int, key_dim: int, val_dim: int,
                 vocab_per_view: int = 8):
        self.H = n_views
        self.N = n_items
        self.K = key_dim
        self.V = val_dim
        self.VOCAB = vocab_per_view

    def sample(self, batch: int = 4, seed: int = 0):
        g = torch.Generator()
        g.manual_seed(seed)
        labels = torch.randint(0, self.VOCAB, (batch, self.N, self.H), generator=g)
        embeds = torch.randn(self.H, self.VOCAB, self.K, generator=g)
        vals = torch.randn(batch, self.N, self.H, self.V, generator=g)
        keys = torch.zeros(batch, self.N, self.H, self.K)
        for h in range(self.H):
            keys[:, :, h, :] = embeds[h][labels[:, :, h]]
        keys = keys + 0.05 * torch.randn_like(keys, generator=g)
        q_idx = torch.randint(0, self.N, (batch, self.N, self.H), generator=g)
        queries = torch.gather(keys, 1,
                               q_idx.unsqueeze(-1).expand(-1, -1, -1, self.K))
        targets = torch.gather(vals, 1,
                               q_idx.unsqueeze(-1).expand(-1, -1, -1, self.V))
        return keys, vals, queries, targets


def _make_mha(dim, n_heads, n_kv_heads, head_dim, max_seq_len):
    return StandardMHA(dim=dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                       max_seq_len=max_seq_len)


def _make_mqa(dim, n_heads, head_dim, window_size, rope_head_dim,
              q_lora_rank, max_seq_len):
    cfg = MQAV4Config(
        dim=dim, n_heads=n_heads, n_kv_heads=1,
        head_dim=head_dim, q_lora_rank=q_lora_rank,
        qk_rope_head_dim=rope_head_dim, window_size=window_size,
        max_seq_len=max_seq_len, use_attention_sink=False,
    )
    return MQAV4Attention(cfg)


class RetrievalModel(nn.Module):
    """把注意力层 + 每头读出投影 封装成端到端可训练模型。"""

    def __init__(self, attn, dim, n_heads, val_dim):
        super().__init__()
        self.attn = attn
        self.n_heads = n_heads
        self.readout = nn.Linear(dim, n_heads * val_dim, bias=False)
        self.val_dim = val_dim

    def forward(self, x_ctx, x_q):
        """x_ctx [B,N,dim] 是上下文，x_q [B,Q,dim] 是 query；返回 [B,Q,H,V]。"""
        B = x_ctx.shape[0]
        N = x_ctx.shape[1]
        full = torch.cat([x_ctx, x_q], dim=1)
        out = self.attn(full)
        q_out = out[:, N:, :]
        return self.readout(q_out).view(B, x_q.shape[1], self.n_heads, self.val_dim)


def multi_view_retrieval_sweep(n_views=4, n_items=6, key_dim=16, val_dim=16,
                               batch=8, steps=400, lr=0.003, seed=0):
    """MHA / GQA-2 / MQA 在多视图检索上的训练曲线。

    返回 (curves, finals)：curves 是 name -> list[float]；
    finals 是 name -> {"final_loss", "err_per_head", "n_params"}。
    """
    torch.manual_seed(seed)
    task = MultiViewRetrievalTask(n_views, n_items, key_dim, val_dim)

    dim = n_views * key_dim
    max_seq_len = n_items * 4

    def _make(kind):
        if kind == "MHA":
            a = _make_mha(dim, n_views, n_views, key_dim, max_seq_len)
        elif kind == "GQA-2":
            a = _make_mha(dim, n_views, max(1, n_views // 2), key_dim, max_seq_len)
        elif kind == "MQA":
            a = _make_mqa(dim, n_views, key_dim,
                          window_size=n_items * 2, rope_head_dim=0,
                          q_lora_rank=0, max_seq_len=max_seq_len)
        else:
            raise ValueError(kind)
        return kind, RetrievalModel(a, dim, n_views, val_dim)

    models = [_make(k) for k in ("MHA", "GQA-2", "MQA")]
    curves = {}
    finals = {}

    keys, vals, queries, targets = task.sample(batch=batch, seed=seed + 1)
    # 固定的输入投影（把多视图 key 投到 dim 维共享空间），不参与训练
    torch.manual_seed(seed + 10)
    proj_in = nn.Linear(n_views * key_dim, dim, bias=False)
    with torch.no_grad():
        x_ctx = proj_in(keys.reshape(batch, n_items, n_views * key_dim))
        x_q = proj_in(queries.reshape(batch, n_items, n_views * key_dim))

    for name, model in models:
        torch.manual_seed(seed + 2)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        hist = []
        for _ in range(steps):
            pred = model(x_ctx, x_q)
            loss = F.mse_loss(pred, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist.append(loss.item())
        curves[name] = hist
        with torch.no_grad():
            pred = model(x_ctx, x_q)
            eph = (pred - targets).pow(2).mean(dim=(0, 1, 3)).tolist()
        finals[name] = {
            "final_loss": hist[-1],
            "err_per_head": eph,
            "n_params": sum(p.numel() for p in model.parameters()),
        }

    return curves, finals


def plot_multi_view_retrieval_figure(n_views=4, n_items=6, key_dim=16,
                                     val_dim=16, batch=8, steps=400,
                                     lr=0.003, seed=0):
    """反方 a：多视图检索上 MQA 明显差于 MHA/GQA。返回 (fig, text)。"""
    curves, finals = multi_view_retrieval_sweep(
        n_views=n_views, n_items=n_items, key_dim=key_dim,
        val_dim=val_dim, batch=batch, steps=steps, lr=lr, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes
    colors = {"MHA": "#1565c0", "GQA-2": "#6a1b9a", "MQA": "#c62828"}

    for name, hist in curves.items():
        ax1.plot(hist, label=name, color=colors[name], linewidth=2)
    ax1.set_xlabel("训练步")
    ax1.set_ylabel("MSE loss")
    ax1.set_yscale("log")
    ax1.set_title(f"多视图检索训练曲线（H={n_views} 视图，N={n_items} 物品）")
    ax1.legend()
    ax1.grid(alpha=0.3)

    head_labels = [f"头 {h}" for h in range(n_views)]
    x = np.arange(n_views)
    w = 0.25
    for i, (name, f) in enumerate(finals.items()):
        ax2.bar(x + (i - 1) * w, f["err_per_head"], width=w, label=name,
                color=colors[name], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_labels)
    ax2.set_ylabel("最终 MSE / 头")
    ax2.set_title("逐头误差拆解：MQA 下某些头学得差")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()

    mha = finals["MHA"]["final_loss"]
    gqa = finals["GQA-2"]["final_loss"]
    mqa = finals["MQA"]["final_loss"]
    ratio = mqa / max(mha, 1e-12)
    text = (
        f"反方 a：多视图检索上，MHA 最终 loss={mha:.4f}、GQA-2={gqa:.4f}、"
        f"MQA={mqa:.4f}。MQA 比 MHA 差 {ratio:.1f} 倍。"
        f"原因：MQA 所有头共享同一份 K/V，无法为每个视图学习独立的 key "
        f"语义空间——视图越多，共享瓶颈越明显。GQA-2 取折中：2 组各有"
        f"独立 K/V，每组内 {max(1, n_views // 2)} 个头共享，效果接近 MHA。"
    )
    return fig, text


# ============================================================
# 实验二：partial RoPE 位置分辨率失败 Case
# ============================================================

def _build_mqa_for_rope(dim, n_heads, head_dim, rope_head_dim,
                        window_size, max_seq_len):
    cfg = MQAV4Config(
        dim=dim, n_heads=n_heads, n_kv_heads=1,
        head_dim=head_dim, q_lora_rank=0,
        qk_rope_head_dim=rope_head_dim, window_size=window_size,
        max_seq_len=max_seq_len, use_attention_sink=False,
    )
    return MQAV4Attention(cfg)


def position_discrimination_score(attn, dim, n_heads, head_dim,
                                  seq_len=32, seed=0):
    """度量：把两个相同内容的 token 放在不同位置，query 放在最末，
    看两个 token 的注意力权重差多少。差越大 = 位置分辨率越高。

    思路：构造 key 序列，位置 p1 和 p2 的 key 内容完全相同（即 nope 维
    相同 + rope 维会因位置不同而旋转）。然后在最末放一个 query，
    它指向这两个位置的注意力权重的差 |a_p1 - a_p2| 越大越好。
    如果 rope_dim=0，两个位置内容完全一样，权重差应该接近 0。
    """
    torch.manual_seed(seed)
    B = 2
    x = torch.randn(B, seq_len, dim)
    # 把位置 7 和 位置 15 的隐藏状态设成完全一样
    p1, p2 = 7, 15
    x[:, p2, :] = x[:, p1, :].clone()

    with torch.no_grad():
        # 取注意力权重：手动复现内部计算
        if isinstance(attn, MQAV4Attention):
            q_mid = attn.q_a_norm(attn.q_a_proj(x)) if attn.q_a_proj is not None else x
            q = (attn.q_b_proj(q_mid).view(B, seq_len, attn.n_heads, attn.head_dim)
                 .transpose(1, 2))
            kv = attn.kv_norm(attn.kv_proj(x)).unsqueeze(1)
            cos, sin = attn._rope_cos_sin(seq_len, x.device)
            cos_q, sin_q = cos[:seq_len], sin[:seq_len]
            q = attn._apply_partial_rope(q, cos_q, sin_q)
            k = attn._apply_partial_rope(kv, cos_q, sin_q)
            k_exp = k.expand(B, attn.n_heads, -1, attn.head_dim)
            scale = 1.0 / math.sqrt(attn.head_dim)
            scores = torch.matmul(q.float(), k_exp.transpose(-1, -2).float()) * scale
            causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"),
                                           device=x.device), diagonal=1)
            scores = scores + causal.unsqueeze(0).unsqueeze(0)
            attn_w = F.softmax(scores, dim=-1)  # [B,H,T,T]
        else:
            raise NotImplementedError
    # 看最后一个 query 对 p1 和 p2 的注意力权重差
    # 注意因果：最后一个 query 能看到 p1 和 p2
    w_last = attn_w[:, :, -1, :]  # [B, H, T]
    diff = (w_last[:, :, p1] - w_last[:, :, p2]).abs().mean().item()
    # 同时看 rope 部分 vs nope 部分的 key 在两个位置上的相似度
    return diff, attn_w


def position_discrimination_sweep(dim=64, n_heads=4, head_dim=16,
                                  seq_len=32, rope_dims=None, seed=0):
    """扫 rope_head_dim：0, 2, 4, 8, 16（head_dim）。

    返回 (rope_dims, diffs)：每个 rope_dim 下的位置分辨率得分。
    """
    if rope_dims is None:
        rope_dims = [0, 2, 4, 8, 12, 16]
    # 只保留偶数
    rope_dims = [r for r in rope_dims if r % 2 == 0 and 0 <= r <= head_dim]
    diffs = []
    for r in rope_dims:
        attn = _build_mqa_for_rope(
            dim, n_heads, head_dim, rope_head_dim=r,
            window_size=seq_len, max_seq_len=seq_len)
        d, _ = position_discrimination_score(
            attn, dim, n_heads, head_dim, seq_len=seq_len, seed=seed)
        diffs.append(d)
    return rope_dims, diffs


def plot_rope_resolution_figure(dim=64, n_heads=4, head_dim=16,
                                seq_len=32, rope_dims=None, seed=0):
    """反方 b：partial RoPE 的位置分辨率随 rope 维减少而下降。"""
    rope_dims, diffs = position_discrimination_sweep(
        dim=dim, n_heads=n_heads, head_dim=head_dim,
        seq_len=seq_len, rope_dims=rope_dims, seed=seed)

    # 1×3 布局：左=位置分辨率折线，中=rope_dim=0 注意力矩阵，右=rope_dim=head_dim 注意力矩阵
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax1 = axes[0]

    ax1.plot(rope_dims, diffs, 'o-', color="#c62828", linewidth=2, markersize=8)
    ax1.set_xlabel("rope_head_dim（旋转维度）")
    ax1.set_ylabel("两位置注意力权重差 |a_p1 - a_p2|")
    ax1.set_title(f"位置分辨率 vs rope 维数（head_dim={head_dim}）")
    ax1.grid(alpha=0.3)
    for r, d in zip(rope_dims, diffs):
        ax1.text(r, d + 0.005, f"{d:.3f}", ha="center", fontsize=9,
                 color="#c62828")

    # 中/右图：完整注意力矩阵在 rope=0 和 rope=head_dim 的对比
    samples = {}
    for r in (0, head_dim):
        if r % 2 != 0:
            continue
        attn = _build_mqa_for_rope(
            dim, n_heads, head_dim, rope_head_dim=r,
            window_size=seq_len, max_seq_len=seq_len)
        _, attn_w = position_discrimination_score(
            attn, dim, n_heads, head_dim, seq_len=seq_len, seed=seed)
        samples[r] = attn_w[0, 0].detach().numpy()  # head=0
    vmax = max(s.max() for s in samples.values())
    for ax, (r, w) in zip(axes[1:], samples.items()):
        im = ax.imshow(w, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(f"注意力矩阵（head=0, rope_dim={r}）")
        ax.set_xlabel("key 位置")
        ax.set_ylabel("query 位置")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    max_diff = max(diffs)
    min_diff = min(diffs)
    ratio = max_diff / max(min_diff, 1e-12)
    text = (
        f"反方 b：rope_head_dim 从 0 到 {head_dim}，位置分辨率"
        f"从 {min_diff:.4f} 升到 {max_diff:.4f}，差 {ratio:.1f} 倍。"
        f" rope=0 时，两个相同内容 token 的注意力权重几乎一样"
        f"（nope 维没有位置信息）；只有 rope 部分能区分位置。"
        f" V4 官方只旋转 64/512 维（约 12.5%），代价是 nope 维上"
        f"的语义无法按位置区分——但它换来 nope 维在长上下文下的稳定。"
    )
    return fig, text


# ============================================================
# 实验三：attention sink 的"捷径"代价
# ============================================================

def sink_suppression_sweep(dim=64, n_heads=4, head_dim=16, seq_len=16,
                           sink_values=None, seed=0):
    """扫 attention_sink 值（从 -10 到 +10），看真实 KV 上的总权重、
    输出范数如何变化。sink 越大 → 模型越"放弃检索" → 输出越接近 0。

    返回 (sink_values, total_weights, output_norms)。
    """
    if sink_values is None:
        sink_values = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]

    torch.manual_seed(seed)
    x = torch.randn(2, seq_len, dim)

    cfg = MQAV4Config(
        dim=dim, n_heads=n_heads, n_kv_heads=1,
        head_dim=head_dim, q_lora_rank=0,
        qk_rope_head_dim=0, window_size=seq_len,
        max_seq_len=seq_len, use_attention_sink=True,
    )
    attn = MQAV4Attention(cfg)

    total_weights = []
    output_norms = []
    with torch.no_grad():
        for sv in sink_values:
            # 所有头的 sink 都设为 sv
            attn.attention_sink.data.fill_(sv)
            out = attn(x)
            output_norms.append(float(out.norm(dim=-1).mean()))

            # 手动算权重分配
            q = attn.q_b_proj(x).view(2, seq_len, n_heads, head_dim).transpose(1, 2)
            kv = attn.kv_norm(attn.kv_proj(x)).unsqueeze(1)
            k = kv
            v = kv
            k_exp = k.expand(2, n_heads, -1, head_dim)
            v_exp = v.expand(2, n_heads, -1, head_dim)
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q.float(), k_exp.transpose(-1, -2).float()) * scale
            # sink
            sink = attn.attention_sink.view(1, n_heads, 1, 1).expand(2, n_heads, seq_len, 1)
            scores = torch.cat([scores, sink.float()], dim=-1)
            probs = F.softmax(scores, dim=-1)
            real_weight = probs[..., :-1].sum(dim=-1).mean().item()
            total_weights.append(real_weight)

    return sink_values, total_weights, output_norms


def plot_sink_suppression_figure(dim=64, n_heads=4, head_dim=16,
                                  seq_len=16, sink_values=None, seed=0):
    """反方 c：attention sink 可以是"合法捷径"但也可能"偷懒"。"""
    sink_values, total_weights, output_norms = sink_suppression_sweep(
        dim=dim, n_heads=n_heads, head_dim=head_dim, seq_len=seq_len,
        sink_values=sink_values, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    ax1.plot(sink_values, total_weights, 'o-', color="#ef8c00",
             linewidth=2, markersize=7)
    ax1.axhline(y=1.0, color="#8b949e", ls="--", alpha=0.5)
    ax1.set_xlabel("attention_sink logit 值")
    ax1.set_ylabel("真实 KV 上的总权重")
    ax1.set_title("sink 越大 → 分给真实 KV 的权重越少")
    ax1.grid(alpha=0.3)

    ax2.plot(sink_values, output_norms, 's-', color="#c62828",
             linewidth=2, markersize=7)
    ax2.set_xlabel("attention_sink logit 值")
    ax2.set_ylabel("输出平均范数")
    ax2.set_title("sink 越大 → 输出范数越小（被短路了）")
    ax2.grid(alpha=0.3)

    fig.tight_layout()

    max_w = max(total_weights)
    min_w = min(total_weights)
    text = (
        f"反方 c：attention_sink 从 -10 调到 +10，真实 KV 总权重从"
        f" {max_w:.3f} 降到 {min_w:.3f}，输出范数从 {max(output_norms):.3f}"
        f" 降到 {min(output_norms):.3f}。"
        f" sink 吸收的权重越大，模型输出越接近零——相当于\"注意力层被跳过\"。"
        f" 好处是\"对无关内容合法摆烂\"；代价是模型可能\"偷懒\"："
        f"在难例上把权重丢给 sink、不去学习真正的检索模式。"
    )
    return fig, text


def main():
    out_dir = _THIS_DIR / "assets"
    out_dir.mkdir(exist_ok=True)

    fig1, t1 = plot_multi_view_retrieval_figure()
    fig1.savefig(out_dir / "plot_discussion_multi_view.png", dpi=120)
    print("生成", out_dir / "plot_discussion_multi_view.png")
    print("结论:", t1)
    print()

    fig2, t2 = plot_rope_resolution_figure()
    fig2.savefig(out_dir / "plot_discussion_rope_resolution.png", dpi=120)
    print("生成", out_dir / "plot_discussion_rope_resolution.png")
    print("结论:", t2)
    print()

    fig3, t3 = plot_sink_suppression_figure()
    fig3.savefig(out_dir / "plot_discussion_sink_suppression.png", dpi=120)
    print("生成", out_dir / "plot_discussion_sink_suppression.png")
    print("结论:", t3)


if __name__ == "__main__":
    main()

