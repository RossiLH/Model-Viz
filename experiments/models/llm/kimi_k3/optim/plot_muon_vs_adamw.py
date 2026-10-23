# -*- coding: utf-8 -*-
"""Muon vs AdamW 端到端训练对比。

在简单 Transformer 上对比 Muon 与 AdamW 的训练效果。
数据来自真实模型训练，非模拟。
"""
from __future__ import annotations

from plot_utils import add_src_to_path
add_src_to_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.optim.muon import Muon
from models.llm.kimi_k3.optim.lr_scheduler import CosineWarmupLR


class SimpleTransformer(nn.Module):
    """用于 demo 的简单 Transformer。"""

    def __init__(self, vocab_size=256, d_model=64, nhead=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(128, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x) + self.pos_embed(pos)
        h = self.transformer(h)
        return self.head(h)


def run_training(optimizer_type="muon", num_steps=200, lr_muon=0.002, lr_adamw=0.0006,
                 weight_decay=0.01, seed=42, d_model=64, nhead=4, n_layers=2,
                 vocab_size=256, batch_size=4, seq_len=32):
    """运行一次训练，返回 loss 历史和一些额外指标。"""
    torch.manual_seed(seed)
    model = SimpleTransformer(
        vocab_size=vocab_size, d_model=d_model, nhead=nhead, n_layers=n_layers,
    )

    if optimizer_type == "muon":
        # 手动构建参数分组（SimpleTransformer 不是 K3 模型，
        # build_muon_param_groups 的 _head_count 无法识别 nn.TransformerEncoderLayer）
        muon_params = []      # 普通 Muon (o_proj, gate_proj, up_proj, down_proj)
        per_head_params = []  # Per-Head Muon (q/k/v_proj)
        adamw_params = []     # AdamW (embedding, bias, norm, output)

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim != 2 or "embed" in name or "head" in name:
                # 1D 参数、embedding、输出层 → AdamW
                adamw_params.append(p)
            elif "self_attn.in_proj_weight" in name:
                # q/k/v 投影合并为一个权重，Per-Head 需要拆分
                # nn.TransformerEncoderLayer 的 in_proj_weight 是 [3*head_dim, in_dim]
                # 但 build_muon_param_groups 无法识别它，所以用普通 Muon
                muon_params.append(p)
            else:
                # o_proj, linear1, linear2 → 普通 Muon
                muon_params.append(p)

        param_groups = []
        if muon_params:
            param_groups.append({"params": muon_params, "use_muon": True, "num_heads": None})
        if per_head_params:
            param_groups.append({"params": per_head_params, "use_muon": True, "num_heads": nhead})
        if adamw_params:
            param_groups.append({"params": adamw_params, "use_muon": False, "num_heads": None})

        optimizer = Muon(
            param_groups, lr=lr_muon, adamw_lr=lr_adamw,
            momentum=0.95, nesterov=True, weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr_adamw, betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )

    scheduler = CosineWarmupLR(
        optimizer, total_steps=num_steps, warmup_steps=max(1, num_steps // 20),
    )

    losses = []
    grad_norms = []
    w_norms = []
    lrs = []

    model.train()
    for step in range(num_steps):
        optimizer.zero_grad()
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits = model(x[:, :-1])
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1),
        )
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_norms.append(float(gn))
        optimizer.step()
        scheduler.step()
        losses.append(float(loss))

        total_w = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5
        w_norms.append(total_w)
        lrs.append(scheduler.get_lr_dict().get("lr", 0.0)
                   if "lr" in scheduler.get_lr_dict()
                   else scheduler.get_lr_dict().get("muon_lr", 0.0))

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "w_norms": w_norms,
        "lrs": lrs,
    }


def _smooth(vals, w=10):
    return [sum(vals[max(0, i - w):i + 1]) / min(i + 1, w)
            for i in range(len(vals))]


def fig_muon_vs_adamw(num_steps=200, seed=42, d_model=64):
    """Muon vs AdamW 训练对比图 (1x2 布局)。"""
    setup_cjk_font()
    fp = get_font_prop()

    muon = run_training("muon", num_steps=num_steps, seed=seed, d_model=d_model)
    adamw = run_training("adamw", num_steps=num_steps, seed=seed, d_model=d_model)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    ax1.plot(muon["losses"], alpha=0.3, color="#2196F3")
    ax1.plot(adamw["losses"], alpha=0.3, color="#F44336")
    ax1.plot(_smooth(muon["losses"]), color="#2196F3", lw=2.5, label="Muon")
    ax1.plot(_smooth(adamw["losses"]), color="#F44336", lw=2.5, label="AdamW")
    _at(ax1, "训练 Loss 对比")
    _al(ax1, "Step", "Loss")
    _leg(ax1, fontsize=10)
    ax1.grid(True, alpha=0.3)

    window = 20
    muon_final = sum(muon["losses"][-window:]) / window
    adamw_final = sum(adamw["losses"][-window:]) / window
    improvement = (1 - muon_final / adamw_final) * 100

    labels = ["Muon", "AdamW"]
    values = [muon_final, adamw_final]
    colors = ["#2196F3", "#F44336"]
    bars = ax2.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", fontproperties=fp)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontproperties=fp)
    _at(ax2, f"最终 Loss 对比 (改善 {improvement:.1f}%)")
    _al(ax2, "", f"Loss (最后 {window} 步平均)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Muon vs AdamW 端到端训练对比 ({num_steps} 步, Transformer 2层)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        f"Muon  最终 loss: {muon_final:.4f}",
        f"AdamW 最终 loss: {adamw_final:.4f}",
        f"相对改善: {improvement:.1f}%",
        f"模型: 2层 Transformer, d_model={d_model}, 4 head",
        "Muon 对 2D 矩阵参数做正交化更新，奇异方向等幅下降。",
    )


if __name__ == "__main__":
    fig, txt = fig_muon_vs_adamw(num_steps=200, seed=42)
    print("saved:", save(fig, "muon_vs_adamw.png"))
    print(txt)
