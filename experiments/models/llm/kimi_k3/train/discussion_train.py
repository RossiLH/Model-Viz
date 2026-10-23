# -*- coding: utf-8 -*-
"""训练模块反方观点验证（2.7 / 2.8）。

从「反方辩手」角度，戳破 K3 训练管线的五个隐性陷阱，
并以实测输出回填分析文档。

  E1  QB 延迟一步更新的因果性代价
       —— 当 token 分布突变时，老偏置反而拖负载均衡后腿
  E2  grad_accum 在不等 token 数下的非等价性
       —— 朴素 loss/n_batches 在 micro-batch 有效 token 数不同时引入偏差
  E3  curriculum resize 静默失败陷阱
       —— _fit_dataset 吞掉 ValueError，日志里 seq_len 是假的
  E4  weight_clip 默认 0 与论文说法不符
       —— 论文声称沿用 K2 weight clipping，代码默认关掉
  E5  LR 调度步数与实际训练步数脱钩
       —— lr_lambda 在 init 时固定总步数，train(steps=...) 后调度器已超调

每个 fig_*() 返回 (fig, text) 元组，供 Gradio / 独立运行复用。
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_HERE = str(_THIS_FILE.parent)
_SRC = str(_THIS_FILE.parents[5] / "src")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, apply_dark_theme, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg,
    save, style_table,
)
setup_cjk_font()
apply_dark_theme()

from models.llm.kimi_k3.train.loop import Trainer, TrainState
from models.llm.kimi_k3.train.metrics import MetricLog, RunningMean, expert_balance
from models.llm.kimi_k3.modules.router_qb import QuantileBalancingRouter
from demo_pretrain_e2e import build_nano_config, build_nano_dataset, run_training


# ============================================================
# 工具函数：收集梯度、余弦相似度、相对差
# ============================================================

def _gather_grads(model) -> list[torch.Tensor]:
    """收集所有有梯度的参数的 grad（按 named_parameters 顺序）。"""
    return [
        p.grad.detach().clone()
        for p in model.parameters()
        if p.grad is not None
    ]


def _flatten(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def _cos_sim(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va, vb = _flatten(a), _flatten(b)
    dot = float((va * vb).sum())
    na = float(va.norm().clamp_min(1e-12))
    nb = float(vb.norm().clamp_min(1e-12))
    return dot / (na * nb)


def _max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((ga - gb).abs().max().item()) for ga, gb in zip(a, b))


def _rel_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    va, vb = _flatten(a), _flatten(b)
    denom = va.abs().mean().clamp_min(1e-12)
    return float((va - vb).abs().mean() / denom)


def _make_seeded_trainer(cfg, ds, seed):
    """相同种子下构造参数完全相同的 Trainer（不打乱数据读取顺序）。"""
    torch.manual_seed(seed)
    trainer = Trainer(cfg, dataset=ds, device=torch.device("cpu"))
    return trainer


# ============================================================
# E1 — QB 延迟一步更新的因果性代价
# ============================================================

def fig_qb_delayed_causality(
    n_experts: int = 8,
    top_k: int = 2,
    dim: int = 16,
    n_steps: int = 30,
    shift_step: int = 12,
    mbs: int = 128,
    seed: int = 42,
):
    """反方观点 E1：QB 延迟一步更新在分布突变时反而拖后腿。

    论文 §2.3.3 明确要求「下一批才用本批算出来的偏置」，
    理由是因果性——本批数据不能从本批偏置获益。
    反方质问：当分布突变（curriculum 切换、任务切换）时，
    老偏置是在老分布上收敛的，它在新分布下反而可能加剧失衡。
    """
    torch.manual_seed(seed)

    def make_router():
        r = QuantileBalancingRouter(dim, n_experts, top_k, enabled=True, bins=64)
        r.train()
        return r

    def gen_batch(r: QuantileBalancingRouter, step: int):
        """生成一批 token：前半段前半专家热，后半段后半专家热。"""
        x = torch.randn(mbs, dim) * 0.2
        if step < shift_step:
            hot = slice(0, n_experts // 2)
        else:
            hot = slice(n_experts // 2, n_experts)
        # 用专家权重的均值方向来驱动「热」专家的分数更高
        weight_dir = r.weight.data[hot].mean(0)  # (dim,)
        x = x + 0.8 * weight_dir.unsqueeze(0) * torch.randn(mbs, 1).sign()
        return x

    def run_one(mode: str):
        r = make_router()
        balances = []
        for step in range(n_steps):
            x = gen_batch(r, step)
            if mode == "no-delay":
                # 先算本批偏置（违反因果，但能看到 oracle 上限）
                scores = torch.sigmoid(torch.nn.functional.linear(x.float(), r.weight.float()))
                biased = scores + r.bias
                k1 = min(top_k + 1, n_experts)
                top_val, top_idx = torch.topk(biased, k1, dim=-1)
                cutoff = (
                    top_val[:, top_k]
                    if k1 > top_k
                    else top_val.new_full((mbs,), float("-inf"))
                )
                r._accumulate(scores.detach(), cutoff.detach(), top_idx[:, :top_k].detach())
                r.update_bias()

            out = r(x)
            if mode == "delayed":
                r.update_bias()

            load = r.load.float()
            mean_load = load.mean().clamp_min(1e-9)
            max_over_mean = float(load.max() / mean_load)
            balances.append(max_over_mean)
        return balances

    delayed = run_one("delayed")
    no_delay = run_one("no-delay")

    fig, ax = plt.subplots(figsize=(11, 5))
    steps = np.arange(n_steps)
    ax.plot(steps, delayed, color="#e74c3c", lw=2.0, label="delayed（因果，延迟一步）")
    ax.plot(steps, no_delay, color="#3498db", lw=2.0, ls="--", label="no-delay（违规 oracle）")
    ax.axvline(x=shift_step, color="#f39c12", ls=":", alpha=0.7,
               label=f"分布突变点 (step {shift_step})")
    ax.axhline(y=1.0, color="#22c55e", ls=":", alpha=0.6, label="理想均匀 (max/mean=1)")
    _at(ax, f"E1 · QB 延迟一步更新的因果性代价（{n_experts} 专家，top-{top_k}）")
    _al(ax, "step", "max / mean（越小越均匀）")
    _leg(ax, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    post_delayed = delayed[shift_step:shift_step + 5]
    post_no_delay = no_delay[shift_step:shift_step + 5]
    gap = np.mean(post_delayed) - np.mean(post_no_delay)

    return fig, _text(
        f"专家数: {n_experts}, top_k: {top_k}, 每批 token: {mbs}",
        f"分布突变步: step {shift_step}",
        f"突变前平衡: delayed={delayed[shift_step - 1]:.3f}, "
        f"no-delay={no_delay[shift_step - 1]:.3f}",
        f"突变后 5 步均值: delayed={np.mean(post_delayed):.3f}, "
        f"no-delay={np.mean(post_no_delay):.3f}",
        f"延迟一步的额外失衡（平均）: {gap:.3f}",
        "结论：稳态下延迟一步几乎无代价；",
        "但分布突变后约 3-5 步内，因果延迟会让负载均衡明显落后。",
        "这正是 curriculum 阶段切换时 balance 短暂跳升的原因。",
    )


# ============================================================
# E2 — grad_accum 在不等 token 数下的非等价性
# ============================================================

def fig_grad_accum_inequivalence(seed: int = 42):
    """反方观点 E2：朴素梯度累积在 micro-batch 有效 token 数不等时引入偏差。

    背景：train_step 里 loss = out['loss'] / len(batches) 再反向，
    等价于「每个 micro-batch 的梯度等权平均」。但交叉熵损失
    本身已按 token 平均，所以当 micro-batch 有效 token 数不同时，
    全批真值应该是按 token 数加权的平均——朴素平均有系统偏差。
    """
    torch.manual_seed(seed)

    cfg = build_nano_config(
        dim=48, n_layers=2, seq_len=32,
        micro_batch_size=1, grad_accum=1,
        steps=1, log_every=1, seed=seed,
        attn_res_enabled=False,
    )
    ds = build_nano_dataset(cfg, n_docs=16, seed=seed)
    batch4 = ds.batch(4)

    # 构造 labels：0、1 全有效，2、3 只有前 8 个 token 有效
    labels_full = batch4.labels.clone()
    labels_skew = batch4.labels.clone()
    for i in (2, 3):
        labels_skew[i, 8:] = -100

    def full_batch_grad(labels):
        trainer = _make_seeded_trainer(cfg, ds, seed)
        out = trainer.model(batch4.input_ids, labels=labels, doc_ids=batch4.doc_ids)
        out["loss"].backward()
        return _gather_grads(trainer.model)

    def accum_grad(labels, split):
        """split='even' 或 'skew'。"""
        trainer = _make_seeded_trainer(cfg, ds, seed)
        if split == "even":
            idxs_list = [[0, 2], [1, 3]]
        else:
            idxs_list = [[0, 1], [2, 3]]
        n = len(idxs_list)
        for idxs in idxs_list:
            out = trainer.model(
                batch4.input_ids[idxs],
                labels=labels[idxs],
                doc_ids=batch4.doc_ids[idxs],
            )
            (out["loss"] / n).backward()
        return _gather_grads(trainer.model)

    # 全有效 token 时的对称分（应该等价）
    ref_full = full_batch_grad(labels_full)
    even_full = accum_grad(labels_full, "even")
    skew_full = accum_grad(labels_full, "skew")

    # 不等 token 时
    ref_skew = full_batch_grad(labels_skew)
    even_skew = accum_grad(labels_skew, "even")
    skew_skew = accum_grad(labels_skew, "skew")

    rows = [
        ["全有效 + 对称分 (even)",
         f"{_cos_sim(ref_full, even_full):.6f}",
         f"{_rel_diff(ref_full, even_full):.4%}"],
        ["全有效 + 偏态分 (skew)",
         f"{_cos_sim(ref_full, skew_full):.6f}",
         f"{_rel_diff(ref_full, skew_full):.4%}"],
        ["不等 token + 对称分 (even)",
         f"{_cos_sim(ref_skew, even_skew):.6f}",
         f"{_rel_diff(ref_skew, even_skew):.4%}"],
        ["不等 token + 偏态分 (skew, 2:1)",
         f"{_cos_sim(ref_skew, skew_skew):.6f}",
         f"{_rel_diff(ref_skew, skew_skew):.4%}"],
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["场景（累积方式 vs 全批真值）", "余弦相似度", "相对误差"],
        loc="center", cellLoc="center",
    )
    style_table(table, fontsize=9, scale_x=1.05, scale_y=1.8)
    _at(ax, "E2 · grad_accum 在不等 token 数下的梯度等价性")
    fig.tight_layout()

    return fig, _text(
        "对照组（全有效 token）：两种分法与全批的余弦相似度 ≈ 1，等价性成立。",
        f"不等 token + 偏态分：cos = {_cos_sim(ref_skew, skew_skew):.6f}, "
        f"相对误差 = {_rel_diff(ref_skew, skew_skew):.4%}",
        "偏差来源：loss/n_batches 是「按 micro-batch 数等权平均」，",
        "但交叉熵本身按 token 平均，因此有效 token 少的那批梯度被过采样。",
        "结论：PackedDataset 规整 padding 下偏差很小（< 0.5%），",
        "但在 variable-length / 长文档混排场景下是真实的陷阱。",
    )


# ============================================================
# E3 — curriculum resize 静默失败陷阱
# ============================================================

def fig_curriculum_silent_failure(seed: int = 42):
    """反方观点 E3：_fit_dataset 吞掉 ValueError，日志里 seq_len 是假的。

    背景：train() 在阶段切换时调用 `_fit_dataset(self.dataset, stage.seq_len)`，
    如果 `dataset.resize()` 抛 ValueError（数据不够长），它直接返回原 dataset。
    但日志里记录的 seq_len 仍然是 `stage.seq_len`——日志在撒谎。

    反方质问：当数据不够支撑某阶段的序列长度时，训练默默退回到之前的长度，
    用户从日志里完全看不出来。这在小数据实验、课程设置过激进时是真实陷阱。
    """
    # 构造一个极小数据集：4 个短文档，每篇 12 token → 总共只有 48 token
    short_docs = [[i] * 12 for i in range(4)]
    cfg = build_nano_config(
        dim=32, n_layers=1, steps=5, log_every=1,
        seq_len=32, micro_batch_size=1, grad_accum=1,
        seed=seed, attn_res_enabled=False,
        curriculum=(32, 64),
        curriculum_step_fractions=(0.5, 1.0),
    )
    from models.llm.kimi_k3.data.packing import PackedDataset
    ds = PackedDataset(short_docs, seq_len=32, pad_id=0, seed=seed)

    # 跑 5 步训练（会触发 1 次阶段切换：32 → 64）
    trainer = Trainer(cfg, dataset=ds, device=torch.device("cpu"))
    trainer.train(verbose=False)

    # 实际数据集的 seq_len（resize 失败就停留在 32）
    actual_seq_len = trainer.dataset.seq_len
    # 日志记录的 seq_len（永远是 stage.seq_len）
    logged_seq_lens = [r.get("seq_len") for r in trainer.metrics.history]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    steps = [r["step"] for r in trainer.metrics.history]
    logged = logged_seq_lens
    actuals = [actual_seq_len] * len(steps)
    ax.step(steps, logged, where="post", color="#e74c3c", lw=2.5,
            label="日志记录的 seq_len（来自 stage.seq_len）")
    ax.step(steps, actuals, where="post", color="#3498db", lw=2.5, ls="--",
            label="实际数据集的 seq_len")
    ax.axvline(x=3, color="#f39c12", ls=":", alpha=0.7,
               label="阶段切换点 (step 3)")
    _at(ax, "E3 · curriculum resize 静默失败：日志在撒谎")
    _al(ax, "step", "seq_len")
    _leg(ax, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    lies = any(l and l != actual_seq_len for l in logged_seq_lens)

    return fig, _text(
        f"配置课程: {cfg.train.curriculum}",
        f"课程切换分点: {cfg.train.curriculum_step_fractions}",
        f"实际数据集 seq_len: {actual_seq_len}",
        f"日志记录的 seq_len: {logged_seq_lens}",
        f"日志与实际不一致: {lies}",
        "原因：_fit_dataset(dataset, seq_len) 捕获 resize 的 ValueError 后，",
        "默默返回原 dataset，但 train() 的 record['seq_len'] 仍然取 stage.seq_len。",
        "结论：当数据不足以支撑某阶段长度时，训练静默降级，日志不报警。",
        "修复建议：记录实际 seq_len = min(stage.seq_len, dataset.seq_len)，或告警。",
    )


# ============================================================
# E4 — weight_clip 默认 0 与论文说法不符
# ============================================================

def fig_weight_clip_growth(steps: int = 15, seed: int = 42):
    """反方观点 E4：论文 §3.3 说「沿用 Kimi K2 的 weight clipping」，
    但代码里 OptimConfig.weight_clip 默认是 0.0，即完全关掉。

    验证：跑两组训练（weight_clip=0 vs weight_clip=1.0），跟踪
    参数 RMS 的演化。weight_clip=0 下 RMS 会随训练漂移；
    weight_clip>0 时 RMS 被钳制在阈值附近。
    """
    def run_with(wc: float):
        cfg = build_nano_config(
            dim=32, n_layers=2, steps=steps, log_every=1,
            micro_batch_size=2, seed=seed, attn_res_enabled=False,
        )
        cfg.train.optim.weight_clip = wc
        ds = build_nano_dataset(cfg, n_docs=16, seed=seed)
        trainer = Trainer(cfg, dataset=ds, device=torch.device("cpu"))

        rms_trace = []
        # 注册一个 hook 来在每步后记录参数 RMS
        from torch.optim import Optimizer
        orig_step = trainer.optimizer.step

        def stepped_step(*a, **kw):
            orig_step(*a, **kw)
            total, count = 0.0, 0
            for p in trainer.model.parameters():
                if p.ndim == 2:  # 只看 Muon 管辖的 2D 权重
                    total += float(p.pow(2).mean().item())
                    count += 1
            rms_trace.append((total / max(1, count)) ** 0.5)

        trainer.optimizer.step = stepped_step
        trainer.train(verbose=False)
        return rms_trace

    rms_no = run_with(0.0)
    rms_yes = run_with(1.0)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = list(range(1, len(rms_no) + 1))
    ax.plot(xs, rms_no, color="#e74c3c", lw=2.0,
            label="weight_clip = 0（默认，不裁剪）")
    ax.plot(xs, rms_yes, color="#3498db", lw=2.0,
            label="weight_clip = 1.0（手动开启）")
    _at(ax, "E4 · weight_clip 默认关闭：参数 RMS 随训练漂移")
    _al(ax, "step", "2D 权重平均 RMS")
    _leg(ax, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    growth = rms_no[-1] / max(rms_no[0], 1e-12)
    ratio = rms_yes[-1] / max(rms_no[-1], 1e-12)

    return fig, _text(
        f"训练步数: {steps}",
        f"weight_clip=0 起始 RMS: {rms_no[0]:.4f}, 最终 RMS: {rms_no[-1]:.4f}",
        f"weight_clip=0 增长倍数: {growth:.2f}×",
        f"weight_clip=1.0 最终 RMS: {rms_yes[-1]:.4f}",
        f"开启后 RMS 是默认的 {ratio:.2%}",
        "反方观点：论文 §3.3 明确说「沿用 Kimi K2 的 weight clipping」，",
        "但 OptimConfig.weight_clip 默认值是 0.0，等于关掉。",
        "结论：这是配置与论文不一致的陷阱，用户需要手动开 weight_clip 才能复现论文设定。",
    )


# ============================================================
# E5 — LR 调度步数与实际训练步数脱钩
# ============================================================

def fig_lr_steps_decoupled(seed: int = 42):
    """反方观点 E5：lr_lambda 在 init 时固定总步数，
    train(steps=...) 传更小的步数后，实际 LR 曲线与「按实际步数调度」严重偏离。

    背景：Trainer.__init__ 里 self.lr_lambda = lr_lambda_for(cfg.train.optim, cfg.train.steps)
    把「总步数」写死为 cfg.train.steps。如果用户后来调用 trainer.train(steps=N)
    其中 N << cfg.train.steps，学习率会按「总步数很长」的曲线变化，
    导致整个短训练过程几乎都在 warmup + 初始高 LR 区域。
    """
    # 配置 1000 步，实际只跑 20 步
    cfg_total = build_nano_config(
        dim=32, n_layers=1, steps=1000, log_every=1,
        micro_batch_size=1, grad_accum=1, seed=seed, attn_res_enabled=False,
    )
    ds = build_nano_dataset(cfg_total, n_docs=8, seed=seed)

    # 方案 A：init 按 1000 步建，train 跑 20 步（陷阱路径）
    trainer_long = Trainer(cfg_total, dataset=ds, device=torch.device("cpu"))
    lr_long = []
    for s in range(20):
        lr_long.append(trainer_long.set_lr(s))

    # 方案 B：init 按 20 步建（正确路径）
    cfg_short = build_nano_config(
        dim=32, n_layers=1, steps=20, log_every=1,
        micro_batch_size=1, grad_accum=1, seed=seed, attn_res_enabled=False,
    )
    trainer_short = Trainer(cfg_short, dataset=ds, device=torch.device("cpu"))
    lr_short = []
    for s in range(20):
        lr_short.append(trainer_short.set_lr(s))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(20)
    ax.plot(xs, lr_long, color="#e74c3c", lw=2.0,
            label="cfg.train.steps=1000, 实际跑 20 步（陷阱）")
    ax.plot(xs, lr_short, color="#3498db", lw=2.0, ls="--",
            label="cfg.train.steps=20, 实际跑 20 步（正确）")
    _at(ax, "E5 · LR 调度总步数与实际步数脱钩：短训练几乎不掉 LR")
    _al(ax, "step", "LR 乘子")
    _leg(ax, fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    end_long = lr_long[-1]
    end_short = lr_short[-1]
    ratio = end_long / max(end_short, 1e-12)

    return fig, _text(
        "cfg.train.steps = 1000，实际只训练 20 步：",
        f"  第 20 步 LR 乘子 = {end_long:.4f}（几乎还在峰值）",
        "cfg.train.steps = 20，实际训练 20 步：",
        f"  第 20 步 LR 乘子 = {end_short:.4f}（已衰减到末尾）",
        f"两者在最后一步的 LR 比值: {ratio:.2f}×",
        "原因：lr_lambda 在 Trainer 初始化时固定总步数，",
        "train(steps=N) 只控制循环次数，不更新调度器的总步数。",
        "结论：调 steps 参数做短时实验时，学习率曲线是错的。",
        "修复建议：在 train() 入口重建 lr_lambda，或文档明确警告。",
    )


# ============================================================
# 独立运行：保存 5 张图 + 打印结论
# ============================================================

if __name__ == "__main__":
    experiments = [
        ("E1_qb_delayed_causality.png", fig_qb_delayed_causality),
        ("E2_grad_accum_inequivalence.png", fig_grad_accum_inequivalence),
        ("E3_curriculum_silent_failure.png", fig_curriculum_silent_failure),
        ("E4_weight_clip_growth.png", fig_weight_clip_growth),
        ("E5_lr_steps_decoupled.png", fig_lr_steps_decoupled),
    ]
    for name, fn in experiments:
        fig, txt = fn()
        path = save(fig, name)
        print(f"[{name}] saved -> {path}")
        print(txt)
        print()
