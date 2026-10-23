# -*- coding: utf-8 -*-
"""K3 训练模块交互式演示（Gradio）

基于 `models/llm/kimi_k3/train/` 真实代码运行，所有图均为 CPU 实时计算。

运行:
  cd Model-Viz
  python experiments\models\llm\kimi_k3\train\gradio_train_demo.py
"""
from __future__ import annotations

import os
import sys
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
import gradio as gr

from plot_utils import setup_cjk_font, apply_dark_theme
setup_cjk_font()
apply_dark_theme()

# 复用 plot 脚本的函数（零重复代码）
from plot_train_loop import (
    fig_training_overview, fig_lr_schedules, fig_curriculum,
    fig_grad_accum, fig_checkpoint_roundtrip, fig_evaluate, fig_device_and_gc,
)
from plot_metrics import (
    fig_running_mean, fig_metric_log, fig_expert_balance,
    fig_decay_stats, fig_metrics_no_model,
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="K3 训练模块交互式演示") as demo:
        gr.Markdown(
            """
            # K3 训练模块交互式演示

            基于 `models/llm/kimi_k3/train/` 真实代码运行，所有图均为 CPU 实时计算。

            **12 个实验板块**：
            - **训练过程总览**：loss / LR / grad_norm / tokens 四联图
            - **LR 调度**：Cosine vs WSD 对比
            - **Context Curriculum**：阶段切换与步数预算
            - **梯度累积**：不同 grad_accum 等价性
            - **Checkpoint 验证**：save→load round-trip
            - **评估指标**：eval_loss / ppl
            - **设备与 GC**：设备选择策略、显存-速度权衡
            - **RunningMean**：滑动窗口平滑
            - **MetricLog**：记录 / 平滑 / JSONL
            - **MoE 负载均衡**：Quantile Balancing 效果
            - **KDA alpha 衰减**：训练中衰减因子分布
            - **无模型回退**：模块缺失时的优雅降级
            """
        )

        with gr.Tabs():
            _tab_training_overview()
            _tab_lr_schedules()
            _tab_curriculum()
            _tab_grad_accum()
            _tab_checkpoint()
            _tab_evaluate()
            _tab_device_gc()
            _tab_running_mean()
            _tab_metric_log()
            _tab_expert_balance()
            _tab_decay_stats()
            _tab_metrics_no_model()

    return demo


# ============================================================
# 各个 Tab
# ============================================================

def _tab_training_overview():
    with gr.TabItem("1. 训练过程总览"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                steps = gr.Slider(5, 60, value=30, step=5, label="训练步数")
                lr = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="初始学习率")
                mbs = gr.Slider(1, 8, value=4, step=1, label="micro_batch_size")
                btn = gr.Button("运行训练", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="训练过程四联图")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_training_overview,
                  inputs=[steps, seed, lr, mbs],
                  outputs=[plot, text])


def _tab_lr_schedules():
    with gr.TabItem("2. LR 调度曲线"):
        with gr.Row():
            with gr.Column(scale=1):
                total_steps = gr.Slider(50, 1000, value=200, step=50, label="总步数")
                warmup_ratio = gr.Slider(0.01, 0.2, value=0.05, step=0.01, label="warmup 比例")
                min_lr_ratio = gr.Slider(0.0, 0.5, value=0.1, step=0.01, label="min_lr_ratio")
                btn = gr.Button("绘制 LR 曲线", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="LR 调度曲线与变化率")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_lr_schedules,
                  inputs=[total_steps, warmup_ratio, min_lr_ratio],
                  outputs=[plot, text])


def _tab_curriculum():
    with gr.TabItem("3. Context Curriculum"):
        with gr.Row():
            with gr.Column(scale=1):
                stages_str = gr.Textbox(value="32,64,128,256", label="各阶段 seq_len（逗号分隔）")
                fractions_str = gr.Textbox(value="0.4,0.7,0.9,1.0", label="各阶段结束比例")
                total_steps = gr.Slider(100, 1000, value=200, step=50, label="总步数")
                max_tokens = gr.Slider(32, 256, value=64, step=16, label="max_tokens_per_rank")
                btn = gr.Button("生成课程表", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Curriculum 阶段与预算")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_curriculum,
                  inputs=[stages_str, fractions_str, total_steps, max_tokens],
                  outputs=[plot, text])


def _tab_grad_accum():
    with gr.TabItem("4. 梯度累积"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(5, 40, value=20, step=5, label="训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                lr = gr.Slider(5e-4, 3e-3, value=1e-3, step=1e-4, label="学习率")
                total_batch = gr.Slider(8, 128, value=32, step=8, label="有效批大小 (tok)")
                btn = gr.Button("对比 grad_accum", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="不同 grad_accum 下的 loss 轨迹")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_grad_accum,
                  inputs=[steps, seed, lr, total_batch],
                  outputs=[plot, text])


def _tab_checkpoint():
    with gr.TabItem("5. Checkpoint 验证"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(5, 40, value=20, step=5, label="预训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                btn = gr.Button("验证 round-trip", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Checkpoint 一致性验证")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_checkpoint_roundtrip,
                  inputs=[steps, seed],
                  outputs=[plot, text])


def _tab_evaluate():
    with gr.TabItem("6. 评估 (eval_loss / ppl)"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(5, 40, value=15, step=5, label="训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                mbs = gr.Slider(1, 8, value=4, step=1, label="micro_batch_size")
                btn = gr.Button("评估前后对比", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="训练前后评估指标对比")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_evaluate,
                  inputs=[steps, seed, mbs],
                  outputs=[plot, text])


def _tab_device_gc():
    with gr.TabItem("7. 设备与 Grad-Checkpoint"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(5, 30, value=15, step=5, label="对比训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                btn = gr.Button("查看设备策略", variant="primary")
                gr.Markdown(
                    "**resolve_device**：显式指定 / auto 智能选择。"
                    "**gradient_checkpointing**：用时间换显存。"
                )
            with gr.Column(scale=2):
                plot = gr.Plot(label="设备选择与显存-速度权衡")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_device_and_gc,
                  inputs=[steps, seed],
                  outputs=[plot, text])


def _tab_running_mean():
    with gr.TabItem("8. RunningMean 滑动均值"):
        with gr.Row():
            with gr.Column(scale=1):
                window = gr.Slider(1, 50, value=10, step=1, label="窗口大小")
                noise = gr.Slider(0.1, 1.0, value=0.3, step=0.05, label="噪声强度")
                n_points = gr.Slider(20, 200, value=100, step=10, label="数据点数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                btn = gr.Button("查看平滑效果", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="RunningMean 滑动窗口平滑")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_running_mean,
                  inputs=[window, noise, n_points, seed],
                  outputs=[plot, text])


def _tab_metric_log():
    with gr.TabItem("9. MetricLog 记录"):
        with gr.Row():
            with gr.Column(scale=1):
                total_steps = gr.Slider(20, 200, value=50, step=10, label="总步数")
                log_every = gr.Slider(1, 10, value=2, step=1, label="log_every")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                btn = gr.Button("生成日志曲线", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="MetricLog：loss / lr / grad_norm")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_metric_log,
                  inputs=[total_steps, log_every, seed],
                  outputs=[plot, text])


def _tab_expert_balance():
    with gr.TabItem("10. MoE 负载均衡"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(10, 60, value=30, step=5, label="训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                mbs = gr.Slider(2, 8, value=4, step=1, label="micro_batch_size")
                top_k = gr.Slider(1, 4, value=2, step=1, label="top_k")
                btn = gr.Button("观察负载均衡", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="MoE 专家负载均衡曲线")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_expert_balance,
                  inputs=[steps, seed, mbs, top_k],
                  outputs=[plot, text])


def _tab_decay_stats():
    with gr.TabItem("11. KDA alpha 衰减"):
        with gr.Row():
            with gr.Column(scale=1):
                steps = gr.Slider(10, 60, value=30, step=5, label="训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                mbs = gr.Slider(2, 8, value=4, step=1, label="micro_batch_size")
                btn = gr.Button("观察 alpha 分布", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="KDA alpha 衰减因子范围")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_decay_stats,
                  inputs=[steps, seed, mbs],
                  outputs=[plot, text])


def _tab_metrics_no_model():
    with gr.TabItem("12. 无模型回退"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "**验证**：当模型不含 KDA 或 MoE 层时，"
                    "`expert_balance()` 和 `decay_stats()` "
                    "返回空 dict 而不报错。"
                )
                btn = gr.Button("验证回退行为", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="无模型时的回退行为")
                text = gr.Textbox(label="结论", lines=4)
        btn.click(fig_metrics_no_model,
                  inputs=[],
                  outputs=[plot, text])


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
