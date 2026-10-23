# -*- coding: utf-8 -*-
"""Gradio UI 构建：MTP / EAGLE-3 演示的 13 个 Tab。

与 _mtp_demo_figs.py 配合：后者提供所有绘图函数，本文件负责 Gradio 控件组装。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
_EXP_ROOT = str(Path(__file__).resolve().parents[5] / "experiments")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from plot_utils import setup_cjk_font
setup_cjk_font()

from plot_mtp import (
    fig_mtp_offset, fig_e1_joint_loss, fig_e2_init_curve, fig_e3_speedup,
)
from eagle3_demo import run_spec_decode, fig_measured_speedup
from discussion import e1_result, e4_result

from _mtp_demo_figs import (
    fig_architecture_overview,
    fig_verify_demo,
    fig_lk_loss_demo,
    fig_draft_pipeline_demo,
    fig_cache_rollback_demo,
    fig_mtp_structure_and_params,
    fig_loss_weight_sweep,
    fig_acceptance_heatmap,
)

plt.rcParams["figure.facecolor"] = "#17181c"
plt.rcParams["axes.facecolor"] = "#17181c"
plt.rcParams["axes.edgecolor"] = "#888899"
plt.rcParams["axes.labelcolor"] = "#dddddd"
plt.rcParams["xtick.color"] = "#bbbbbb"
plt.rcParams["ytick.color"] = "#bbbbbb"
plt.rcParams["text.color"] = "#e8e8e8"
plt.rcParams["axes.titlecolor"] = "#ffffff"
plt.rcParams["legend.facecolor"] = "#22232a"


def _run_measured(seed, prompt_len, max_new_tokens, draft_len):
    agree, stats_re, stats_ls, L = run_spec_decode(
        seed=int(seed), prompt_len=int(prompt_len),
        max_new_tokens=int(max_new_tokens), draft_len=int(draft_len),
    )
    r_theo = stats_ls.measured_acceptance
    fig, _ = fig_measured_speedup(L, stats_ls, r_theo)
    lines = [
        f"prompt_len={int(prompt_len)}, max_new={int(max_new_tokens)}, draft_len={L}",
        f"Recompute vs Rollback token 一致: {agree}",
        f"greedy 路径: 接受率={stats_re.measured_acceptance:.4f}, "
        f"tokens/forward={stats_re.tokens_per_forward:.3f}",
        f"lossless 路径: 接受率={stats_ls.measured_acceptance:.4f}, "
        f"tokens/forward={stats_ls.tokens_per_forward:.3f} (基线 1.0)",
        "两路径一致 = cache 回滚实现正确的数值证明。",
    ]
    return fig, "\n".join(lines)


def _e3_wrapper(r_min, r_max, hl_r, hl_l):
    return fig_e3_speedup(
        rates_range=(float(r_min), float(r_max)),
        draft_lens=(1, 2, 4, 7),
        highlight_rate=float(hl_r),
        highlight_len=int(hl_l),
    )


def build_demo():
    with gr.Blocks(title="MTP / EAGLE-3 交互式演示") as demo:
        gr.Markdown(
            """
            # MTP / EAGLE-3 交互式演示

            基于 `models/llm/kimi_k3/mtp/` 真实代码运行，所有图均为 CPU 实时计算。

            13 个实验板块：
            1. **架构总览** — MTP 训练 + EAGLE-3 推理全链路
            2. **偏移对齐** — x[t]→x[t+1]→x[t+2] 的 token 流
            3. **E1 联合损失** — 纯 LM vs LM+MTP 的 lm_loss 收敛
            4. **E2 融合初值** — [0 0 I] vs 随机 / 全零
            5. **E3 加速比理论** — 接受率 × 草稿长度
            6. **E5 实测加速** — 真实投机解码数值
            7. **Verify 判据** — lossless / greedy 接受/拒绝
            8. **LK 损失** — 直接最大化接受率
            9. **Draft 链路** — EAGLE-3 自回归起草
            10. **Cache 回滚** — KDA / MLA 快照-恢复正确性
            11. **结构与参数** — MTP 层结构 + 参数量分解
            12. **loss_weight 扫描** — λ 对联合损失的影响
            13. **加速比热力图** — r × L 二维视图
            """
        )

        with gr.Tabs():

            # ---- Tab 1 ----
            with gr.TabItem("1. 架构总览"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**MTP 预训练 → EAGLE-3 投机解码** 全链路。")
                        btn = gr.Button("查看架构图", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="架构图")
                        text = gr.Textbox(label="说明", lines=4)
                btn.click(fig_architecture_overview, inputs=[], outputs=[plot, text])

            # ---- Tab 2 ----
            with gr.TabItem("2. 偏移对齐"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("MTP 预测 x[t+2]，而非 x[t+1]。")
                        btn = gr.Button("查看偏移示意", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="偏移示意图")
                        text = gr.Textbox(label="说明", lines=3)
                btn.click(fig_mtp_offset, inputs=[], outputs=[plot, text])

            # ---- Tab 3 ----
            with gr.TabItem("3. E1 联合损失"):
                with gr.Row():
                    with gr.Column(scale=1):
                        lw = gr.Slider(0.1, 2.0, value=0.5, step=0.1,
                                       label="MTP loss_weight")
                        st = gr.Slider(10, 80, value=40, step=5,
                                       label="训练步数")
                        btn = gr.Button("运行 E1 实验", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="lm_loss 曲线")
                        txt = gr.Textbox(label="数值结论", lines=5)
                btn.click(fig_e1_joint_loss,
                          inputs=[lw, st], outputs=[plot, txt])

            # ---- Tab 4 ----
            with gr.TabItem("4. E2 融合初值"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dim = gr.Slider(16, 128, value=32, step=16,
                                        label="特征维度")
                        ns = gr.Slider(20, 200, value=60, step=10,
                                       label="优化步数")
                        btn = gr.Button("对比三种初值", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="融合误差曲线")
                        txt = gr.Textbox(label="数值结论", lines=5)
                btn.click(fig_e2_init_curve,
                          inputs=[dim, ns], outputs=[plot, txt])

            # ---- Tab 5 ----
            with gr.TabItem("5. E3 加速比理论"):
                with gr.Row():
                    with gr.Column(scale=1):
                        r_min = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                          label="接受率范围左")
                        r_max = gr.Slider(0.5, 0.99, value=0.98, step=0.01,
                                          label="接受率范围右")
                        hl_r = gr.Slider(0.1, 0.99, value=0.8, step=0.01,
                                         label="高亮接受率 r")
                        hl_l = gr.Slider(1, 10, value=7, step=1,
                                         label="高亮草稿长度 L")
                        btn = gr.Button("画理论曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="加速比曲线")
                        txt = gr.Textbox(label="说明", lines=3)
                btn.click(_e3_wrapper,
                          inputs=[r_min, r_max, hl_r, hl_l],
                          outputs=[plot, txt])

            # ---- Tab 6 ----
            with gr.TabItem("6. E5 实测加速"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed_e5 = gr.Slider(0, 9999, value=1234, step=1,
                                            label="随机种子")
                        plen = gr.Slider(4, 16, value=8, step=1,
                                         label="prompt 长度")
                        mnew = gr.Slider(8, 48, value=24, step=4,
                                         label="max_new_tokens")
                        dlen = gr.Slider(1, 7, value=3, step=1,
                                         label="draft_len")
                        btn = gr.Button("运行投机解码", variant="primary")
                        gr.Markdown("运行较慢（CPU 小模型真实前向）")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="实测点 vs 理论曲线")
                        txt = gr.Textbox(label="统计数据", lines=6)
                btn.click(_run_measured,
                          inputs=[seed_e5, plen, mnew, dlen],
                          outputs=[plot, txt])

            # ---- Tab 7 ----
            with gr.TabItem("7. Verify 判据"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v = gr.Slider(4, 64, value=16, step=2,
                                      label="词表大小 V")
                        L = gr.Slider(1, 10, value=5, step=1,
                                      label="草稿长度 L")
                        pb = gr.Slider(0.0, 10.0, value=3.0, step=0.5,
                                       label="目标偏向 token 0 强度")
                        qb = gr.Slider(0.0, 10.0, value=0.5, step=0.5,
                                       label="草稿偏向 token V-1 强度")
                        mode = gr.Radio(["greedy", "lossless"], value="lossless",
                                        label="判据模式")
                        seed_v = gr.Slider(0, 999, value=42, step=1,
                                           label="随机种子")
                        btn = gr.Button("运行 verify", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="接受/拒绝可视化")
                        txt = gr.Textbox(label="结果", lines=4)
                btn.click(fig_verify_demo,
                          inputs=[v, L, pb, qb, mode, seed_v],
                          outputs=[plot, txt])

            # ---- Tab 8 ----
            with gr.TabItem("8. LK 损失"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v8 = gr.Slider(8, 128, value=32, step=8,
                                       label="词表大小")
                        s8 = gr.Slider(10, 200, value=50, step=10,
                                       label="疏离步数")
                        d8 = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                       label="最终疏离程度")
                        btn = gr.Button("展示 LK 损失", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="LK 损失曲线")
                        txt = gr.Textbox(label="结论", lines=4)
                btn.click(fig_lk_loss_demo,
                          inputs=[v8, s8, d8], outputs=[plot, txt])

            # ---- Tab 9 ----
            with gr.TabItem("9. Draft 链路"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed9 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        ns9 = gr.Slider(1, 12, value=5, step=1,
                                        label="draft 步数")
                        lk9 = gr.Radio(["mla", "kda"], value="mla",
                                       label="MTP 注意力类型")
                        temp9 = gr.Slider(0.0, 2.0, value=1.0, step=0.1,
                                          label="采样温度（0=greedy，1=标准采样）")
                        btn = gr.Button("运行 draft 链路", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="draft 分布演化")
                        txt = gr.Textbox(label="结果", lines=4)
                btn.click(fig_draft_pipeline_demo,
                          inputs=[seed9, ns9, lk9, temp9], outputs=[plot, txt])

            # ---- Tab 10 ----
            with gr.TabItem("10. Cache 回滚"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed10 = gr.Slider(0, 999, value=0, step=1,
                                           label="随机种子")
                        ct = gr.Radio(["KDA", "MLA"], value="KDA",
                                      label="Cache 类型")
                        btn = gr.Button("验证回滚正确性", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="回滚数值对比")
                        txt = gr.Textbox(label="结论", lines=4)
                btn.click(fig_cache_rollback_demo,
                          inputs=[seed10, ct], outputs=[plot, txt])

            # ---- Tab 11 ----
            with gr.TabItem("11. 结构与参数"):
                with gr.Row():
                    with gr.Column(scale=1):
                        lk11 = gr.Radio(["mla", "kda"], value="mla",
                                        label="MTP 注意力类型")
                        btn = gr.Button("查看结构与参数量", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="MTP 层结构 + 参数量")
                        txt = gr.Textbox(label="结论", lines=4)
                btn.click(fig_mtp_structure_and_params,
                          inputs=[lk11], outputs=[plot, txt])

            # ---- Tab 12 ----
            with gr.TabItem("12. loss_weight 扫描"):
                with gr.Row():
                    with gr.Column(scale=1):
                        steps12 = gr.Slider(10, 100, value=30, step=10,
                                            label="训练步数")
                        btn = gr.Button("扫描 loss_weight", variant="primary")
                        gr.Markdown("运行较慢（多个权重分别训练）")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="loss_weight 扫描")
                        txt = gr.Textbox(label="结论", lines=4)
                btn.click(fig_loss_weight_sweep,
                          inputs=[steps12], outputs=[plot, txt])

            # ---- Tab 13 ----
            with gr.TabItem("13. 加速比热力图"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "接受率 r × 草稿长度 L 的二维热力图，"
                            "直观展示加速比的联合分布。"
                        )
                        btn = gr.Button("生成热力图", variant="primary")
                    with gr.Column(scale=2):
                        plot = gr.Plot(label="加速比热力图")
                        txt = gr.Textbox(label="结论", lines=3)
                btn.click(fig_acceptance_heatmap,
                          inputs=[], outputs=[plot, txt])

    return demo
