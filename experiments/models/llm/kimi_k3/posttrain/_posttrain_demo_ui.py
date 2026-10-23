# -*- coding: utf-8 -*-
"""Gradio UI 构建：K3 PostTrain 演示的 17 个 Tab。"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from plot_utils import setup_cjk_font
setup_cjk_font()

from plot_overview import fig_pipeline_overview
from plot_xtml import fig_xtml_zones, fig_roundtrip, fig_grammar_fsm
from plot_sft import fig_sft_mask, fig_sft_loss
from plot_budget import fig_budget_cliff, fig_effort_annealing
from plot_grm import fig_grm_rubric, fig_tournament
from plot_mopd import fig_opd_reward, fig_multi_teacher
from plot_rollout import fig_rollout_schedule, fig_regularizer
from plot_sandbox import (
    fig_sandbox_lifecycle, fig_incremental_snapshot, fig_pool_overcommit,
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

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="K3 PostTrain 交互式演示") as demo:
        gr.Markdown(
            """
            # K3 后训练交互式演示

            基于 `models/llm/kimi_k3/posttrain/` 真实代码运行，所有图均为 CPU 实时计算。

            7 大模块 / 17 个实验板块：
            1. **流水线全景** — XTML → SFT → RL → MOPD → AgentENV 全链路
            2. **XTML 区域划分** — 4 类区域长度分布
            3. **XTML 往返一致性** — serialize → parse 往返验证
            4. **语法约束状态机** — 4 状态 FSM 转移图
            5. **SFT Loss Mask** — 监督 token 比例 & mask 可视化
            6. **SFT Loss** — 交叉熵损失计算
            7. **预算控制：奖励悬崖** — token 数 vs reward 曲线
            8. **推理强度退火** — 三阶段阈值随难度变化
            9. **GRM Rubric 评分** — 多维度加权打分
            10. **GRM Tournament** — 锦标赛排名 + 对战矩阵
            11. **MOPD 裁剪奖励** — OPD clipped reward 曲线
            12. **MOPD 多教师路由** — 9 教师热力图
            13. **部分 Rollout 调度** — active/paused/completed 进度
            14. **Staleness 正则化** — regularizer vs PPO clip 对比
            15. **沙箱生命周期** — 运行/暂停切换时间线
            16. **增量快照** — 每次只保存脏页
            17. **池超分配** — 暂停沙箱不计入容量
            """
        )

        with gr.Tabs():

            # ---- Tab 1: 流水线全景 ----
            with gr.TabItem("1. 流水线全景"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("K3 后训练 7 大组件全链路架构。")
                        btn1 = gr.Button("查看全景图", variant="primary")
                    with gr.Column(scale=2):
                        plot1 = gr.Plot(label="流水线全景图")
                        text1 = gr.Textbox(label="说明", lines=4)
                btn1.click(fig_pipeline_overview, inputs=[], outputs=[plot1, text1])

            # ---- Tab 2: XTML 区域划分 ----
            with gr.TabItem("2. XTML 区域划分"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_kind = gr.Radio(
                            ["weather", "multi", "coding"],
                            value="weather", label="对话类型 kind")
                        t2_thinking = gr.Checkbox(value=True, label="thinking 模式")
                        t2_tools = gr.Checkbox(value=True, label="with_tools")
                        t2_effort = gr.Radio(
                            ["low", "medium", "high", "max"],
                            value="high", label="effort")
                        btn2 = gr.Button("查看区域划分", variant="primary")
                    with gr.Column(scale=2):
                        plot2 = gr.Plot(label="区域划分图")
                        text2 = gr.Textbox(label="统计", lines=5)
                btn2.click(fig_xtml_zones,
                           inputs=[t2_kind, t2_thinking, t2_tools, t2_effort],
                           outputs=[plot2, text2])

            # ---- Tab 3: XTML 往返一致性 ----
            with gr.TabItem("3. XTML 往返一致性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_kind = gr.Radio(
                            ["weather", "multi", "coding"],
                            value="multi", label="对话类型 kind")
                        t3_thinking = gr.Checkbox(value=True, label="thinking 模式")
                        btn3 = gr.Button("验证往返一致性", variant="primary")
                    with gr.Column(scale=2):
                        plot3 = gr.Plot(label="验证结果")
                        text3 = gr.Textbox(label="详情", lines=5)
                btn3.click(fig_roundtrip,
                           inputs=[t3_kind, t3_thinking],
                           outputs=[plot3, text3])

            # ---- Tab 4: 语法约束状态机 ----
            with gr.TabItem("4. 语法约束状态机"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t4_thinking = gr.Checkbox(value=True, label="thinking 模式")
                        btn4 = gr.Button("查看状态机", variant="primary")
                    with gr.Column(scale=2):
                        plot4 = gr.Plot(label="FSM 状态转移图")
                        text4 = gr.Textbox(label="状态说明", lines=4)
                btn4.click(fig_grammar_fsm,
                           inputs=[t4_thinking],
                           outputs=[plot4, text4])

            # ---- Tab 5: SFT Loss Mask ----
            with gr.TabItem("5. SFT Loss Mask"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_kind = gr.Radio(
                            ["weather", "multi", "coding"],
                            value="multi", label="对话类型 kind")
                        t5_thinking = gr.Checkbox(value=True, label="thinking 模式")
                        t5_mask_prompt = gr.Checkbox(value=True, label="mask_prompt")
                        t5_max_len = gr.Slider(100, 5000, value=1000, step=100,
                                               label="max_length")
                        btn5 = gr.Button("查看 Mask 分布", variant="primary")
                    with gr.Column(scale=2):
                        plot5 = gr.Plot(label="Mask 可视化")
                        text5 = gr.Textbox(label="统计", lines=5)
                btn5.click(fig_sft_mask,
                           inputs=[t5_kind, t5_thinking, t5_mask_prompt, t5_max_len],
                           outputs=[plot5, text5])

            # ---- Tab 6: SFT Loss ----
            with gr.TabItem("6. SFT Loss"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t6_bs = gr.Slider(1, 16, value=4, step=1,
                                          label="batch_size")
                        t6_sl = gr.Slider(8, 128, value=32, step=4,
                                          label="seq_len")
                        t6_vs = gr.Slider(32, 512, value=128, step=32,
                                          label="vocab_size")
                        t6_seed = gr.Slider(0, 9999, value=42, step=1,
                                            label="随机种子")
                        t6_mr = gr.Slider(0.0, 0.8, value=0.4, step=0.05,
                                          label="mask_ratio")
                        btn6 = gr.Button("计算 SFT Loss", variant="primary")
                    with gr.Column(scale=2):
                        plot6 = gr.Plot(label="Loss 分析图")
                        text6 = gr.Textbox(label="结果", lines=5)
                btn6.click(fig_sft_loss,
                           inputs=[t6_bs, t6_sl, t6_vs, t6_seed, t6_mr],
                           outputs=[plot6, text6])

            # ---- Tab 7: 预算控制 - 奖励悬崖 ----
            with gr.TabItem("7. 预算控制：奖励悬崖"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t7_pid = gr.Textbox(value="math_easy", label="problem_id")
                        t7_bb = gr.Slider(100, 5000, value=500, step=100,
                                          label="base_budget")
                        t7_tau = gr.Slider(0.5, 10, value=3.0, step=0.5, label="tau")
                        t7_tr = gr.Slider(0, 2, value=1.0, step=0.1,
                                          label="task_reward")
                        t7_hc = gr.Slider(0, 50000, value=0, step=1000,
                                          label="hard_cap (0=不设)")
                        t7_cm = gr.Radio(
                            ["thinking", "output"],
                            value="thinking", label="count_mode")
                        btn7 = gr.Button("绘制悬崖曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot7 = gr.Plot(label="悬崖曲线图")
                        text7 = gr.Textbox(label="结果", lines=5)
                btn7.click(fig_budget_cliff,
                           inputs=[t7_pid, t7_bb, t7_tau, t7_tr, t7_hc, t7_cm],
                           outputs=[plot7, text7])

            # ---- Tab 8: 推理强度退火 ----
            with gr.TabItem("8. 推理强度退火"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t8_tmax = gr.Slider(4, 16, value=8.0, step=0.5,
                                            label="tau_max")
                        t8_thigh = gr.Slider(1, 6, value=3.0, step=0.5,
                                             label="tau_high")
                        t8_tlow = gr.Slider(0.3, 2, value=1.0, step=0.1,
                                            label="tau_low")
                        t8_np = gr.Slider(3, 10, value=5, step=1,
                                          label="n_problems")
                        btn8 = gr.Button("绘制退火曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot8 = gr.Plot(label="退火曲线图")
                        text8 = gr.Textbox(label="结果", lines=4)
                btn8.click(fig_effort_annealing,
                           inputs=[t8_tmax, t8_thigh, t8_tlow, t8_np],
                           outputs=[plot8, text8])

            # ---- Tab 9: GRM Rubric 评分 ----
            with gr.TabItem("9. GRM Rubric 评分"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t9_nc = gr.Slider(2, 6, value=3, step=1,
                                          label="n_criteria")
                        t9_seed = gr.Slider(0, 9999, value=42, step=1,
                                            label="随机种子")
                        btn9 = gr.Button("运行 Rubric 评分", variant="primary")
                    with gr.Column(scale=2):
                        plot9 = gr.Plot(label="评分结果")
                        text9 = gr.Textbox(label="详情", lines=4)
                btn9.click(fig_grm_rubric,
                           inputs=[t9_nc, t9_seed],
                           outputs=[plot9, text9])

            # ---- Tab 10: GRM Tournament ----
            with gr.TabItem("10. GRM Tournament"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t10_ncand = gr.Slider(2, 8, value=4, step=1,
                                              label="n_candidates")
                        t10_sigma = gr.Slider(0.5, 10, value=3.0, step=0.5,
                                              label="sigma")
                        t10_l0 = gr.Slider(20, 500, value=100, step=20,
                                           label="l0")
                        t10_seed = gr.Slider(0, 9999, value=42, step=1,
                                             label="随机种子")
                        btn10 = gr.Button("运行锦标赛", variant="primary")
                    with gr.Column(scale=2):
                        plot10 = gr.Plot(label="锦标赛结果")
                        text10 = gr.Textbox(label="详情", lines=4)
                btn10.click(fig_tournament,
                            inputs=[t10_ncand, t10_sigma, t10_l0, t10_seed],
                            outputs=[plot10, text10])

            # ---- Tab 11: MOPD 裁剪奖励 ----
            with gr.TabItem("11. MOPD 裁剪奖励"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t11_rmax = gr.Slider(0.5, 10, value=4.0, step=0.5,
                                             label="r_max")
                        t11_ming = gr.Slider(-15, -2, value=-8, step=1,
                                             label="min_gap")
                        t11_maxg = gr.Slider(2, 15, value=8, step=1,
                                             label="max_gap")
                        btn11 = gr.Button("绘制裁剪曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot11 = gr.Plot(label="OPD 裁剪奖励曲线")
                        text11 = gr.Textbox(label="参数", lines=3)
                btn11.click(fig_opd_reward,
                            inputs=[t11_rmax, t11_ming, t11_maxg],
                            outputs=[plot11, text11])

            # ---- Tab 12: MOPD 多教师路由 ----
            with gr.TabItem("12. MOPD 多教师路由"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t12_seed = gr.Slider(0, 9999, value=42, step=1,
                                             label="随机种子")
                        t12_sl = gr.Slider(5, 50, value=20, step=5,
                                           label="seq_len")
                        t12_domain = gr.Radio(
                            ["general", "general_agent", "coding_agent"],
                            value="general", label="domain")
                        t12_effort = gr.Radio(
                            ["low", "high", "max"],
                            value="high", label="effort")
                        btn12 = gr.Button("查看多教师热力图", variant="primary")
                    with gr.Column(scale=2):
                        plot12 = gr.Plot(label="9 教师热力图")
                        text12 = gr.Textbox(label="结果", lines=3)
                btn12.click(fig_multi_teacher,
                            inputs=[t12_seed, t12_sl, t12_domain, t12_effort],
                            outputs=[plot12, text12])

            # ---- Tab 13: 部分 Rollout 调度 ----
            with gr.TabItem("13. 部分 Rollout 调度"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t13_np = gr.Slider(2, 20, value=8, step=1,
                                           label="n_prompts")
                        t13_ks = gr.Slider(1, 4, value=2, step=1,
                                           label="k_samples")
                        t13_lam = gr.Slider(0.2, 0.9, value=0.5, step=0.05,
                                            label="lambda")
                        t13_mi = gr.Slider(5, 30, value=15, step=1,
                                           label="max_iter")
                        t13_ltr = gr.Slider(0.1, 0.6, value=0.3, step=0.05,
                                            label="long_tail_ratio")
                        t13_seed = gr.Slider(0, 9999, value=42, step=1,
                                             label="随机种子")
                        btn13 = gr.Button("运行 Rollout 调度", variant="primary")
                    with gr.Column(scale=2):
                        plot13 = gr.Plot(label="Rollout 进度")
                        text13 = gr.Textbox(label="结果", lines=4)
                btn13.click(fig_rollout_schedule,
                            inputs=[t13_np, t13_ks, t13_lam, t13_mi, t13_ltr, t13_seed],
                            outputs=[plot13, text13])

            # ---- Tab 14: Staleness 正则化 ----
            with gr.TabItem("14. Staleness 正则化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t14_eps = gr.Slider(0.05, 0.5, value=0.2, step=0.05,
                                            label="epsilon")
                        t14_adv = gr.Slider(-2, 2, value=1.0, step=0.2,
                                            label="advantage")
                        t14_mr = gr.Slider(1.5, 5, value=2.5, step=0.5,
                                           label="max_ratio")
                        btn14 = gr.Button("绘制正则化曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot14 = gr.Plot(label="正则化曲线")
                        text14 = gr.Textbox(label="结果", lines=4)
                btn14.click(fig_regularizer,
                            inputs=[t14_eps, t14_adv, t14_mr],
                            outputs=[plot14, text14])

            # ---- Tab 15: 沙箱生命周期 ----
            with gr.TabItem("15. 沙箱生命周期"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t15_pf = gr.Slider(0.3, 0.9, value=0.7, step=0.05,
                                           label="pause_fraction")
                        t15_tt = gr.Slider(20, 200, value=100, step=10,
                                           label="total_ticks")
                        btn15 = gr.Button("查看生命周期", variant="primary")
                    with gr.Column(scale=2):
                        plot15 = gr.Plot(label="沙箱生命周期")
                        text15 = gr.Textbox(label="结果", lines=3)
                btn15.click(fig_sandbox_lifecycle,
                            inputs=[t15_pf, t15_tt],
                            outputs=[plot15, text15])

            # ---- Tab 16: 增量快照 ----
            with gr.TabItem("16. 增量快照"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t16_np = gr.Slider(10, 200, value=50, step=10,
                                           label="n_pages")
                        t16_ns = gr.Slider(3, 15, value=8, step=1,
                                           label="n_snapshots")
                        t16_dr = gr.Slider(0.05, 0.5, value=0.2, step=0.05,
                                           label="dirty_ratio")
                        t16_seed = gr.Slider(0, 9999, value=42, step=1,
                                             label="随机种子")
                        btn16 = gr.Button("查看增量快照", variant="primary")
                    with gr.Column(scale=2):
                        plot16 = gr.Plot(label="增量快照")
                        text16 = gr.Textbox(label="结果", lines=3)
                btn16.click(fig_incremental_snapshot,
                            inputs=[t16_np, t16_ns, t16_dr, t16_seed],
                            outputs=[plot16, text16])

            # ---- Tab 17: 池超分配 ----
            with gr.TabItem("17. 池超分配"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t17_cap = gr.Slider(2, 32, value=8, step=2,
                                            label="capacity")
                        t17_ns = gr.Slider(10, 100, value=40, step=10,
                                           label="n_sandboxes")
                        t17_pr = gr.Slider(0.5, 0.95, value=0.8, step=0.05,
                                           label="pause_ratio")
                        t17_seed = gr.Slider(0, 9999, value=42, step=1,
                                             label="随机种子")
                        btn17 = gr.Button("查看超分配", variant="primary")
                    with gr.Column(scale=2):
                        plot17 = gr.Plot(label="池超分配")
                        text17 = gr.Textbox(label="结果", lines=3)
                btn17.click(fig_pool_overcommit,
                            inputs=[t17_cap, t17_ns, t17_pr, t17_seed],
                            outputs=[plot17, text17])

    return demo
