# -*- coding: utf-8 -*-
"""Gradio UI 构建：K3 并行机制演示。

设计原则:
  - 所有可视化数据都来自 models.llm.kimi_k3.parallel 真实代码调用
  - 复用 plot_xxx.py 的 fig_xxx() 函数，不重复实现
  - 每个 Tab 遵循「左参数 + 右图表 + 结论」的标准布局

19 个 Tab 分布在 6 大模块：
  - 进程网格 (2): mesh 布局 / 轴分组
  - KCP 上下文并行 (3): 等价性 / 前缀扫描自洽 / 通信量
  - MoonEP 专家并行 (4): 负载均衡 / 附录 E 界 / 派发置换 / 缓冲大小
  - ZeRO 梯度分片 (3): 分片区间 / 显存阶梯 / 往返重建
  - 流水线 (4): 层切分 / 1F1B 调度 / AttnRes 缓存 / 激活均衡
  - 激活管理 (3): 策略账本 / FP8 块大小 / 恢复误差对比
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from plot_utils import setup_cjk_font, get_font_prop

setup_cjk_font()

# 导入各 plot 模块的 fig_xxx 函数（零重复代码）
from plot_groups import fig_mesh_layout, fig_axis_groups
from plot_kcp import (
    fig_kcp_vs_recurrent, fig_prefix_scan_self_consistency,
    fig_communication_volume,
)
from plot_moonep import (
    fig_load_balance, fig_appendix_e_bound,
    fig_dispatch_permutation, fig_buffer_size,
)
from plot_zero import (
    fig_shard_layout, fig_memory_staircase, fig_roundtrip_world1,
)
from plot_pipeline import (
    fig_layer_partition, fig_1f1b_schedule,
    fig_attnres_cache, fig_activation_balance,
)
from plot_activation import (
    fig_policy_ledger, fig_fp8_block_size, fig_policies_restore,
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="K3 并行机制交互式演示") as demo:
        gr.Markdown(
            """
            # K3 并行机制交互式演示

            基于 `models/llm/kimi_k3/parallel/` 真实代码运行，
            所有图均为 CPU 实时计算。

            **6 大模块 / 19 个实验板块**：
            - **进程网格** — PP x EP x CP x DP 四维 mesh 布局
            - **KCP 上下文并行** — KDA 的前缀扫描式分段并行
            - **MoonEP 专家并行** — 复制专家换完美均衡 + 静态 shape
            - **ZeRO 梯度分片** — 梯度分片 + offload CPU
            - **流水线** — 1F1B 调度 / AttnRes 增量通信 / 激活均衡
            - **激活管理** — 5 种存储策略 / FP8 块量化 / 精度-显存 trade-off
            """
        )

        with gr.Tabs():
            _tab_mesh_layout()
            _tab_axis_groups()
            _tab_kcp_equivalence()
            _tab_kcp_prefix_scan()
            _tab_kcp_comm_volume()
            _tab_moonep_load_balance()
            _tab_moonep_appendix_e()
            _tab_moonep_dispatch()
            _tab_moonep_buffer()
            _tab_zero_shard_layout()
            _tab_zero_staircase()
            _tab_zero_roundtrip()
            _tab_pipeline_partition()
            _tab_pipeline_schedule()
            _tab_pipeline_attnres()
            _tab_pipeline_act_balance()
            _tab_activation_ledger()
            _tab_activation_fp8()
            _tab_activation_restore()

    return demo


# ============================================================
# 1. 进程网格
# ============================================================

def _tab_mesh_layout():
    with gr.TabItem("1.1 网格坐标映射"):
        with gr.Row():
            with gr.Column(scale=1):
                pp = gr.Slider(1, 8, value=2, step=1, label="PP (流水线)")
                ep = gr.Slider(1, 8, value=4, step=1, label="EP (专家)")
                cp = gr.Slider(1, 8, value=2, step=1, label="CP (上下文)")
                dp = gr.Slider(1, 4, value=2, step=1, label="DP (数据)")
                btn = gr.Button("生成网格", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="rank -> 坐标映射")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_mesh_layout,
                  inputs=[pp, ep, cp, dp],
                  outputs=[plot, text])


def _tab_axis_groups():
    with gr.TabItem("1.2 轴分组"):
        with gr.Row():
            with gr.Column(scale=1):
                pp2 = gr.Slider(1, 4, value=2, step=1, label="PP")
                ep2 = gr.Slider(1, 4, value=2, step=1, label="EP")
                cp2 = gr.Slider(1, 4, value=2, step=1, label="CP")
                dp2 = gr.Slider(1, 2, value=1, step=1, label="DP")
                hl = gr.Slider(0, 7, value=3, step=1, label="高亮 rank")
                btn = gr.Button("查看分组", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="四轴分组（红=高亮 rank 所在组）")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_axis_groups,
                  inputs=[pp2, ep2, cp2, dp2, hl],
                  outputs=[plot, text])


# ============================================================
# 2. KCP 上下文并行
# ============================================================

def _tab_kcp_equivalence():
    with gr.TabItem("2.1 KCP vs 递推等价性"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=0, step=1, label="随机种子")
                seq = gr.Slider(32, 256, value=128, step=32, label="序列长度")
                dk = gr.Slider(8, 32, value=16, step=4, label="Dk (K 维度)")
                dv = gr.Slider(8, 32, value=16, step=4, label="Dv (V 维度)")
                heads = gr.Slider(1, 8, value=3, step=1, label="头数")
                nseg = gr.Slider(2, 8, value=4, step=1, label="n_segments")
                cs = gr.Slider(8, 32, value=16, step=4, label="chunk_size")
                btn = gr.Button("验证等价性", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="误差分布 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_kcp_vs_recurrent,
                  inputs=[seed, seq, dk, dv, heads, nseg, cs],
                  outputs=[plot, text])


def _tab_kcp_prefix_scan():
    with gr.TabItem("2.2 前缀扫描自洽性"):
        with gr.Row():
            with gr.Column(scale=1):
                seed2 = gr.Slider(0, 999, value=0, step=1, label="随机种子")
                seq2 = gr.Slider(32, 256, value=64, step=32, label="序列长度")
                dk2 = gr.Slider(8, 32, value=16, step=4, label="Dk")
                dv2 = gr.Slider(8, 32, value=16, step=4, label="Dv")
                heads2 = gr.Slider(1, 4, value=2, step=1, label="头数")
                nseg2 = gr.Slider(2, 8, value=4, step=1, label="n_segments")
                cs2 = gr.Slider(8, 32, value=16, step=4, label="chunk_size")
                btn = gr.Button("运行扫描", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="前缀扫描 (1x3)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_prefix_scan_self_consistency,
                  inputs=[seed2, seq2, dk2, dv2, heads2, nseg2, cs2],
                  outputs=[plot, text])


def _tab_kcp_comm_volume():
    with gr.TabItem("2.3 通信量与序列长度无关"):
        with gr.Row():
            with gr.Column(scale=1):
                dk3 = gr.Slider(8, 64, value=16, step=4, label="Dk")
                dv3 = gr.Slider(8, 64, value=16, step=4, label="Dv")
                heads3 = gr.Slider(8, 64, value=32, step=4, label="头数")
                batch3 = gr.Slider(1, 8, value=1, step=1, label="batch")
                btn = gr.Button("计算通信量", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="通信量对比 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_communication_volume,
                  inputs=[dk3, dv3, heads3, batch3],
                  outputs=[plot, text])


# ============================================================
# 3. MoonEP 专家并行
# ============================================================

def _tab_moonep_load_balance():
    with gr.TabItem("3.1 负载均衡"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=0, step=1, label="随机种子")
                ne = gr.Slider(16, 128, value=64, step=16, label="专家数 E")
                nr = gr.Slider(2, 16, value=8, step=1, label="rank 数 R")
                tpr = gr.Slider(128, 1024, value=512, step=128, label="S*K (每 rank 目标)")
                skew = gr.Slider(0.5, 5.0, value=2.0, step=0.5, label="skew (热点程度)")
                btn = gr.Button("规划并对比", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="naive vs MoonEP (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_load_balance,
                  inputs=[seed, ne, nr, tpr, skew],
                  outputs=[plot, text])


def _tab_moonep_appendix_e():
    with gr.TabItem("3.2 附录 E 界 M(I)≤⌈E/R⌉"):
        with gr.Row():
            with gr.Column(scale=1):
                ne2 = gr.Slider(16, 128, value=64, step=16, label="专家数 E")
                nr2 = gr.Slider(2, 16, value=8, step=1, label="rank 数 R")
                tpr2 = gr.Slider(64, 512, value=256, step=64, label="S*K")
                nseeds = gr.Slider(20, 200, value=100, step=20, label="实验次数")
                skew2 = gr.Slider(1.0, 5.0, value=3.0, step=0.5, label="skew")
                btn = gr.Button("验证附录 E 定理", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="max_redundant & moves 分布 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_appendix_e_bound,
                  inputs=[ne2, nr2, tpr2, nseeds, skew2],
                  outputs=[plot, text])


def _tab_moonep_dispatch():
    with gr.TabItem("3.3 派发置换矩阵"):
        with gr.Row():
            with gr.Column(scale=1):
                seed3 = gr.Slider(0, 999, value=0, step=1, label="随机种子")
                ne3 = gr.Slider(8, 32, value=16, step=4, label="专家数 E")
                nr3 = gr.Slider(2, 8, value=4, step=1, label="rank 数 R")
                nt3 = gr.Slider(64, 512, value=256, step=64, label="token 数 N")
                btn = gr.Button("生成派发", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="派发置换 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_dispatch_permutation,
                  inputs=[seed3, ne3, nr3, nt3],
                  outputs=[plot, text])


def _tab_moonep_buffer():
    with gr.TabItem("3.4 静态 shape 缓冲"):
        with gr.Row():
            with gr.Column(scale=1):
                nr4 = gr.Slider(2, 16, value=8, step=1, label="rank 数 R")
                tpr4 = gr.Slider(64, 512, value=128, step=64, label="S*K (token/rank)")
                btn = gr.Button("计算缓冲", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="动态 vs 静态缓冲大小")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_buffer_size,
                  inputs=[nr4, tpr4],
                  outputs=[plot, text])


# ============================================================
# 4. ZeRO 梯度分片
# ============================================================

def _tab_zero_shard_layout():
    with gr.TabItem("4.1 分片区间"):
        with gr.Row():
            with gr.Column(scale=1):
                numel = gr.Slider(100, 5000, value=1000, step=100, label="元素总数")
                world = gr.Slider(2, 16, value=8, step=1, label="world size")
                btn = gr.Button("生成分片", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="shard_ranges 可视化")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_shard_layout,
                  inputs=[numel, world],
                  outputs=[plot, text])


def _tab_zero_staircase():
    with gr.TabItem("4.2 显存阶梯"):
        with gr.Row():
            with gr.Column(scale=1):
                total = gr.Slider(100_000, 5_000_000, value=1_000_000,
                                  step=100_000, label="总参数量")
                btn = gr.Button("计算显存", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="显存阶梯 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_memory_staircase,
                  inputs=[total],
                  outputs=[plot, text])


def _tab_zero_roundtrip():
    with gr.TabItem("4.3 往返重建 (world=1)"):
        with gr.Row():
            with gr.Column(scale=1):
                np2 = gr.Slider(1, 8, value=3, step=1, label="参数个数")
                dim2 = gr.Slider(16, 128, value=64, step=16, label="每参数维度")
                btn = gr.Button("验证往返", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="重建误差 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_roundtrip_world1,
                  inputs=[np2, dim2],
                  outputs=[plot, text])


# ============================================================
# 5. 流水线
# ============================================================

def _tab_pipeline_partition():
    with gr.TabItem("5.1 层切分"):
        with gr.Row():
            with gr.Column(scale=1):
                nl = gr.Slider(20, 200, value=93, step=1, label="总层数")
                pp = gr.Slider(2, 32, value=16, step=1, label="PP stages")
                vp = gr.Slider(1, 4, value=1, step=1, label="virtual stages")
                btn = gr.Button("切分层", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="层切分可视化")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_layer_partition,
                  inputs=[nl, pp, vp],
                  outputs=[plot, text])


def _tab_pipeline_schedule():
    with gr.TabItem("5.2 1F1B 调度"):
        with gr.Row():
            with gr.Column(scale=1):
                ns = gr.Slider(2, 8, value=4, step=1, label="stages")
                nm = gr.Slider(4, 32, value=16, step=2, label="micro-batches")
                vp2 = gr.Slider(1, 4, value=2, step=1, label="virtual stages")
                btn = gr.Button("生成调度", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="1F1B 甘特图 + warmup 倾斜 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_1f1b_schedule,
                  inputs=[ns, nm, vp2],
                  outputs=[plot, text])


def _tab_pipeline_attnres():
    with gr.TabItem("5.3 AttnRes 增量通信"):
        with gr.Row():
            with gr.Column(scale=1):
                ns2 = gr.Slider(2, 8, value=4, step=1, label="stages")
                nm2 = gr.Slider(4, 16, value=8, step=1, label="micro-batches")
                bps = gr.Slider(1, 8, value=3, step=1, label="每 stage block 数")
                btn = gr.Button("模拟 AttnRes", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="增量 vs naive 通信 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_attnres_cache,
                  inputs=[ns2, nm2, bps],
                  outputs=[plot, text])


def _tab_pipeline_act_balance():
    with gr.TabItem("5.4 激活均衡"):
        with gr.Row():
            with gr.Column(scale=1):
                ns3 = gr.Slider(3, 8, value=4, step=1, label="stages")
                sk = gr.Slider(0.1, 0.8, value=0.3, step=0.1, label="倾斜度 skew")
                btn = gr.Button("均衡激活", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="激活均衡前后 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_activation_balance,
                  inputs=[ns3, sk],
                  outputs=[plot, text])


# ============================================================
# 6. 激活管理
# ============================================================

def _tab_activation_ledger():
    with gr.TabItem("6.1 策略账本"):
        with gr.Row():
            with gr.Column(scale=1):
                bs = gr.Slider(1, 8, value=2, step=1, label="batch")
                sq = gr.Slider(32, 256, value=64, step=32, label="seq_len")
                hd = gr.Slider(1024, 8192, value=4096, step=1024, label="hidden")
                btn = gr.Button("计算账本", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="5 种策略 + 混合策略账本 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_policy_ledger,
                  inputs=[bs, sq, hd],
                  outputs=[plot, text])


def _tab_activation_fp8():
    with gr.TabItem("6.2 FP8 块大小 vs 误差"):
        with gr.Row():
            with gr.Column(scale=1):
                sq2 = gr.Slider(16, 128, value=64, step=16, label="seq_len")
                hd2 = gr.Slider(128, 1024, value=512, step=128, label="hidden")
                os3 = gr.Slider(100, 500000, value=100000, step=1000,
                                label="离群通道 scale")
                btn = gr.Button("测量误差", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="FP8 块大小误差曲线 + 逐通道误差对比 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_fp8_block_size,
                  inputs=[sq2, hd2, os3],
                  outputs=[plot, text])


def _tab_activation_restore():
    with gr.TabItem("6.3 策略恢复误差对比"):
        with gr.Row():
            with gr.Column(scale=1):
                sq3 = gr.Slider(8, 64, value=16, step=4, label="seq_len")
                hd3 = gr.Slider(64, 512, value=256, step=32, label="hidden")
                btn = gr.Button("对比所有策略", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="5 种策略恢复误差 + 显存-精度 trade-off (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_policies_restore,
                  inputs=[sq3, hd3],
                  outputs=[plot, text])
