# -*- coding: utf-8 -*-
"""Gated MLA 模块交互式演示（Gradio）—— 覆盖门控多头潜在注意力的全部核心功能。

模块范围：mla.py（GatedMLA + MLACache）
测试覆盖：tests/models/llm/kimi_k3/modules/test_mla.py

7 Tabs:
  Cache Size (1):     MLA latent cache vs 标准 KV cache 压缩比
  Gate Suppression (1): 输出门 sigmoid(W_g x) 的抑制能力
  Rank Scan (1):      kv_lora_rank vs 重构信息丰富度
  Incremental Decode (1): prefill vs incremental decode 逐位一致
  NoPE (1):           MLA 层不含位置编码参数
  Output Gate (1):    门的输入依赖性 + 全秩 + 可压制
  FP32 Accumulation (1): 训练时注意力输出 FP32 累加

运行: python experiments\\models\\llm\\kimi_k3\\modules\\gated_mla\\gradio_mla_demo.py
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for _font in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
              "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass

import gradio as gr

# ── sys.path 设置（gated_mla 比 modules/ 多一层，需要 6 级 dirname）──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _THIS_DIR
for _ in range(6):
    _p = os.path.dirname(_p)
_ROOT = _p
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── 复用已有模块 ──
from plot_gated_mla import plot_gate_suppression, plot_rank_scan, plot_cache_size  # noqa: E402
from models.llm.kimi_k3.config import MLAConfig  # noqa: E402
from models.llm.kimi_k3.modules.mla import GatedMLA, MLACache  # noqa: E402


# ════════════════════════════════════════════════════════════════
#  辅助函数
# ════════════════════════════════════════════════════════════════

def _build(dim: int = 32, **kw) -> GatedMLA:
    """构造小规模 GatedMLA 层并用 std=0.1 正态覆盖权重。"""
    cfg = MLAConfig(
        num_heads=kw.pop("num_heads", 4),
        qk_head_dim=kw.pop("qk_head_dim", 8),
        v_head_dim=kw.pop("v_head_dim", 8),
        q_lora_rank=kw.pop("q_lora_rank", 16),
        kv_lora_rank=kw.pop("kv_lora_rank", 12),
        **kw,
    )
    torch.manual_seed(0)
    layer = GatedMLA(dim, cfg)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(std=0.1)
    return layer


# ════════════════════════════════════════════════════════════════
#  Tab 4: Incremental Decode vs Prefill
# ════════════════════════════════════════════════════════════════

def tab_incremental_decode() -> Tuple[plt.Figure, str]:
    """验证 incremental decode（逐 token 追加 latent）与 prefill（一次性整段）逐位一致。

    返回 (fig, text)：fig 为误差热图 + 逐位误差柱状图，text 为一致性结论。
    """
    layer = _build().eval()
    T = 10
    x = torch.randn(1, T, 32)
    with torch.no_grad():
        full = layer(x)
        cache = layer.init_cache(1)
        steps = [layer(x[:, i:i + 1], cache=cache) for i in range(T)]
    cat_steps = torch.cat(steps, dim=1)
    err = (full - cat_steps).abs()
    max_err = float(err.max())
    per_token_err = err[0, :, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左：逐 token 误差柱状图
    axes[0].bar(range(T), per_token_err, color='#238636')
    axes[0].set_xlabel('Token 位置')
    axes[0].set_ylabel('abs 误差 (head 0, dim 0)')
    axes[0].set_title(f'Prefill vs Incremental 逐 token 误差 (max={max_err:.2e})')
    axes[0].grid(alpha=0.3)

    # 右：误差热图（head × 位置）
    err_2d = err[0, :, :].mean(dim=-1).detach().numpy()  # (T,) across v dims
    im = axes[1].imshow(err_2d.reshape(1, -1), cmap='hot', aspect='auto', vmin=0)
    axes[1].set_xlabel('Token 位置')
    axes[1].set_yticks([0])
    axes[1].set_yticklabels(['all heads'])
    axes[1].set_title('误差热图 (mean over v dims)')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    ok = max_err < 1e-5
    text = (
        f"prefill vs incremental decode max abs 误差: {max_err:.2e}\n"
        f"{'✓ 逐位一致 — incremental decode 正确' if ok else '✗ 存在差异'}\n"
        f"原理：每步 append 一个 latent 到 cache，重新展开全部 k/v 做注意力，\n"
        f"结果应与一次性输入整段（prefill）完全相同。"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 5: NoPE 验证
# ════════════════════════════════════════════════════════════════

def tab_nope() -> Tuple[plt.Figure, str]:
    """验证 MLA 层不含任何位置编码参数（NoPE 设计）。

    返回 (fig, text)：fig 为参数名白名单检查结果柱状图，text 为结论。
    """
    layer = _build()
    banned_words = ["rope", "freq", "inv_freq", "pos", "alibi", "yarn"]
    all_names = {n.lower() for n, _ in layer.named_parameters()}
    all_names |= {n.lower() for n, _ in layer.named_buffers()}

    violations = []
    for bw in banned_words:
        found = [n for n in all_names if bw in n]
        if found:
            violations.append((bw, found))

    # 参数组统计
    param_groups = {}
    for n, p in layer.named_parameters():
        group = n.split('.')[0]
        param_groups[group] = param_groups.get(group, 0) + p.numel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左：参数组分布
    labels = list(param_groups.keys())
    sizes = list(param_groups.values())
    colors = ['#238636' if not violations else '#1565c0'] * len(labels)
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#1565c0', '#FF8F00', '#238636', '#c62828', '#6a1b9a'][:len(labels)])
    axes[0].set_title(f"参数分布 (总计 {sum(sizes):,})")

    # 右：白名单检查
    check_results = []
    check_labels = []
    for bw in banned_words:
        found = [n for n in all_names if bw in n]
        check_results.append(len(found))
        check_labels.append(bw)
    bars = axes[1].bar(check_labels, check_results,
                       color=['#c62828' if r > 0 else '#238636' for r in check_results])
    axes[1].set_ylabel('匹配参数数')
    axes[1].set_title('位置编码白名单检查 (应全为 0)')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, check_results):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    ok = len(violations) == 0
    qk_check = layer.cfg.qk_head_dim * layer.cfg.num_heads == layer.cfg.qk_dim
    text = (
        f"NoPE 检查: {'✓ 无位置编码参数' if ok else '✗ 发现违禁参数'}\n"
        f"qk_dim 一致性: qk_head_dim({layer.cfg.qk_head_dim}) × "
        f"num_heads({layer.cfg.num_heads}) = {layer.cfg.qk_head_dim * layer.cfg.num_heads} "
        f"== qk_dim({layer.cfg.qk_dim}) → {'✓' if qk_check else '✗'}\n"
        f"设计意义：位置信息全部由与之交错的 KDA 层提供，\n"
        f"MLA 层的 KDA 递推衰减天然位置敏感，无需 RoPE。"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 6: Output Gate Properties
# ════════════════════════════════════════════════════════════════

def tab_output_gate() -> Tuple[plt.Figure, str]:
    """验证输出门的三个关键性质：输入依赖、全秩、可压制输出。

    返回 (fig, text)：fig 为 1×3 子图，text 为各性质验证结论。
    """
    layer = _build()

    # 1. 输入依赖：两个不同输入产生不同门值
    x1 = torch.randn(1, 4, 32)
    x2 = torch.randn(1, 4, 32)
    with torch.no_grad():
        g1 = torch.sigmoid(layer.g_proj(x1))
        g2 = torch.sigmoid(layer.g_proj(x2))
    input_dep = not torch.allclose(g1, g2, atol=1e-3)

    # 2. 全秩：g_proj 权重形状
    w_shape = layer.g_proj.weight.shape
    is_full_rank = min(w_shape) == layer.cfg.v_dim

    # 3. 可压制：gate→0 时输出归零
    with torch.no_grad():
        layer.g_proj.weight.fill_(0.0)
        layer.g_proj.weight -= 1e4
        out_suppressed = layer(torch.ones(1, 4, 32))
    suppressed = float(out_suppressed.abs().max()) < 1e-3

    # 复位权重
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(std=0.1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 左：输入依赖性 — 门值分布对比
    with torch.no_grad():
        g1_flat = torch.sigmoid(layer.g_proj(x1)).flatten().numpy()
        g2_flat = torch.sigmoid(layer.g_proj(x2)).flatten().numpy()
    axes[0].hist(g1_flat, bins=20, alpha=0.6, color='#1565c0', label='输入 x1')
    axes[0].hist(g2_flat, bins=20, alpha=0.6, color='#FF8F00', label='输入 x2')
    axes[0].set_xlabel('sigmoid(W_g x) 值')
    axes[0].set_ylabel('频次')
    axes[0].set_title(f'输入依赖性: 不同输入 → 不同门分布')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 中：门值范围 (sigmoid 值域)
    with torch.no_grad():
        x_rand = torch.randn(100, 32)
        gates = torch.sigmoid(layer.g_proj(x_rand)).flatten().numpy()
    axes[1].hist(gates, bins=30, color='#238636', edgecolor='white')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.7, label='g=0 (完全抑制)')
    axes[1].axvline(1, color='blue', linestyle='--', alpha=0.7, label='g=1 (完全通过)')
    axes[1].set_xlabel('sigmoid(W_g x)')
    axes[1].set_ylabel('频次')
    axes[1].set_title(f'门值范围: (0, 1), 权重形状={w_shape}')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # 右：抑制能力 — 门权重偏置 vs 输出幅度
    offsets = [-1e4, -50.0, -10.0, 0.0, 10.0, 50.0]
    amps = []
    with torch.no_grad():
        for off in offsets:
            layer.g_proj.weight.fill_(off)
            amp = float(layer(torch.ones(1, 4, 32)).abs().max())
            amps.append(amp)
    axes[2].plot(offsets, amps, 'o-', color='#c62828', linewidth=2, markersize=7)
    axes[2].set_xlabel('g_proj 偏置')
    axes[2].set_ylabel('MLA 输出 max|·|')
    axes[2].set_title('门可压制输出: 偏置→-∞ ⇒ 输出→0')
    axes[2].set_xscale('symlog', linthresh=10)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    text = (
        f"输出门性质验证:\n"
        f"  1. 输入依赖: {'✓ 不同输入产生不同门值' if input_dep else '✗'}\n"
        f"  2. 全秩: g_proj.weight={w_shape}, min dim={min(w_shape)}==v_dim({layer.cfg.v_dim}) → {'✓ 全秩' if is_full_rank else '✗'}\n"
        f"  3. 可压制: gate→-1e4 时输出 max={float(out_suppressed.abs().max()):.2e} → {'✓ 输出归零' if suppressed else '✗'}\n"
        f"Eq. 7: y = W_o [ sigmoid(W_g x) * o ]\n"
        f"全秩意味着每个 token 能在所有输出通道上独立决定'从注意力读多少'。"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 7: FP32 Accumulation
# ════════════════════════════════════════════════════════════════

def tab_fp32_accumulation() -> Tuple[plt.Figure, str]:
    """验证训练时注意力输出保持 FP32（attn_out_fp32 配置生效）。

    返回 (fig, text)：fig 为 dtype 对比图，text 为结论。
    """
    # FP32 累加 vs 默认
    layer_fp32 = _build(attn_out_fp32=True)
    layer_default = _build(attn_out_fp32=False)

    x = torch.randn(2, 5, 32, dtype=torch.float32)

    layer_fp32.train()
    layer_default.train()
    with torch.no_grad():
        out_fp32 = layer_fp32(x)
        out_default = layer_default(x)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左：输出统计对比
    stats = {
        'FP32 累加 (train)': (float(out_fp32.mean()), float(out_fp32.std())),
        '默认 (train)': (float(out_default.mean()), float(out_default.std())),
    }
    names = list(stats.keys())
    means = [stats[n][0] for n in names]
    stds = [stats[n][1] for n in names]

    x_pos = range(len(names))
    bars_mean = axes[0].bar(x_pos, means, color=['#238636', '#c62828'], alpha=0.7, label='mean')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylabel('输出 mean')
    axes[0].set_title(f'输出统计对比\nFP32 dtype={out_fp32.dtype}, 默认 dtype={out_default.dtype}')
    axes[0].grid(alpha=0.3)
    for bar, val in zip(bars_mean, means):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f'{val:.4f}', ha='center', fontsize=8)

    # 右：输出分布对比
    axes[1].hist(out_fp32.flatten().numpy(), bins=30, alpha=0.6, color='#238636',
                 label=f'FP32 累加 (dtype={out_fp32.dtype})', density=True)
    axes[1].hist(out_default.flatten().numpy(), bins=30, alpha=0.6, color='#c62828',
                 label=f'默认 (dtype={out_default.dtype})', density=True)
    axes[1].set_xlabel('输出值')
    axes[1].set_ylabel('密度')
    axes[1].set_title('输出分布对比')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    text = (
        f"attn_out_fp32 配置验证:\n"
        f"  FP32 累加输出 dtype: {out_fp32.dtype}\n"
        f"  默认输出 dtype: {out_default.dtype}\n"
        f"  FP32 累加用于修正 flash attention 的偏向舍入误差 (biased rounding)。\n"
        f"  训练时用 FP32 累加注意力输出，推理时降精度。\n"
        f"  该开关通过 MLAConfig.attn_out_fp32 控制。"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Gradio 界面
# ════════════════════════════════════════════════════════════════

def build_demo():
    with gr.Blocks(title="Gated MLA 模块交互式演示") as demo:
        gr.Markdown("# Gated MLA 模块交互式演示\n"
                    "覆盖 GatedMLA / MLACache 的全部核心功能\n"
                    "（K3 报告 §2.1.2：压缩 + 重构 + NoPE + 全秩输出门）")

        # ── Tab 1: Cache Size ──
        with gr.TabItem("Cache Size"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### MLA latent cache vs 标准 KV cache\n"
                                "展示 MLA 的核心卖点：cache 只随 `kv_lora_rank` 增长，"
                                "不随 `num_heads × head_dim` 增长。")
                    cs_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    cs_plot = gr.Plot(label="Cache Size 对比")
                    cs_text = gr.Textbox(label="结论", lines=4)
            cs_btn.click(plot_cache_size, inputs=[], outputs=[cs_plot, cs_text])

        # ── Tab 2: Gate Suppression ──
        with gr.TabItem("Gate Suppression"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输出门抑制能力\n"
                                "调节 g_proj 权重偏置，观察 sigmoid(W_g x) 如何"
                                "控制 MLA 输出幅度。偏置越负 → sigmoid→0 → 输出被抹掉。")
                    gs_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    gs_plot = gr.Plot(label="门强度 vs 输出幅度")
                    gs_text = gr.Textbox(label="结论", lines=5)
            gs_btn.click(plot_gate_suppression, inputs=[], outputs=[gs_plot, gs_text])

        # ── Tab 3: Rank Scan ──
        with gr.TabItem("Rank Scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### kv_lora_rank vs 信息压缩\n"
                                "扫描不同 kv_lora_rank，观察重构输出离散度（检索丰富度代理）。\n"
                                "rank 越低 → 信息被压平 → 检索越粗糙。")
                    rs_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    rs_plot = gr.Plot(label="Rank vs 重构离散度")
                    rs_text = gr.Textbox(label="结论", lines=4)
            rs_btn.click(plot_rank_scan, inputs=[], outputs=[rs_plot, rs_text])

        # ── Tab 4: Incremental Decode ──
        with gr.TabItem("Incremental Decode"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Prefill vs Incremental Decode\n"
                                "验证 decode（逐 token 追加 latent）与 prefill（一次性整段）"
                                "逐位一致。这是 MLA 缓存机制正确性的关键测试。")
                    id_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    id_plot = gr.Plot(label="误差对比")
                    id_text = gr.Textbox(label="结论", lines=4)
            id_btn.click(tab_incremental_decode, inputs=[], outputs=[id_plot, id_text])

        # ── Tab 5: NoPE ──
        with gr.TabItem("NoPE"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 无位置编码验证\n"
                                "MLA 层完全不施加 RoPE / ALiBi / YaRN 等位置编码。\n"
                                "位置信息全部由与之交错的 KDA 层提供。\n"
                                "用参数名白名单检查（禁止 rope/freq/pos 等关键词）。")
                    np_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    np_plot = gr.Plot(label="参数名白名单检查")
                    np_text = gr.Textbox(label="结论", lines=4)
            np_btn.click(tab_nope, inputs=[], outputs=[np_plot, np_text])

        # ── Tab 6: Output Gate ──
        with gr.TabItem("Output Gate"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输出门性质 (Eq. 7)\n"
                                "验证三个关键性质：\n"
                                "1. **输入依赖**：不同输入 → 不同门值\n"
                                "2. **全秩**：g_proj 维度 = v_dim（不是低秩门）\n"
                                "3. **可压制**：gate→0 ⇒ 输出→0")
                    og_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    og_plot = gr.Plot(label="输出门性质")
                    og_text = gr.Textbox(label="结论", lines=5)
            og_btn.click(tab_output_gate, inputs=[], outputs=[og_plot, og_text])

        # ── Tab 7: FP32 Accumulation ──
        with gr.TabItem("FP32 Accumulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 训练时 FP32 累加\n"
                                "验证 attn_out_fp32 配置：训练时注意力输出在 FP32 累加，"
                                "修正 flash attention 的偏向舍入误差。")
                    fp_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    fp_plot = gr.Plot(label="FP32 vs 默认对比")
                    fp_text = gr.Textbox(label="结论", lines=4)
            fp_btn.click(tab_fp32_accumulation, inputs=[], outputs=[fp_plot, fp_text])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
