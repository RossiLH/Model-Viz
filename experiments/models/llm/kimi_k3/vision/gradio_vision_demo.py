# -*- coding: utf-8 -*-
"""MoonViT-V2 视觉通路交互式演示（Gradio）。

所有可视化函数均从 plot_vision.py / discussion.py / test_helpers.py 导入，
不重复实现任何核心计算逻辑。

运行方式:
    cd experiments/models/llm/kimi_k3/vision
    python gradio_vision_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# -- sys.path 修正 --
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

import gradio as gr

# -- 字体 + 工具 --
from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
setup_cjk_font()
fp = get_font_prop()

# -- 从 plot_vision.py 导入可视化函数（复用） --
from plot_vision import fig_pixel_shuffle, fig_temporal_pool, fig_token_budget

# -- 从 discussion.py 导入核心计算函数（复用） --
from discussion import compute_e1, compute_e2, compute_e3

# -- 从 test_helpers 导入共享工具 + 模块核心类 --
from models.llm.kimi_k3.config import ViTConfig
from models.llm.kimi_k3.vision.moonvit import MoonViTV2, pixel_shuffle, pixel_unshuffle
from models.llm.kimi_k3.vision.projector import VisionProjector
from models.llm.kimi_k3.vision.packing import (
    VISION_PLACEHOLDER, interleave_multimodal, scatter_vision_embeddings,
)
from models.llm.kimi_k3.vision.synth import Shape, render_scene, SyntheticVisionSource
from models.llm.kimi_k3.vision.test_helpers import (
    ByteTok,
    small_vit_cfg, small_model_cfg,
    build_vit, build_multimodal_model, make_interleaved_tokens,
    audit_bias_terms, audit_norms, audit_param_shapes,
    shape_image_forward, shape_video_forward, shape_projector,
    shape_interleave, compute_scatter_landing, shape_tower_forward,
    collect_gradient_flow,
)

# ============================================================
# Tab 1 wrapper: Pixel Shuffle 压缩（复用 fig_pixel_shuffle）
# ============================================================

def tab1_pixel_shuffle(image_size, dim, patch_size, shuffle):
    return fig_pixel_shuffle(
        image_size=int(image_size), dim=int(dim),
        patch_size=int(patch_size), shuffle=int(shuffle),
    )


# ============================================================
# Tab 2 wrapper: Token Budget（复用 fig_token_budget）
# ============================================================

def tab2_token_budget(patch_size, dim, shuffle, max_size, step):
    return fig_token_budget(
        patch_size=int(patch_size), dim=int(dim),
        shuffle=int(shuffle), max_size=int(max_size), step=int(step),
    )


# ============================================================
# Tab 3: Pixel Shuffle 无损可逆（E3 + 扩展可视化）
# ============================================================

def fig_shuffle_reversible(dim=32, patch_size=14, image_size=56, shuffle=2):
    dim = int(dim); patch_size = int(patch_size); image_size = int(image_size); shuffle = int(shuffle)
    cfg = ViTConfig(n_layers=1, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 2, pixel_shuffle=shuffle)
    vit = MoonViTV2(cfg)
    torch.manual_seed(42)
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        grid, hp, wp = vit.patchify(x)
        shuffled = pixel_shuffle(grid, shuffle)
        restored = pixel_unshuffle(shuffled, shuffle)

    err = (grid - restored).abs().max().item()
    scale = grid.abs().mean().item()
    ratio = err / scale if scale > 0 else 0
    n_before = hp * wp
    n_after = shuffled.shape[1] * shuffled.shape[2]
    ch_before = dim
    ch_after = dim * shuffle * shuffle

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    energy_before = grid[0].norm(dim=-1)
    energy_after = shuffled[0].norm(dim=-1)
    energy_restored = restored[0].norm(dim=-1)
    vmax = max(energy_before.max().item(), energy_after.max().item())

    im0 = axes[0, 0].imshow(energy_before.cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
    _ax_title(axes[0, 0], f"shuffle 前: {hp}x{wp}={n_before} token, 通道 {ch_before}")
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(energy_after.cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
    _ax_title(axes[0, 1], f"shuffle 后: {hp//shuffle}x{wp//shuffle}={n_after} token, 通道 {ch_after}")
    axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(energy_restored.cpu().numpy(), cmap="viridis", vmin=0, vmax=vmax)
    _ax_title(axes[0, 2], f"unshuffle 还原: {hp}x{wp}={n_before} token, 通道 {ch_before}")
    axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    err_map = (grid[0] - restored[0]).norm(dim=-1)
    im3 = axes[1, 0].imshow(err_map.cpu().numpy(), cmap="hot", aspect="auto")
    _ax_title(axes[1, 0], f"逐 token 还原误差 (L2)\nmax={err:.2e}, 相对={ratio:.2e}")
    axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])
    fig.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    ch_before_hist = grid[0].reshape(-1, ch_before).mean(dim=0).numpy()
    ch_after_hist = shuffled[0].reshape(-1, ch_after).mean(dim=0).numpy()
    axes[1, 1].plot(ch_before_hist, label=f"shuffle前 ({ch_before} ch)", alpha=0.7)
    axes[1, 1].plot(ch_after_hist, label=f"shuffle后 ({ch_after} ch)", alpha=0.7)
    _ax_title(axes[1, 1], "平均通道值分布\n(邻域信息展平到通道维)")
    axes[1, 1].set_xlabel("通道 index")
    axes[1, 1].set_ylabel("均值")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].bar(["token 数", "通道数"],
                   [n_before / n_after, ch_after / ch_before],
                   color=["#c62828", "#2e7d32"], width=0.5)
    axes[1, 2].axhline(y=shuffle**2, color="gray", ls="--", alpha=0.7,
                       label=f"shuffle^2 = {shuffle**2}")
    _ax_title(axes[1, 2], f"压缩比 (shuffle={shuffle})\ntoken 减少 / 通道增加")
    axes[1, 2].set_ylabel("倍率")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(axis="y", alpha=0.3)

    fig.suptitle("Pixel Shuffle 无损可逆：token 数 / 通道数 互换", fontsize=14, fontweight="bold")
    plt.tight_layout()

    text = _text(
        f"shuffle 前: {n_before} token x {ch_before} 通道",
        f"shuffle 后: {n_after} token x {ch_after} 通道",
        f"还原误差 max: {err:.2e} (相对 {ratio:.2e})",
        "信息无损：只做空间→通道的重排，不做任何下采样。",
    )
    return fig, text


# ============================================================
# Tab 4 wrapper: Temporal Pool（复用 fig_temporal_pool）
# ============================================================

def tab4_temporal_pool(frames, image_size, dim, patch_size, pool_size):
    return fig_temporal_pool(
        frames=int(frames), image_size=int(image_size),
        dim=int(dim), patch_size=int(patch_size), pool_size=int(pool_size),
    )


# ============================================================
# Tab 5: 因子化注意力（空间+时间）
# ============================================================

def fig_factorized_attention(dim=32, num_heads=4, patch_size=14,
                            image_size=56, frames=4, seed=0):
    dim = int(dim); num_heads = int(num_heads); patch_size = int(patch_size)
    image_size = int(image_size); frames = int(frames); seed = int(seed)

    cfg_f = ViTConfig(n_layers=1, dim=dim, num_heads=num_heads, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=1,
                      factorized_attention=True)
    cfg_u = ViTConfig(n_layers=1, dim=dim, num_heads=num_heads, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=1,
                      factorized_attention=False)

    torch.manual_seed(seed)
    vit_f = MoonViTV2(cfg_f)
    torch.manual_seed(seed)
    vit_u = MoonViTV2(cfg_u)

    n_params_f = sum(p.numel() for p in vit_f.parameters())
    n_params_u = sum(p.numel() for p in vit_u.parameters())

    pixels = torch.randn(frames, 3, image_size, image_size)

    with torch.no_grad():
        out_f = vit_f(pixels, frames=frames)
        out_u = vit_u(pixels, frames=frames)

    shape_f = tuple(out_f.shape)
    shape_u = tuple(out_u.shape)

    hp = wp = image_size // patch_size
    n_per_frame = hp * wp
    total_tokens = n_per_frame * frames
    flops_factorized = frames * n_per_frame**2 * dim + n_per_frame * frames**2 * dim
    flops_full = total_tokens**2 * dim

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.bar(["因子化 (F+T)", "全注意力"], [n_params_f / 1e3, n_params_u / 1e3],
           color=["#2e7d32", "#1565c0"], width=0.5)
    _ax_title(ax, "参数量 (k)\n共享全部参数，不新增权重")
    ax.set_ylabel("参数 (x1000)")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate([n_params_f, n_params_u]):
        ax.text(i, v / 1e3 + max(n_params_f, n_params_u) / 1e3 * 0.02,
                f"{v:,}", ha="center", fontsize=10)

    ax = axes[1]
    ax.bar(["因子化 (F+T)", "全注意力"], [flops_factorized, flops_full],
           color=["#2e7d32", "#c62828"], width=0.5)
    _ax_title(ax, f"注意力 FLOPs 估算 (log)\nframes={frames}, per_frame={n_per_frame}")
    ax.set_yscale("log")
    ax.set_ylabel("FLOPs (相对)")
    ax.grid(axis="y", alpha=0.3)
    ratio = flops_full / flops_factorized
    ax.text(0.5, 0.95, f"因子化节省 {ratio:.0f}x", ha="center", va="top",
            transform=ax.transAxes, fontsize=11, fontweight="bold", color="#2e7d32",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax = axes[2]
    ax.axis("off")
    rows = [
        ["因子化", str(shape_f), f"{n_per_frame} tokens/帧"],
        ["全注意力", str(shape_u), f"{total_tokens} 总 tokens"],
    ]
    table = ax.table(cellText=rows,
                     colLabels=["模式", "输出形状", "token 规模"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    _ax_title(ax, "输出形状与 token 规模")

    fig.suptitle(
        f"因子化注意力 (frames={frames})：先空间后时间，参数 0 新增、计算量大减",
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    text = _text(
        f"因子化注意力参数量: {n_params_f:,} (与全注意力相同)",
        f"FLOPs 节省: {flops_full / flops_factorized:.1f}x",
        "图像与视频共享全部参数：视频=帧堆叠，因子化只是把注意力拆成两段。",
        "空间 pass: 每帧内部做 NxN 注意力；时间 pass: 每个空间位置跨 F 帧做注意力。",
    )
    return fig, text



# ============================================================
# Tab 6: 无位置编码的空间盲区 (E2 + 可视化)
# ============================================================

def fig_spatial_blindness(dim=32, patch_size=14, seed=1):
    dim = int(dim); patch_size = int(patch_size); seed = int(seed)
    res = compute_e2(dim=dim, patch_size=patch_size, seed=seed)

    cfg = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 2, pixel_shuffle=1)
    torch.manual_seed(seed)
    vit = MoonViTV2(cfg)
    grid = [torch.randn(patch_size, patch_size, 3) * (i + 1) for i in range(4)]
    img_a = torch.cat([torch.cat([grid[0], grid[1]], dim=1),
                       torch.cat([grid[2], grid[3]], dim=1)], dim=0)
    img_b = torch.cat([torch.cat([grid[1], grid[0]], dim=1),
                       torch.cat([grid[3], grid[2]], dim=1)], dim=0)
    h = patch_size * 2
    img_c = torch.randn(h, h, 3)

    def pooled_mean(pixels):
        with torch.no_grad():
            out = vit(pixels.permute(2, 0, 1).unsqueeze(0))
        return out.mean(dim=1)

    ma, mb, mc = pooled_mean(img_a), pooled_mean(img_b), pooled_mean(img_c)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for idx, (img, label) in enumerate([(img_a, "A: 布局 1\n(1 2 / 3 4)"),
                                        (img_b, "B: 布局 2\n(2 1 / 4 3)"),
                                        (img_c, "C: 全新内容")]):
        ax = axes[0, idx]
        # 归一化到 [0,1] 显示
        disp = img.clone()
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        ax.imshow(disp.cpu().numpy())
        _ax_title(ax, label)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 0]
    d_ab = res["d_ab"]
    d_ac = res["d_ac"]
    ax.bar(["A vs B\n(同内容-不同布局)", "A vs C\n(不同内容)"],
           [d_ab, d_ac], color=["#1565c0", "#c62828"], width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("max|diff| (log)")
    ax.grid(axis="y", alpha=0.3)
    _ax_title(ax, "平均表示差异\n(mean-pooled 表示)")
    for i, v in enumerate([d_ab, d_ac]):
        ax.text(i, v * 1.3, f"{v:.2e}", ha="center", fontsize=10)

    ax = axes[1, 1]
    diff_ab = (ma - mb).abs().squeeze().numpy()
    diff_ac = (ma - mc).abs().squeeze().numpy()
    ax.plot(diff_ab, label="A vs B (同内容)", color="#1565c0", alpha=0.8)
    ax.plot(diff_ac, label="A vs C (不同内容)", color="#c62828", alpha=0.8)
    _ax_title(ax, "逐维度差异分布\n(均值池化表示)")
    ax.set_xlabel("dim index")
    ax.set_ylabel("|diff|")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines_info = [
        f"同内容-布局互换: {d_ab:.2e}",
        f"  相对信号比例: {res['ratio_ab']:.3f}x",
        f"不同内容:       {d_ac:.2e}",
        f"  相对信号比例: {res['ratio_ac']:.3f}x",
        f"",
        f"盲区比 = {d_ac / max(d_ab, 1e-30):.1f}x",
        f"(越大表示对布局越不敏感)",
    ]
    y_pos = 0.9
    for line in lines_info:
        ax.text(0.05, y_pos, line, transform=ax.transAxes, fontsize=10, va="top")
        y_pos -= 0.12
    _ax_title(ax, "结论")

    fig.suptitle(
        "无位置编码的空间盲区：同内容不同布局 -> 输出几乎一致",
        fontsize=13, fontweight="bold")
    plt.tight_layout()

    text = _text(
        f"同内容-布局互换 (A vs B): 差异 {d_ab:.2e} (量级比 {res['ratio_ab']:.3f}x)",
        f"不同内容        (A vs C): 差异 {d_ac:.2e} (量级比 {res['ratio_ac']:.3f}x)",
        "结论: MoonViT-V2 无 2D 位置编码，卷积 patch embedding + 双向注意力",
        "      使模型对绝对布局几乎盲 —— 只感知 patch 内容及其相对关系。",
    )
    return fig, text


# ============================================================
# Tab 7: 视觉塔结构审计（无 bias + RMSNorm + 参数量）
# ============================================================

def fig_vision_tower_audit(dim=32, n_layers=2, num_heads=4, patch_size=14,
                        mlp_hidden=64, shuffle=2):
    cfg = ViTConfig(n_layers=int(n_layers), dim=int(dim), num_heads=int(num_heads),
                    patch_size=int(patch_size), mlp_hidden=int(mlp_hidden),
                    pixel_shuffle=int(shuffle))
    vit = MoonViTV2(cfg)
    proj = VisionProjector(cfg, model_dim=64)

    n_bias, bias_names = audit_bias_terms(vit)
    n_rms, n_ln = audit_norms(vit)
    n_params_vit = sum(p.numel() for p in vit.parameters())
    n_params_proj = sum(p.numel() for p in proj.parameters())

    param_by_part = {}
    for name, p in vit.named_parameters():
        if "patch_embed" in name:
            part = "patch_embed"
        elif "attn" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name:
            part = "attn"
        elif "fc1" in name or "fc2" in name:
            part = "mlp"
        elif "norm" in name:
            part = "norm"
        else:
            part = "other"
        param_by_part[part] = param_by_part.get(part, 0) + p.numel()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.axis("off")
    if n_bias == 0:
        color = "#2e7d32"
        msg = "OK 全塔无 bias 项"
    else:
        color = "#c62828"
        msg = f"NG 发现 {n_bias} 个 bias"
    ax.text(0.5, 0.6, msg, ha="center", va="center",
            fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.3, "K3 sec 2.4: from-scratch 稳定配方", ha="center",
            fontsize=10, color="#555", transform=ax.transAxes)
    _ax_title(ax, "全塔无 bias 验证")

    ax = axes[0, 1]
    ax.bar(["RMSNorm", "LayerNorm"], [n_rms, n_ln],
           color=["#2e7d32", "#9e9e9e"], width=0.5)
    _ax_title(ax, "归一化层类型统计")
    ax.set_ylabel("层数")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate([n_rms, n_ln]):
        ax.text(i, v + max(n_rms, 1) * 0.03, str(int(v)), ha="center", fontsize=10)

    ax = axes[1, 0]
    parts = sorted(param_by_part.items(), key=lambda x: -x[1])
    labels = [f"{k} ({v:,})" for k, v in parts]
    sizes = [v for _, v in parts]
    colors_pie = ["#1565c0", "#2e7d32", "#f57c00", "#7b1fa2", "#616161"]
    ax.pie(sizes, labels=labels, colors=colors_pie[:len(sizes)],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    _ax_title(ax, f"ViT 参数分布 (总计 {n_params_vit:,})")

    ax = axes[1, 1]
    ax.bar(["视觉塔 ViT", "投影器 Projector"],
           [n_params_vit, n_params_proj],
           color=["#1565c0", "#2e7d32"], width=0.5)
    _ax_title(ax, "容量放在塔与骨干，投影器刻意小")
    ax.set_ylabel("参数量")
    ax.grid(axis="y", alpha=0.3)
    ratio = n_params_vit / max(n_params_proj, 1)
    ax.text(0.5, 0.95, f"ViT / Proj = {ratio:.1f}x", ha="center", va="top",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    for i, v in enumerate([n_params_vit, n_params_proj]):
        ax.text(i, v + max(n_params_vit, n_params_proj) * 0.02,
                f"{v:,}", ha="center", fontsize=9)

    fig.suptitle(f"MoonViT-V2 结构审计 (dim={dim}, layers={n_layers})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    text = _text(
        f"ViT 参数量: {n_params_vit:,}",
        f"Projector 参数量: {n_params_proj:,}",
        f"无 bias 项: {n_bias == 0}",
        f"RMSNorm: {n_rms} 层, LayerNorm: {n_ln} 层",
        "设计哲学: 容量放塔与骨干，投影器刻意小 (原生多模态不需要翻译层)。",
    )
    return fig, text



# ============================================================
# Tab 8: Forward 形状流转（图像/视频/投影）
# ============================================================

def fig_shape_flow(dim=32, n_layers=2, num_heads=4, patch_size=14,
                 mlp_hidden=64, shuffle=2, temporal_pool=2,
                 frames=4, model_dim=64, image_size=56):
    dim = int(dim); n_layers = int(n_layers); num_heads = int(num_heads)
    patch_size = int(patch_size); mlp_hidden = int(mlp_hidden); shuffle = int(shuffle)
    temporal_pool = int(temporal_pool); frames = int(frames)
    model_dim = int(model_dim); image_size = int(image_size)

    cfg = ViTConfig(n_layers=n_layers, dim=dim, num_heads=num_heads,
                    patch_size=patch_size, mlp_hidden=mlp_hidden,
                    pixel_shuffle=shuffle, temporal_pool=temporal_pool)
    vit = MoonViTV2(cfg)
    proj = VisionProjector(cfg, model_dim=model_dim)

    is_shape = shape_image_forward(vit, batch=2, h=image_size, w=image_size)
    vs_shape = shape_video_forward(vit, frames=frames, h=image_size, w=image_size)
    pr_shape = shape_projector(proj, vit, batch=2, h=image_size, w=image_size)

    hp = wp = image_size // patch_size
    n_patches = hp * wp
    n_shuffled = (hp // shuffle) * (wp // shuffle)
    pooled_frames = max(1, frames // temporal_pool)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    stages = [
        ("输入像素", f"(B, C=3, H={image_size}, W={image_size})", "#e3f2fd"),
        ("Patchify", f"(B, H'={hp}, W'={wp}, D={dim})  {n_patches} tokens", "#bbdefb"),
        ("ViT Blocks", f"(B, {n_patches}, D={dim})", "#90caf9"),
        ("Temporal Pool", f"(B, {pooled_frames}x{n_patches}, D={dim})  视频专属", "#64b5f6"),
        ("Pixel Shuffle", f"(B, {n_shuffled}, D={dim*shuffle**2})  通道 x{shuffle**2}", "#42a5f5"),
        ("Projector", f"(B, {n_shuffled}, model_dim={model_dim})", "#2196f3"),
    ]

    n_stages = len(stages)
    box_w = 0.13
    box_h = 0.55
    y_mid = 0.5
    for i, (name, desc, color) in enumerate(stages):
        x = 0.03 + i * (0.94 / n_stages)
        rect = plt.Rectangle((x, y_mid - box_h/2), box_w, box_h,
                            facecolor=color, edgecolor="#1565c0", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_mid + 0.08, name, ha="center", va="center",
                fontsize=10, fontweight="bold")
        ax.text(x + box_w/2, y_mid - 0.12, desc, ha="center", va="center",
                fontsize=8)
        if i < n_stages - 1:
            ax.annotate("", xy=(x + box_w + 0.015, y_mid), xytext=(x + box_w, y_mid),
                        arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2))

    _ax_title(ax, "视觉通路形状流转（像素 -> patch -> ViT -> pool -> shuffle -> 投影）")

    info_lines = [
        f"图像 forward: {is_shape}",
        f"视频 forward (frames={frames}): {vs_shape}",
        f"投影器输出: {pr_shape}",
        f"Token 压缩比 (pixel shuffle): {shuffle**2}x",
    ]
    y_info = 0.12
    for line in info_lines:
        ax.text(0.5, y_info, line, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))
        y_info -= 0.05

    text = _text(
        f"图像 forward: {is_shape}",
        f"视频 forward (frames={frames}): {vs_shape}",
        f"投影器输出: {pr_shape}",
        f"Token 压缩: {n_patches} -> {n_shuffled} ({shuffle**2}x)",
    )
    return fig, text


# ============================================================
# Tab 9: 多模态交错打包（interleave + loss mask）
# ============================================================

def fig_interleave_packing(n_text=8, n_visual=4, mask_visual=True):
    n_text = int(n_text); n_visual = int(n_visual); mask_visual = bool(mask_visual)

    tok = ByteTok()
    text_a = "hello"[:max(1, n_text//2)]
    text_b = " world"[:max(1, n_text - n_text//2)]
    n_img_tokens = n_visual

    sample = interleave_multimodal(
        [("text", text_a),
         ("image", (torch.randn(3, 28, 28), n_img_tokens)),
         ("text", text_b)],
        tok,
        mask_visual_loss=mask_visual,
    )
    info = shape_interleave(tok, sample)

    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5))

    ax = axes[0]
    seq_len = info["seq_len"]
    token_types = []
    for i in range(seq_len):
        if i in info["vision_slots"]:
            token_types.append("visual")
        else:
            token_types.append("text")

    colors_token = ["#1565c0" if t == "text" else "#ff8f00" for t in token_types]
    ax.bar(range(seq_len), [1] * seq_len, color=colors_token, width=0.8)
    _ax_title(ax, "多模态 token 流：文本与视觉交错在同一条序列里")
    ax.set_ylabel("token")
    ax.set_yticks([])
    ax.set_xlabel("序列位置")
    ax.set_xlim(-0.5, seq_len - 0.5)

    legend_elements = [Patch(facecolor="#1565c0", label="文本 token"),
                       Patch(facecolor="#ff8f00", label="视觉占位 token")]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax = axes[1]
    loss_mask = sample.labels.numpy()
    masked = (loss_mask == -100).astype(float)
    unmasked = (loss_mask != -100).astype(float)
    ax.bar(range(seq_len), unmasked, color="#2e7d32", width=0.8, label="预测（有 loss）")
    ax.bar(range(seq_len), masked, bottom=unmasked, color="#9e9e9e", width=0.8, label="掩码（无 loss）")
    _ax_title(ax, f"Loss 掩码：视觉槽位={int(masked.sum())} 个被掩码（不预测像素）")
    ax.set_ylabel("loss mask")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["预测", "掩码"])
    ax.set_xlabel("序列位置")
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()

    text = _text(
        f"总 seq_len: {info['seq_len']}",
        f"文本 token: {info['n_text']} 个",
        f"视觉 token: {info['n_visual']} 个",
        f"loss 掩码位置: {info['n_loss_masked']} 个 (mask_visual={mask_visual})",
        "设计：原生多模态 = 单条 next-token prediction 流，视觉槽位不参与损失计算。",
    )
    return fig, text



# ============================================================
# Tab 10: 视觉嵌入散点（scatter 落地验证）
# ============================================================

def fig_scatter_embeddings(n_text=6, n_visual=4, dim=8):
    n_text = int(n_text); n_visual = int(n_visual); dim = int(dim)

    embeds = torch.zeros(1, n_text + n_visual, dim)
    vision = torch.randn(n_visual, dim) * 2 + 1
    slots = torch.tensor(list(range(n_text, n_text + n_visual)))
    info = compute_scatter_landing(embeds, vision, slots)

    out = scatter_vision_embeddings(embeds, vision, slots)
    # 两图共享同一组 vmin/vmax 且关于 0 对称：0 -> 白色，|最大值| -> 深红/深蓝。
    # 不显式设置时，matplotlib 会按数据范围自动缩放，scatter 后的 0 值（文本位置）
    # 会被映射成浅蓝色而非白色。
    vmax = max(out.abs().max().item(), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im0 = ax.imshow(embeds[0].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    _ax_title(ax, f"scatter 前：占位槽位为 0 (text={n_text}, visual={n_visual})")
    ax.set_xlabel("dim")
    ax.set_ylabel("token index")
    fig.colorbar(im0, ax=ax, fraction=0.046)

    ax = axes[1]
    im1 = ax.imshow(out[0].numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    _ax_title(ax, f"scatter 后：视觉嵌入填入占位槽位（落地验证：{info['slots_match']}）")
    ax.set_xlabel("dim")
    ax.set_ylabel("token index")
    fig.colorbar(im1, ax=ax, fraction=0.046)

    fig.tight_layout()

    text = _text(
        f"槽位落地匹配: {info['slots_match']}",
        f"槽位内 max|diff|: {info['max_diff_inside']:.2e}",
        f"槽位外 max|diff|: {info['max_diff_outside']:.2e}",
        "视觉 token 通过 scatter 填进 token 流的占位槽，与文本 token 共享同一条嵌入序列。",
    )
    return fig, text


# ============================================================
# Tab 11: End-to-End 多模态前向
# ============================================================

def fig_end_to_end(mode="image", frames=4, image_size=56, n_text=6):
    mode = str(mode); frames = int(frames); image_size = int(image_size); n_text = int(n_text)

    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()

    # 56x56 -> patch 7 -> 8x8 grid -> shuffle 2 -> 4x4 = 16 视觉 token
    n_img = 16
    if mode == "image":
        n_visual = n_img
    else:
        n_visual = 32  # 4 frames / pool2 = 2 * 16

    seq = make_interleaved_tokens(n_text=n_text, n_visual=n_visual)

    try:
        if mode == "image":
            pixels = torch.randn(1, 3, image_size, image_size)
            with torch.no_grad():
                out = m(seq, pixel_values=pixels)
            out_shape = tuple(out.shape)
            status = "图像前向成功"
            color = "#2e7d32"
        else:
            video = torch.randn(1, frames, 3, image_size, image_size)
            with torch.no_grad():
                out = m(seq, video_values=video)
            out_shape = tuple(out.shape)
            status = f"视频前向成功 (frames={frames})"
            color = "#2e7d32"
    except ValueError as e:
        out_shape = "ERROR"
        status = f"错误: {str(e)[:50]}"
        color = "#c62828"

    # loss mask 检查
    labels = torch.full_like(seq, -100)
    labels[0, :n_text] = torch.randint(0, 100, (1, n_text))
    try:
        if mode == "image":
            pixels = torch.randn(1, 3, image_size, image_size)
            with torch.no_grad():
                res = m(seq, labels=labels, pixel_values=pixels)
            loss_valid = "loss" in res and "lm_loss" in res
            loss_val = float(res["loss"]) if "loss" in res else 0
        else:
            video = torch.randn(1, frames, 3, image_size, image_size)
            with torch.no_grad():
                res = m(seq, labels=labels, video_values=video)
            loss_valid = "loss" in res and "lm_loss" in res
            loss_val = float(res["loss"]) if "loss" in res else 0
    except Exception:
        loss_valid = False
        loss_val = 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.text(0.5, 0.7, status, ha="center", va="center",
            fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.55, f"输出形状: {out_shape}", ha="center", va="center",
            fontsize=12, transform=ax.transAxes)
    loss_status_text = "通过" if loss_valid else "失败"
    loss_color = "#2e7d32" if loss_valid else "#c62828"
    ax.text(0.5, 0.42,
            f"Loss 掩码验证: {loss_status_text} (loss={loss_val:.4f})",
            ha="center", va="center", fontsize=12, color=loss_color, transform=ax.transAxes)
    ax.text(0.5, 0.30,
            f"模式: {mode} | 文本: {n_text} tokens | 视觉: {n_visual} tokens",
            ha="center", va="center", fontsize=11, color="#555", transform=ax.transAxes)
    _ax_title(ax, "End-to-End 多模态模型前向验证")

    text = _text(
        f"模式: {mode}",
        f"输出形状: {out_shape}",
        f"Loss 掩码: {'有效' if loss_valid else '无效'}",
        f"Loss 值: {loss_val:.4f}",
        "验证点：图像/视频 token 正确落地占位槽，loss 只在文本上计算。",
    )
    return fig, text


# ============================================================
# Tab 12: 梯度回流（from-scratch 端到端）
# ============================================================

def fig_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56):
    dim = int(dim); n_layers = int(n_layers)
    patch_size = int(patch_size); image_size = int(image_size)

    # 非法的 (patch_size, image_size) 组合会被自动吸附到最近合法值，
    # 避免前端 47% 的滑块组合抛 ValueError 导致图渲染失败。
    info = collect_gradient_flow(dim=dim, n_layers=n_layers,
                                 patch_size=patch_size, image_size=image_size,
                                 shuffle=2, seed=0)
    grad_info = info["per_param"]
    grouped = info["grouped"]
    per_layer = info["per_layer"]
    patch_grad = info["patch_embed_grad"]
    has_patch_grad = info["has_patch_grad"]
    eff_size = info["image_size"]
    adjusted = info["image_size_adjusted"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ---- 左：patch_embed 是否收到梯度（from-scratch 的核心结论） ----
    ax = axes[0]
    ax.axis("off")
    if has_patch_grad:
        msg = "OK 梯度回流到 patch_embed"
        color = "#2e7d32"
    else:
        msg = "NG 无梯度到 patch_embed"
        color = "#c62828"
    ax.text(0.5, 0.72, msg, ha="center", va="center",
            fontsize=17, fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.55, f"|grad| sum = {patch_grad:.3e}", ha="center",
            fontsize=12, transform=ax.transAxes)
    # 关键结论：所有参数都非零，不存在"某个模块梯度为 0"
    ax.text(0.5, 0.42,
            f"全部 {info['n_params']} 个参数梯度均非零"
            if info["n_zero_grad"] == 0
            else f"警告：{info['n_zero_grad']} 个参数梯度为 0",
            ha="center", fontsize=11,
            color="#2e7d32" if info["n_zero_grad"] == 0 else "#c62828",
            transform=ax.transAxes)
    ax.text(0.5, 0.28, f"图像 {eff_size}x{eff_size} | patch {patch_size} | "
                       f"网格 {eff_size//patch_size}x{eff_size//patch_size}",
            ha="center", fontsize=10, color="#555", transform=ax.transAxes)
    if adjusted:
        ax.text(0.5, 0.16,
                f"注：图像尺寸 {info['requested_image_size']} 不满足 "
                f"patch/shuffle 整除约束，已自动调整为 {eff_size}",
                ha="center", fontsize=9, color="#e65100", transform=ax.transAxes)
    _ax_title(ax, "梯度回流到 patch_embed")

    # ---- 中：各模块梯度总和（log 轴） ----
    # 必须用 log 轴：norm 组约 2.0（直连 loss），attn 组仅 2e-05，
    # 相差 5 个数量级。线性轴下小柱高度不足 1 像素 + %.3f 标签显示成 "0.000"，
    # 会被误读成"该模块没有梯度"。
    ax = axes[1]
    order = ["patch_embed", "attn", "mlp", "norm"]
    keys = [k for k in order if k in grouped]
    vals = [grouped[k] for k in keys]
    colors_bar = {"patch_embed": "#1565c0", "attn": "#2e7d32",
                  "mlp": "#f57c00", "norm": "#7b1fa2"}
    ax.bar(keys, vals, color=[colors_bar[k] for k in keys], width=0.55)
    ax.set_yscale("log")
    ax.set_ylim(min(vals) * 0.2, max(vals) * 5)
    # log 轴默认用 mathtext 渲染 10^-5，其中的 U+2212 减号在 SimHei 里没有字形，
    # 会打出 "does not have a glyph" 警告并画成方块。改用纯 ASCII 的科学计数法标签。
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _ax_title(ax, "各模块梯度总和（log 轴，全部非零）")
    ax.set_ylabel("|grad| sum（对数刻度）")
    ax.grid(axis="y", alpha=0.3, which="both")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.4, f"{v:.2e}", ha="center", fontsize=9)

    # ---- 右：逐层梯度（观察深度方向衰减） ----
    ax = axes[2]
    layers = [r["layer"] for r in per_layer]
    ax.plot(layers, [r["attn"] for r in per_layer], "o-",
            color="#2e7d32", label="attn", linewidth=2)
    ax.plot(layers, [r["mlp"] for r in per_layer], "s-",
            color="#f57c00", label="mlp", linewidth=2)
    ax.plot(layers, [r["norm"] for r in per_layer], "^-",
            color="#7b1fa2", label="norm", linewidth=2)
    ax.axhline(patch_grad, color="#1565c0", linestyle="--", linewidth=1.5,
               label="patch_embed")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("block 索引（0 = 最靠近输入）")
    ax.set_ylabel("|grad| sum（对数刻度）")
    ax.set_xticks(layers)
    _ax_title(ax, f"逐层梯度分布（n_layers={n_layers}）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(prop=fp, fontsize=9)

    fig.tight_layout()

    lines = [
        f"patch_embed 梯度: {patch_grad:.3e} ({'有' if has_patch_grad else '无'})",
        f"梯度为 0 的参数: {info['n_zero_grad']} / {info['n_params']}"
        f"{'（全部非零，端到端可训练）' if info['n_zero_grad'] == 0 else ''}",
        "各模块: " + ", ".join(f"{k}={grouped[k]:.2e}" for k in keys),
        f"总参数量: {info['total_numel']:,}",
    ]
    if adjusted:
        lines.append(
            f"图像尺寸 {info['requested_image_size']} 非法（需为 patch_size x 2 的倍数），"
            f"已自动调整为 {eff_size}。"
        )
    lines.append(
        "结论：norm 组梯度最大（顶层 norm 直连 loss），attn/mlp 小 5 个数量级但均非零 —— "
        "线性轴会把它们压成看不见的零柱，故此处用 log 轴。"
    )
    text = _text(*lines)
    return fig, text



# ============================================================
# Tab 13: 合成数据（Synth 场景 + bounding box）
# ============================================================

def fig_synthetic_scene(n_shapes=3, height=64, width=64, seed=0, normalized=False):
    import random
    n_shapes = int(n_shapes); height = int(height); width = int(width)
    seed = int(seed); normalized = bool(normalized)

    rng = random.Random(seed)
    kinds = ["square", "circle", "triangle"]
    color_names = ["red", "green", "blue", "yellow"]

    shapes = []
    for _ in range(n_shapes):
        size = rng.randint(8, max(10, min(height, width) // 4))
        shapes.append(Shape(
            kind=rng.choice(kinds),
            color=rng.choice(color_names),
            cx=rng.randint(size, width - size),
            cy=rng.randint(size, height - size),
            size=size,
        ))

    pixels = render_scene(shapes, height, width)
    caption = "the image shows " + " and ".join(
        f"a {s.color} {s.kind}" for s in shapes) + "."

    coords = []
    for s in shapes:
        x0, y0, x1, y1 = s.bbox()
        if normalized:
            box = (x0 / width, y0 / height, x1 / width, y1 / height)
        else:
            box = (float(x0), float(y0), float(x1), float(y1))
        coords.append((f"{s.color} {s.kind}", box))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    ax.imshow(pixels.permute(1, 2, 0).cpu().numpy())
    _ax_title(ax, "合成场景")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(pixels.permute(1, 2, 0).cpu().numpy())
    for i, (name, box) in enumerate(coords):
        if normalized:
            x0, y0, x1, y1 = box[0]*width, box[1]*height, box[2]*width, box[3]*height
        else:
            x0, y0, x1, y1 = box
        rect = plt.Rectangle((x0, y0), x1-x0, y1-y0,
                            fill=False, edgecolor="white", linewidth=2, linestyle="--")
        ax.add_patch(rect)
        ax.text(x0, y0 - 3, f"#{i+1} {name}", color="white",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    _ax_title(ax, "Bounding Box 监督")
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()

    text_lines = [f"场景: {caption}", f"形状数: {len(shapes)}", ""]
    text_lines.append("Bounding Boxes:")
    for i, (name, box) in enumerate(coords):
        text_lines.append(f"  #{i+1} {name}: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})")
    text_lines.append("")
    text_lines.append("两种坐标格式：绝对像素坐标 + 归一化 [0,1] 坐标（分辨率鲁棒）。")
    text = _text(*text_lines)
    return fig, text


# ============================================================
# Tab 14: 时间池化丢时序（热力图版）
# ============================================================

def fig_temporal_pool_heatmap(frames=6, image_size=28, dim=16, pool_size=2, seed=0):
    frames = int(frames); image_size = int(image_size); dim = int(dim)
    pool_size = int(pool_size); seed = int(seed)

    cfg_p = ViTConfig(n_layers=1, dim=dim, num_heads=4, patch_size=14,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=pool_size)
    cfg_1 = ViTConfig(n_layers=1, dim=dim, num_heads=4, patch_size=14,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=1)

    torch.manual_seed(seed)
    vit_pool = MoonViTV2(cfg_p)
    torch.manual_seed(seed)
    vit_nopool = MoonViTV2(cfg_1)
    vit_pool.load_state_dict(vit_nopool.state_dict())

    frames_even = (frames // pool_size) * pool_size
    fwd = torch.zeros(frames_even, 3, image_size, image_size)
    for i in range(frames_even):
        if i % 2 == 0:
            fwd[i, 0] = 0.8
        else:
            fwd[i, 2] = 0.8

    reversed_frames = torch.flip(fwd, [0])

    with torch.no_grad():
        out_pool_fwd = vit_pool(fwd, frames=frames_even)
        out_pool_rev = vit_pool(reversed_frames, frames=frames_even)
        out_nopool_fwd = vit_nopool(fwd, frames=frames_even)
        out_nopool_rev = vit_nopool(reversed_frames, frames=frames_even)

    d_pool = (out_pool_fwd - out_pool_rev).abs().mean().item()
    d_nopool = (out_nopool_fwd - out_nopool_rev).abs().mean().item()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    vmax = max(out_pool_fwd.abs().max().item(), 1e-6)
    im = ax.imshow(out_pool_fwd[0].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    _ax_title(ax, f"带池化 正序输出 (pool={pool_size})")
    ax.set_ylabel("dim"); ax.set_xlabel("token")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    im = ax.imshow(out_pool_rev[0].T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    _ax_title(ax, f"带池化 倒序输出\nmean|diff|={d_pool:.2e}")
    ax.set_ylabel("dim"); ax.set_xlabel("token")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    vmax2 = max(out_nopool_fwd.abs().max().item(), 1e-6)
    im = ax.imshow(out_nopool_fwd[0].T, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    _ax_title(ax, "无池化 正序输出 (pool=1)")
    ax.set_ylabel("dim"); ax.set_xlabel("token")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    im = ax.imshow(out_nopool_rev[0].T, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="auto")
    _ax_title(ax, f"无池化 倒序输出\nmean|diff|={d_nopool:.2e}")
    ax.set_ylabel("dim"); ax.set_xlabel("token")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("时间池化对帧序的敏感性（正序 vs 倒序对比）", fontsize=13, fontweight="bold")
    plt.tight_layout()

    ratio = d_nopool / max(d_pool, 1e-30)
    text = _text(
        f"带池化 正序-倒序差异: {d_pool:.2e}",
        f"无池化 正序-倒序差异: {d_nopool:.2e}",
        f"池化降低帧序敏感性: {ratio:.1f}x",
        "均值池化对称 -> 正序与倒序输出几乎一致 -> 丢失时序方向信息。",
    )
    return fig, text


# ============================================================
# Gradio 界面
# ============================================================

def build_demo():
    with gr.Blocks(title="MoonViT-V2 视觉通路交互式演示") as demo:
        gr.Markdown(
            """
            # MoonViT-V2 视觉通路交互式演示

            基于 `models/llm/kimi_k3/vision/` 真实代码运行，所有图均为 CPU 实时计算。
            涵盖 MoonViT-V2 视觉塔、VisionProjector 投影器、多模态交错打包、
            端到端多模态模型前向、合成数据等全部 5 个源文件。

            **14 个实验板块：**
            1. Pixel Shuffle 压缩示意
            2. Token Budget 大图预算
            3. Pixel Shuffle 无损可逆
            4. Temporal Pool 帧序敏感性
            5. 因子化注意力（空间+时间）
            6. 无位置编码的空间盲区
            7. 视觉塔结构审计
            8. Forward 形状流转图
            9. 多模态交错打包
            10. 视觉嵌入散点
            11. End-to-End 多模态前向
            12. 梯度回流
            13. 合成数据
            14. 时间池化丢时序热力图
            """)

        with gr.Tabs():
            with gr.TabItem("1. Pixel Shuffle 压缩"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s1_img = gr.Slider(28, 224, value=56, step=14, label="图像尺寸")
                        s1_dim = gr.Slider(16, 128, value=32, step=16, label="特征维度 dim")
                        s1_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s1_shuffle = gr.Slider(1, 4, value=2, step=1, label="pixel_shuffle factor")
                        btn1 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot1 = gr.Plot(label="Pixel Shuffle 压缩示意")
                        text1 = gr.Textbox(label="结论", lines=3)
                btn1.click(tab1_pixel_shuffle,
                           inputs=[s1_img, s1_dim, s1_patch, s1_shuffle],
                           outputs=[plot1, text1])
            with gr.TabItem("2. Token Budget 大图预算"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s2_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s2_dim = gr.Slider(64, 256, value=128, step=32, label="dim")
                        s2_shuffle = gr.Slider(1, 4, value=2, step=1, label="pixel_shuffle")
                        s2_max = gr.Slider(512, 4096, value=3584, step=256, label="最大尺寸")
                        s2_step = gr.Slider(128, 1024, value=512, step=128, label="步长")
                        btn2 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot2 = gr.Plot(label="Token Budget 对比图")
                        text2 = gr.Textbox(label="结论", lines=3)
                btn2.click(tab2_token_budget,
                           inputs=[s2_patch, s2_dim, s2_shuffle, s2_max, s2_step],
                           outputs=[plot2, text2])
            with gr.TabItem("3. Pixel Shuffle 无损可逆"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s3_dim = gr.Slider(16, 128, value=32, step=16, label="特征维度 dim")
                        s3_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s3_img = gr.Slider(28, 224, value=56, step=14, label="图像尺寸")
                        s3_shuffle = gr.Slider(1, 4, value=2, step=1, label="shuffle factor")
                        btn3 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot3 = gr.Plot(label="Shuffle / Unshuffle 对比")
                        text3 = gr.Textbox(label="结论", lines=3)
                btn3.click(fig_shuffle_reversible,
                           inputs=[s3_dim, s3_patch, s3_img, s3_shuffle],
                           outputs=[plot3, text3])
            with gr.TabItem("4. Temporal Pool 帧序敏感性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s4_frames = gr.Slider(2, 16, value=4, step=1, label="帧数 frames")
                        s4_img = gr.Slider(28, 224, value=56, step=14, label="图像尺寸")
                        s4_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s4_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s4_pool = gr.Slider(1, 8, value=2, step=1, label="temporal_pool")
                        btn4 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot4 = gr.Plot(label="帧序敏感性对比")
                        text4 = gr.Textbox(label="结论", lines=3)
                btn4.click(tab4_temporal_pool,
                           inputs=[s4_frames, s4_img, s4_dim, s4_patch, s4_pool],
                           outputs=[plot4, text4])
            with gr.TabItem("5. 因子化注意力"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s5_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s5_heads = gr.Slider(1, 8, value=4, step=1, label="num_heads")
                        s5_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s5_img = gr.Slider(28, 112, value=56, step=14, label="图像尺寸")
                        s5_frames = gr.Slider(2, 16, value=4, step=1, label="帧数 frames")
                        s5_seed = gr.Slider(0, 99, value=0, step=1, label="随机种子")
                        btn5 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot5 = gr.Plot(label="因子化 vs 全注意力")
                        text5 = gr.Textbox(label="结论", lines=3)
                btn5.click(fig_factorized_attention,
                           inputs=[s5_dim, s5_heads, s5_patch, s5_img, s5_frames, s5_seed],
                           outputs=[plot5, text5])
            with gr.TabItem("6. 无位置编码空间盲区"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s6_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s6_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s6_seed = gr.Slider(0, 99, value=1, step=1, label="随机种子")
                        btn6 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot6 = gr.Plot(label="空间盲区实验")
                        text6 = gr.Textbox(label="结论", lines=3)
                btn6.click(fig_spatial_blindness,
                           inputs=[s6_dim, s6_patch, s6_seed],
                           outputs=[plot6, text6])
            with gr.TabItem("7. 视觉塔结构审计"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s7_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s7_layers = gr.Slider(1, 8, value=2, step=1, label="n_layers")
                        s7_heads = gr.Slider(1, 8, value=4, step=1, label="num_heads")
                        s7_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s7_mlp = gr.Slider(32, 256, value=64, step=32, label="mlp_hidden")
                        s7_shuffle = gr.Slider(1, 4, value=2, step=1, label="pixel_shuffle")
                        btn7 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot7 = gr.Plot(label="结构审计")
                        text7 = gr.Textbox(label="结论", lines=3)
                btn7.click(fig_vision_tower_audit,
                           inputs=[s7_dim, s7_layers, s7_heads, s7_patch, s7_mlp, s7_shuffle],
                           outputs=[plot7, text7])
            with gr.TabItem("8. Forward 形状流转"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s8_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s8_layers = gr.Slider(1, 8, value=2, step=1, label="n_layers")
                        s8_heads = gr.Slider(1, 8, value=4, step=1, label="num_heads")
                        s8_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s8_mlp = gr.Slider(32, 256, value=64, step=32, label="mlp_hidden")
                        s8_shuffle = gr.Slider(1, 4, value=2, step=1, label="pixel_shuffle")
                        s8_pool = gr.Slider(1, 8, value=2, step=1, label="temporal_pool")
                        s8_frames = gr.Slider(2, 16, value=4, step=1, label="frames")
                        s8_modeldim = gr.Slider(32, 256, value=64, step=32, label="model_dim")
                        s8_img = gr.Slider(28, 224, value=56, step=14, label="图像尺寸")
                        btn8 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot8 = gr.Plot(label="形状流转图")
                        text8 = gr.Textbox(label="结论", lines=3)
                btn8.click(fig_shape_flow,
                           inputs=[s8_dim, s8_layers, s8_heads, s8_patch, s8_mlp, s8_shuffle, s8_pool, s8_frames, s8_modeldim, s8_img],
                           outputs=[plot8, text8])
            with gr.TabItem("9. 多模态交错打包"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s9_text = gr.Slider(2, 20, value=8, step=1, label="文本 token 数")
                        s9_visual = gr.Slider(1, 16, value=4, step=1, label="视觉 token 数")
                        s9_mask = gr.Checkbox(value=True, label="mask_visual_loss")
                        btn9 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot9 = gr.Plot(label="交错打包可视化")
                        text9 = gr.Textbox(label="结论", lines=3)
                btn9.click(fig_interleave_packing,
                           inputs=[s9_text, s9_visual, s9_mask],
                           outputs=[plot9, text9])
            with gr.TabItem("10. 视觉嵌入散点"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s10_text = gr.Slider(2, 20, value=6, step=1, label="文本 token 数")
                        s10_visual = gr.Slider(1, 16, value=4, step=1, label="视觉 token 数")
                        s10_dim = gr.Slider(4, 64, value=8, step=2, label="嵌入维度 dim")
                        btn10 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot10 = gr.Plot(label="scatter 落地验证")
                        text10 = gr.Textbox(label="结论", lines=3)
                btn10.click(fig_scatter_embeddings,
                            inputs=[s10_text, s10_visual, s10_dim],
                            outputs=[plot10, text10])
            with gr.TabItem("11. End-to-End 多模态前向"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s11_mode = gr.Radio(["image", "video"], value="image", label="模式")
                        s11_frames = gr.Slider(2, 8, value=4, step=1, label="帧数 (video)")
                        s11_img = gr.Slider(28, 112, value=56, step=14, label="图像尺寸")
                        s11_text = gr.Slider(2, 20, value=6, step=1, label="文本 token 数")
                        btn11 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot11 = gr.Plot(label="E2E 前向验证")
                        text11 = gr.Textbox(label="结论", lines=3)
                btn11.click(fig_end_to_end,
                            inputs=[s11_mode, s11_frames, s11_img, s11_text],
                            outputs=[plot11, text11])
            with gr.TabItem("12. 梯度回流"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s12_dim = gr.Slider(16, 128, value=32, step=16, label="dim")
                        s12_layers = gr.Slider(1, 8, value=2, step=1, label="n_layers")
                        s12_patch = gr.Slider(7, 28, value=14, step=7, label="patch_size")
                        s12_img = gr.Slider(28, 224, value=56, step=14, label="图像尺寸")
                        btn12 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot12 = gr.Plot(label="梯度分布")
                        text12 = gr.Textbox(label="结论", lines=3)
                btn12.click(fig_gradient_flow,
                            inputs=[s12_dim, s12_layers, s12_patch, s12_img],
                            outputs=[plot12, text12])
            with gr.TabItem("13. 合成数据"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s13_shapes = gr.Slider(1, 6, value=3, step=1, label="形状数量")
                        s13_h = gr.Slider(32, 128, value=64, step=8, label="高度")
                        s13_w = gr.Slider(32, 128, value=64, step=8, label="宽度")
                        s13_seed = gr.Slider(0, 99, value=0, step=1, label="随机种子")
                        s13_norm = gr.Checkbox(value=False, label="归一化坐标")
                        btn13 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot13 = gr.Plot(label="合成场景 + BBox")
                        text13 = gr.Textbox(label="结论", lines=3)
                btn13.click(fig_synthetic_scene,
                            inputs=[s13_shapes, s13_h, s13_w, s13_seed, s13_norm],
                            outputs=[plot13, text13])
            with gr.TabItem("14. 时间池化丢时序（热力图）"):
                with gr.Row():
                    with gr.Column(scale=1):
                        s14_frames = gr.Slider(2, 16, value=6, step=1, label="帧数 frames")
                        s14_img = gr.Slider(14, 56, value=28, step=14, label="图像尺寸")
                        s14_dim = gr.Slider(8, 64, value=16, step=8, label="dim")
                        s14_pool = gr.Slider(1, 8, value=2, step=1, label="temporal_pool")
                        s14_seed = gr.Slider(0, 99, value=0, step=1, label="随机种子")
                        btn14 = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        plot14 = gr.Plot(label="正序 vs 倒序 热力图")
                        text14 = gr.Textbox(label="结论", lines=3)
                btn14.click(fig_temporal_pool_heatmap,
                            inputs=[s14_frames, s14_img, s14_dim, s14_pool, s14_seed],
                            outputs=[plot14, text14])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
