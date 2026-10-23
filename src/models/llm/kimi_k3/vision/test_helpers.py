# -*- coding: utf-8 -*-
"""vision 模块共享工具函数。

从 test_moonvit.py / test_packing.py / test_integration.py 中提取的可复用辅助函数,
供测试脚本和 Gradio demo 共同使用。

所有函数都不做 assert, 只返回数据或做记录 (供调用方决定是否断言 / 怎么用。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..config import (
    AttnResConfig,
    KDAConfig,
    Kimi3Config,
    Kimi3ModelConfig,
    MoEConfig,
    ViTConfig,
)
from .packing import VISION_PLACEHOLDER, scatter_vision_embeddings
from .moonvit import MoonViTV2

__all__ = [
    "ByteTok",
    "VisionAudit",
    "small_vit_cfg",
    "build_vit",
    "small_model_cfg",
    "build_multimodal_model",
    "make_interleaved_tokens",
    "audit_bias_terms",
    "audit_norms",
    "audit_param_shapes",
    "shape_image_forward",
    "shape_video_forward",
    "shape_tower_forward",
    "shape_projector",
    "shape_interleave",
    "compute_scatter_landing",
    "snap_image_size",
    "classify_param_module",
    "collect_gradient_flow",
]


# ---------------------------------------------------------------------------
# 简易 tokenizer
# ---------------------------------------------------------------------------

class ByteTok:
    """最小字节级 tokenizer 替身：按字符编码，仅测试/演示用。"""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


# ---------------------------------------------------------------------------
# 配置 / 构造器
# ---------------------------------------------------------------------------

def small_vit_cfg(**kw) -> ViTConfig:
    """教学尺寸 ViT 配置。与 tests/test_moonvit.py 基准。"""
    base = dict(n_layers=2, dim=32, num_heads=4, patch_size=14, mlp_hidden=64,
                pixel_shuffle=2, temporal_pool=2)
    base.update(kw)
    return ViTConfig(**base)


def build_vit(cfg: ViTConfig | None = None, **kw):
    """快捷构造 MoonViTV2（默认 small_vit_cfg + 可覆盖。"""
    from .moonvit import MoonViTV2  # 延迟导入避免循环
    return MoonViTV2(cfg or small_vit_cfg(**kw))


def small_model_cfg(vit_enabled: bool = True, mtp_enabled: bool = False) -> Kimi3Config:
    """教学尺寸 K3 多模态配置，与 tests/test_integration.py 一致。"""
    base_kda = dict(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                    conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0,
                    alpha_init=0.5, use_triton=False)
    base_moe = dict(n_routed=4, n_shared=1, top_k=2, latent_dim=8, expert_hidden=16,
                    shared_hidden=32, situ_beta_gate=1.0, situ_beta_up=1.0,
                    rmsnorm_before_up=True, qb_enabled=False, qb_bins=10, qb_ema=0.9)
    mc = Kimi3ModelConfig(
        dim=64, n_layers=2, n_heads=4, vocab_size=100, n_dense_layers=1,
        kda=KDAConfig(**base_kda),
        moe=MoEConfig(**base_moe),
        attn_res=AttnResConfig(enabled=False),
        vit=ViTConfig(enabled=vit_enabled, n_layers=2, dim=16, num_heads=2,
                      patch_size=7, mlp_hidden=32, pixel_shuffle=2, temporal_pool=2),
    )
    if mtp_enabled:
        from ..config import MTPConfig
        mc.mtp = MTPConfig(enabled=True, n_layers=1, layer_kind="mla", loss_weight=0.3)
    return Kimi3Config(model=mc)


def build_multimodal_model(cfg: Kimi3Config | None = None):
    """快捷构造 Kimi3ForCausalLMWithVision。"""
    from ..model_with_vision import Kimi3ForCausalLMWithVision
    return Kimi3ForCausalLMWithVision(cfg or small_model_cfg())


# ---------------------------------------------------------------------------
# token 流构造器
# ---------------------------------------------------------------------------

def make_interleaved_tokens(n_text: int, n_visual: int, vocab: int = 100,
                          batch: int = 1) -> Tensor:
    """构造含 n_visual 个 VISION_PLACEHOLDER 的 token 流。

    前 n_text 个为随机文本 token，后 n_visual 个为占位符，
    形状 (batch, n_text + n_visual)。
    """
    seq = torch.full((batch, n_text + n_visual), VISION_PLACEHOLDER, dtype=torch.long)
    seq[:, :n_text] = torch.randint(0, vocab, (batch, n_text))
    return seq


# ---------------------------------------------------------------------------
# 审计类：统一返回结构化结果（供测试断言 + 供 demo 可视化
# ---------------------------------------------------------------------------

@dataclass
class VisionAudit:
    """视觉塔/投影器/多模态模型的结构审计结果。"""
    n_params: int = 0
    n_bias_terms: int = 0
    bias_names: list = None
    n_rmsnorm: int = 0
    n_layernorm: int = 0
    param_shape_diff: dict = None  # {name: (expected_shape, actual_shape)}


def audit_bias_terms(module: torch.nn.Module) -> tuple[int, list[str]]:
    """统计 Linear / Conv2d 中 bias 非 None 的数量和名字。

    返回 ``(count, names)``。测试断言 count==0 即"全塔无 bias"。
    """
    names = []
    for name, m in module.named_modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            if m.bias is not None:
                names.append(name)
    return len(names), names


def audit_norms(module: torch.nn.Module) -> tuple[int, int]:
    """统计 RMSNorm 与 LayerNorm 的数量。"""
    from ..modules.norm import RMSNorm
    n_rms = sum(1 for m in module.modules() if isinstance(m, RMSNorm))
    n_ln = sum(1 for m in module.modules() if isinstance(m, torch.nn.LayerNorm))
    return n_rms, n_ln


def audit_param_shapes(module: torch.nn.Module, expected: dict) -> dict[str, tuple]:
    """对比参数名与形状，返回差异 dict（空表示完全一致）。

    ``expected``: {param_name: tuple_of_shape}。
    返回 ``{name: (expected, actual)}``，仅包含不一致的项。
    """
    actual = {n: tuple(p.shape) for n, p in module.named_parameters()}
    diff = {}
    for name, shape in expected.items():
        if name not in actual:
            diff[name] = (shape, "MISSING")
        elif actual[name] != shape:
            diff[name] = (shape, actual[name])
    for name in actual:
        if name not in expected:
            diff[name] = ("EXTRA", actual[name])
    return diff


# ---------------------------------------------------------------------------
# 形状探针（前向后形状）
# ---------------------------------------------------------------------------

def shape_image_forward(vit, batch: int = 2, h: int = 56, w: int = 56) -> tuple:
    """图像前向：返回输出形状 (b, tokens, dim*shuffle^2)。"""
    out = vit(torch.randn(batch, 3, h, w))
    return tuple(out.shape)


def shape_video_forward(vit, frames: int = 4, h: int = 56, w: int = 56) -> tuple:
    """视频前向：返回输出形状。"""
    out = vit(torch.randn(frames, 3, h, w), frames=frames)
    return tuple(out.shape)


def shape_tower_forward(tower, kind: str = "image", **kw) -> tuple:
    """VisionTower 图像或视频前向，返回输出形状。"""
    if kind == "image":
        b = kw.get("batch", 2)
        h = kw.get("h", 56)
        w = kw.get("w", 56)
        return tuple(tower.encode_image(torch.randn(b, 3, h, w)).shape)
    else:
        m = kw.get("m", 1)
        f = kw.get("frames", 4)
        h = kw.get("h", 56)
        w = kw.get("w", 56)
        return tuple(tower.encode_video(torch.randn(m, f, 3, h, w)).shape)


def shape_projector(proj, vit, batch: int = 2, h: int = 56, w: int = 56) -> tuple:
    """投影器输出形状：(batch, tokens, model_dim)。"""
    feats = vit(torch.randn(batch, 3, h, w))
    return tuple(proj(feats).shape)


def shape_interleave(tok, sample) -> dict:
    """interleave_multimodal 的结果统计。

    返回 dict: seq_len, n_visual, n_loss_masked, n_text。
    """
    n_visual = int((sample.token_ids == VISION_PLACEHOLDER).sum().item())
    n_masked = int((sample.labels == -100).sum().item())
    n_text = sample.token_ids.numel() - n_visual
    return {
        "seq_len": sample.token_ids.numel(),
        "n_visual": n_visual,
        "n_loss_masked": n_masked,
        "n_text": n_text,
        "vision_slots": sample.vision_slots.tolist(),
    }


def compute_scatter_landing(embeds: Tensor, vision: Tensor, slots: Tensor) -> dict:
    """验证 scatter_vision_embeddings 落地正确性。

    返回 dict: slots_match (bool), max_diff_inside (float), max_diff_outside (float)。
    """
    out = scatter_vision_embeddings(embeds, vision, slots)
    flat = out.view(-1, out.shape[-1])
    inside = flat[slots]
    diff_in = (inside - vision.to(flat.dtype)).abs().max().item()

    all_idx = torch.arange(flat.shape[0])
    outside_mask = torch.ones(flat.shape[0], dtype=torch.bool)
    outside_mask[slots] = False
    outside = flat[outside_mask]
    orig_flat = embeds.view(-1, embeds.shape[-1])
    diff_out = (outside - orig_flat[outside_mask]).abs().max().item()
    return {
        "slots_match": diff_in < 1e-6,
        "max_diff_inside": diff_in,
        "max_diff_outside": diff_out,
    }


# ---------------------------------------------------------------------------
# 梯度回流（from-scratch 端到端可训练性验证）
# ---------------------------------------------------------------------------

def snap_image_size(image_size: int, patch_size: int, shuffle: int = 2) -> int:
    """把图像尺寸吸附到最近的合法值。

    MoonViT-V2 对图像尺寸有两重约束：
      ① `image_size % patch_size == 0`  —— patchify 要能整除切分；
      ② `(image_size // patch_size) % shuffle == 0` —— pixel_shuffle 要能整除折叠，
         否则 moonvit.py:53 抛 `ValueError: (h, w) is not divisible by ...`。

    两条合并后，合法尺寸必为 `patch_size * shuffle` 的正整数倍。
    本函数取距离 image_size 最近的那个倍数（至少 1 倍），使前端滑块的
    任意组合都能出图，而不是 47% 的组合直接崩掉。

    参数:
        image_size: 用户请求的图像边长。
        patch_size: patch 边长。
        shuffle:    pixel shuffle 因子（K3 默认 2；为 1 时约束退化为仅需整除 patch）。

    返回:
        吸附后的合法边长（int）。
    """
    step = patch_size * max(1, shuffle)
    # round(x / step) 取最近的倍数；至少保留 1 倍，避免吸附到 0
    n = max(1, int(round(image_size / step)))
    return n * step


def classify_param_module(name: str) -> str:
    """把参数名归入 patch_embed / attn / mlp / norm 四组之一。

    判定顺序很关键：`norm` 必须先于 `attn` 判断。否则 `blocks.N.attn_norm.weight`
    会因为名字里含 "attn" 而被归进 attn 组，与同为 RMSNorm 的 `mlp_norm.weight`
    分裂到两个不同的组，导致柱状图语义错乱。

    参数:
        name: `named_parameters()` 给出的参数全名。

    返回:
        四个组名之一（保证互斥且完全覆盖 MoonViTV2 的所有参数）。
    """
    if "patch_embed" in name:
        return "patch_embed"
    if "norm" in name:                     # 必须在 attn 之前：attn_norm / mlp_norm / 顶层 norm
        return "norm"
    if "attn" in name or "_proj" in name:  # q/k/v/o_proj
        return "attn"
    if "fc1" in name or "fc2" in name:
        return "mlp"
    return "other"


def collect_gradient_flow(dim: int = 32, n_layers: int = 2, patch_size: int = 14,
                          image_size: int = 56, shuffle: int = 2,
                          seed: int = 0) -> dict:
    """跑一次 MoonViT-V2 前向 + 反向，采集各参数/模块/层的梯度幅值。

    验证 from-scratch 训练成立的前提：训练信号能从 loss 一路传回 patch_embed。
    图像尺寸非法时自动吸附（见 `snap_image_size`），不抛异常。

    参数:
        dim / n_layers / patch_size / image_size / shuffle: 模型与输入规模。
        seed: 随机种子，保证可复现。

    返回 dict：
        per_param            {参数名: |grad| 求和}，共 1 + n_layers*8 + 1 项
        grouped              {组名: |grad| 求和}，四组
        per_layer            [{layer, attn, mlp, norm, total}, ...]，长度 = n_layers
        patch_embed_grad     patch_embed.weight 的 |grad| 求和
        has_patch_grad       patch_embed 是否收到非零梯度
        n_zero_grad          梯度恰为 0 的参数个数（正常应为 0）
        n_params             参数张量个数
        total_numel          总参数量
        image_size           实际使用的边长（已吸附）
        requested_image_size 用户请求的边长
        image_size_adjusted  是否发生了吸附
    """
    dim = int(dim); n_layers = int(n_layers)
    patch_size = int(patch_size); requested = int(image_size); shuffle = int(shuffle)

    eff_size = snap_image_size(requested, patch_size, shuffle)

    torch.manual_seed(int(seed))
    cfg = ViTConfig(n_layers=n_layers, dim=dim, num_heads=4,
                    patch_size=patch_size, mlp_hidden=dim * 2,
                    pixel_shuffle=shuffle)
    vit = MoonViTV2(cfg)

    x = torch.randn(1, 3, eff_size, eff_size)
    vit(x).square().mean().backward()

    per_param: dict[str, float] = {}
    for name, param in vit.named_parameters():
        per_param[name] = 0.0 if param.grad is None else float(param.grad.abs().sum())

    grouped: dict[str, float] = {}
    for name, val in per_param.items():
        grouped[classify_param_module(name)] = grouped.get(classify_param_module(name), 0.0) + val

    # 按 block 展开，用于观察深度方向的梯度衰减
    per_layer: list[dict] = []
    for i in range(n_layers):
        prefix = f"blocks.{i}."
        row = {"layer": i, "attn": 0.0, "mlp": 0.0, "norm": 0.0}
        for name, val in per_param.items():
            if not name.startswith(prefix):
                continue
            key = classify_param_module(name)
            if key in row:
                row[key] += val
        row["total"] = row["attn"] + row["mlp"] + row["norm"]
        per_layer.append(row)

    patch_grad = per_param.get("patch_embed.weight", 0.0)
    return {
        "per_param": per_param,
        "grouped": grouped,
        "per_layer": per_layer,
        "patch_embed_grad": patch_grad,
        "has_patch_grad": patch_grad > 0,
        "n_zero_grad": sum(1 for v in per_param.values() if v == 0.0),
        "n_params": len(per_param),
        "total_numel": sum(p.numel() for p in vit.parameters()),
        "image_size": eff_size,
        "requested_image_size": requested,
        "image_size_adjusted": eff_size != requested,
    }
