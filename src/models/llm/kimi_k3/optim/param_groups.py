"""参数分组逻辑：将模型参数拆分为 Muon 组和 AdamW 组。

与 kimi3-master muon.py:222-261 的 muon_param_groups 一致。
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn

# Per-Head 正交化的注意力投影层后缀
_PER_HEAD_SUFFIXES = ("q_proj", "k_proj", "v_proj", "q_up", "k_up", "v_up")


def _head_count(model: nn.Module, full_name: str) -> int | None:
    """获取注意力投影层的头数，用于 Per-Head 正交化。

    与 kimi3-master muon.py:202-219 的 _head_count 一致。
    """
    if not full_name.endswith(".weight"):
        return None

    base = full_name[: -len(".weight")]
    if "." not in base:
        # 顶层参数（如 head.weight），不可能是注意力投影
        return None
    module_path, attr = base.rsplit(".", 1)

    if attr not in _PER_HEAD_SUFFIXES:
        return None

    owner = model
    for part in module_path.split("."):
        if not hasattr(owner, part):
            return None
        owner = getattr(owner, part)

    # 检查模块是否有 num_heads 属性（KDA/MLA 模块）
    if hasattr(owner, "num_heads"):
        return owner.num_heads
    return None


def build_muon_param_groups(
    model: nn.Module,
    *,
    per_head: bool = True,
) -> list[dict[str, Any]]:
    """将模型参数拆分为 Muon 和 AdamW 组。

    与 kimi3-master muon.py:222-261 的 muon_param_groups 一致：
    - 2D 矩阵 → Muon（use_muon=True）
    - 注意力投影 → Per-Head Muon（num_heads=N）
    - 其余（embedding/bias/norm/router）→ AdamW（use_muon=False）

    Args:
        model: PyTorch 模型
        per_head: 是否启用 Per-Head 正交化

    Returns:
        参数组列表，可直接传给 Muon 优化器
    """
    per_head_groups: dict[int, list] = {}  # {num_heads: [params]}
    matrices: list = []                     # 普通2D矩阵
    others: list = []                       # AdamW 参数

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_matrix = p.ndim == 2
        excluded = (
            "embed_tokens" in name
            or "embed" in name
            or "lm_head" in name
            or "router" in name
            or name.endswith("bias")
            or (".weight" in name and p.ndim == 1)
        )

        if not is_matrix or excluded:
            others.append(p)
            continue

        heads = _head_count(model, name) if per_head else None
        if heads and p.shape[0] % heads == 0:
            per_head_groups.setdefault(heads, []).append(p)
        else:
            matrices.append(p)

    groups: list[dict[str, Any]] = []
    if matrices:
        groups.append({"params": matrices, "use_muon": True, "num_heads": None})
    for heads, params in sorted(per_head_groups.items()):
        groups.append({"params": params, "use_muon": True, "num_heads": heads})
    if others:
        groups.append({"params": others, "use_muon": False, "num_heads": None})

    return groups


def build_optimizer(model: nn.Module, optim_cfg) -> "Muon":
    """从模型和配置构建 Muon 优化器（对齐 kimi3-master optim/muon.py:264-276）。

    封装 build_muon_param_groups + Muon 构造，一步完成优化器构建。

    Args:
        model: Kimi3ForCausalLM 模型
        optim_cfg: OptimConfig 配置对象

    Returns:
        Muon 优化器实例
    """
    from models.llm.kimi_k3.optim.muon import Muon

    return Muon(
        build_muon_param_groups(model, per_head=optim_cfg.per_head),
        lr=optim_cfg.lr,
        adamw_lr=optim_cfg.adamw_lr,
        momentum=optim_cfg.muon_momentum,
        nesterov=optim_cfg.muon_nesterov,
        ns_steps=optim_cfg.newton_schulz_steps,
        weight_decay=optim_cfg.weight_decay,
        betas=optim_cfg.betas,
        eps=optim_cfg.eps,
        weight_clip=optim_cfg.weight_clip,
    )
