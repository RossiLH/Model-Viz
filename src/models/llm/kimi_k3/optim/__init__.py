"""Muon 优化器模块。

公开 API:
    - newton_schulz: Newton-Schulz 正交化
    - match_update_rms: Update RMS 缩放
    - Muon: Muon 优化器
    - per_head_orthogonalize: Per-Head 正交化
    - build_muon_param_groups: 参数分组
    - HybridMuonAdamW: 混合优化器
    - cosine_with_warmup: Cosine 调度函数
    - CosineWarmupLR: Cosine 调度器
    - OwnershipMap: P2P 所有权映射
    - P2PMuonOrthogonalizer: P2P Muon 正交化器
    - orthogonalize_owned: 一次性 P2P 正交化
"""
from models.llm.kimi_k3.optim.newton_schulz import match_update_rms, newton_schulz
from models.llm.kimi_k3.optim.muon import Muon, per_head_orthogonalize
from models.llm.kimi_k3.optim.param_groups import build_muon_param_groups
from models.llm.kimi_k3.optim.hybrid import HybridMuonAdamW
from models.llm.kimi_k3.optim.lr_scheduler import CosineWarmupLR, cosine_with_warmup
from models.llm.kimi_k3.optim.muon_p2p import OwnershipMap, P2PMuonOrthogonalizer, orthogonalize_owned

__all__ = [
    "newton_schulz",
    "match_update_rms",
    "Muon",
    "per_head_orthogonalize",
    "build_muon_param_groups",
    "HybridMuonAdamW",
    "CosineWarmupLR",
    "cosine_with_warmup",
    "OwnershipMap",
    "P2PMuonOrthogonalizer",
    "orthogonalize_owned",
]
