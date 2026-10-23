# -*- coding: utf-8 -*-
"""Muon 优化器端到端训练 Demo（向后兼容入口）。

对比 Muon vs AdamW 在简单 Transformer 上的训练效果。
核心逻辑已迁移到 plot_muon_vs_adamw.py，本文件保留为命令行入口。

运行方式:
    cd Model-Viz
    python experiments/models/llm/kimi_k3/optim/demo_muon_e2e.py
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_muon_vs_adamw import (
    fig_muon_vs_adamw, run_training, SimpleTransformer,
)
from plot_utils import save


def main():
    """向后兼容的主入口。"""
    print("=" * 60)
    print("  Muon vs AdamW 端到端训练对比 Demo")
    print("=" * 60)

    fig, txt = fig_muon_vs_adamw(num_steps=200, seed=42, d_model=64)
    print()
    print("=" * 60)
    print("  训练结果对比")
    print("=" * 60)
    for line in txt.split("\n"):
        print(f"  {line}")

    out_path = save(fig, "muon_vs_adamw.png")
    print(f"\n  Chart saved: {out_path}")
    print("\n完成!")


if __name__ == "__main__":
    main()
