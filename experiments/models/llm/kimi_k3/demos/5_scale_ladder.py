#!/usr/bin/env python3
"""从你刚运行的模型到论文中的模型——相同代码，四种配置。

    python experiments/demos/5_scale_ladder.py

kimi3/ 中没有任何代码在规模上分支。宽度、深度、专家数量和并行度都是配置，
所以其他 demo 中的 19.8M 模型和论文的 2.8T 模型是同一源文件的不同 YAML。

这不分配任何资源。每个数字都是从配置解析计算的，这是询问
"它会花多少钱"的唯一方式——关于一个放不下的模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import load_config_from_yaml, ROOT

from models.llm.kimi_k3.config import count_params, flops_per_token

LADDER = [
    ("k3_nano", "ran in demos 1-4, on this CPU"),
    ("k3_micro", "multi-process parity checks"),
    ("k3_small", "first real multi-node run"),
    ("k3_2p8t", "the paper's configuration"),
]

#: 每参数 20 个 token，Chinchilla 风格的配比。
TOKENS_PER_PARAM = 20
#: H100 at bf16, 宽松的 40% MFU。
GPU_FLOPS, MFU = 989e12, 0.40


def fmt(n: float, unit: str = "") -> str:
    """格式化大数字。"""
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= scale:
            return f"{n/scale:.1f}{suffix}{unit}"
    return f"{n:.0f}{unit}"


def main() -> int:
    rows = []
    for name, note in LADDER:
        path = ROOT / "configs" / f"{name}.yaml"
        if not path.exists():
            continue
        cfg = load_config_from_yaml(name)
        pc = count_params(cfg)
        fpt = flops_per_token(cfg)["total"]
        tokens = pc.activated * TOKENS_PER_PARAM
        gpu_hours = (3 * fpt * tokens) / (GPU_FLOPS * MFU) / 3600
        rows.append((name, pc, fpt, tokens, gpu_hours, note))

    print(f"{'config':<11} {'total':>9} {'active':>9} {'layers':>7} {'experts':>8} "
          f"{'FLOPs/tok':>10} {'20x tokens':>11} {'GPU-hours':>11}")
    print("─" * 92)
    for name, pc, fpt, tokens, hours, _ in rows:
        cfg = load_config_from_yaml(name)
        print(f"{name:<11} {fmt(pc.total):>9} {fmt(pc.activated):>9} "
              f"{cfg.model.n_layers:>7} {cfg.model.moe.n_routed:>8} "
              f"{fmt(fpt):>10} {fmt(tokens):>11} {fmt(hours):>11}")
    print()
    for name, _, _, _, _, note in rows:
        print(f"  {name:<11} {note}")

    print("\nGPU-hours assume H100 bf16 at 40 % MFU and 3x forward FLOPs for the")
    print("backward pass. They are the honest order of magnitude, not a quote --")
    print("real runs lose time to data loading, checkpointing and failures.")
    print("\nCheck any config yourself, still without allocating anything:")
    print("  python3 scripts/count_params.py configs/k3_2p8t.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
