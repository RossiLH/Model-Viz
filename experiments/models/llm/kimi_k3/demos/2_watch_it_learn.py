#!/usr/bin/env python3
"""同一提示词，训练三个阶段的对比。

    python experiments/demos/2_watch_it_learn.py

除了训练时长，其他完全不变：相同架构、相同数据、相同提示词、相同种子。
800 步时输出 Python 形状的噪声；到 3000 步，输出 CPython 解析器能接受的签名。

Held-out loss 变化：1.2677 -> 1.0541 -> 0.8434。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import complete, head, load

from models.llm.kimi_k3.eval.sample_quality import python_signature_valid

STAGES = [("800", 1.2677), ("1400", 1.0541), ("3000", 0.8434)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"prompt: {args.prompt!r}\n")
    for steps, loss in STAGES:
        try:
            model, tok = load(steps)
        except SystemExit as exc:
            print(f"{steps:>5} steps  — {exc}")
            continue
        text = complete(model, tok, args.prompt, n=100, seed=args.seed)
        first = text.split("\n")[0]
        mark = "\033[32mparses\033[0m" if python_signature_valid(text) else "\033[31mfails \033[0m"
        print(f"\033[1m{steps:>5} steps\033[0m  held-out {loss:.4f}   [{mark}]")
        for line in head(text, 3).split("\n"):
            print(f"              {line}")
        print()

    print("在 480 次生成上正式测量（不只是上面六行）：")
    print("  parser validity 0.1389 (300 steps) -> 0.7021 (3,000 steps)")
    print("  语料库本身得分 0.8182，所以这达到了自身上限的 86%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
