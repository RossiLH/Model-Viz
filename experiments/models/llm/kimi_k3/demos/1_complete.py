#!/usr/bin/env python3
"""让模型写代码。

    python experiments/demos/1_complete.py
    python experiments/demos/1_complete.py --prompt "class Config"
    python experiments/demos/1_complete.py --prompt "def " --temperature 1.2 --n 200

这是一个 19.8M 参数的模型（每 token 激活 8.7M），在 3000 步 Python 数据上训练。
它续写文本。它不遵循指令、回答问题或进行对话——没有指令调优，这个规模也没有什么可调的。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import complete, load

DEFAULT_PROMPTS = [
    "def ",
    "class ",
    "import ",
    "    return ",
    "    if ",
    "        self.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="续写这个提示词（默认使用内置列表）")
    ap.add_argument("--checkpoint", default="3000")
    ap.add_argument("--n", type=int, default=90, help="生成的 token 数")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model, tok = load(args.checkpoint)
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    print(f"k3 nano @ {args.checkpoint} steps | temperature {args.temperature}, "
          f"top_p {args.top_p}, seed {args.seed}\n")

    for prompt in prompts:
        text = complete(model, tok, prompt, n=args.n, temperature=args.temperature,
                        top_p=args.top_p, seed=args.seed)
        print(f"\033[2m{'─' * 74}\033[0m")
        print(text.rstrip() or "(nothing)")
    print(f"\033[2m{'─' * 74}\033[0m")
    print("\n提高 --temperature 使输出更随机；降低它使输出更重复。")
    print("试试  --temperature 0.3  和  --temperature 1.3  对比同一提示词。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
