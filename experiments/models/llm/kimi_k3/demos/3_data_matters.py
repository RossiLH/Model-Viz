#!/usr/bin/env python3
"""为什么语料库是全部赌注。

    python experiments/demos/3_data_matters.py

两个模型。相同架构、相同优化器、相同种子、相同步数。唯一区别是读了什么。

- **filler** 在程序生成的文本上训练——语法正确但无意义的英语句子。
- **real** 在精选的 Python 和文档语料库上训练。

filler 模型几乎完美学到了自己的分布：训练 loss 0.4475，比 real 模型的 2.5925 好得多。
但它什么都没迁移。在 held-out 真实文本上评分 6.2787，比起点还差，
而 ln(512) = 6.238 是 512 符号词汇表的随机基线。

**下降的 loss 曲线证明优化器在工作，但对模型质量什么也说明不了。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import complete, head, load

from models.llm.kimi_k3.eval.sample_quality import english_word_rate, python_signature_valid

ARMS = [
    ("filler-300", "synthetic filler", 0.4475, 6.2787),
    ("real-300", "curated real text", 2.5925, 1.9164),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"both arms: 300 steps, same seed, same architecture. prompt {args.prompt!r}\n")
    print(f"{'arm':<20} {'train loss':>11} {'held-out':>10}")
    for _, label, train, held in ARMS:
        print(f"{label:<20} {train:>11.4f} {held:>10.4f}")
    print(f"{'chance':<20} {'':>11} {6.238:>10.4f}   ln(512)\n")

    for name, label, _, _ in ARMS:
        try:
            model, tok = load(name)
        except SystemExit as exc:
            print(f"{label}: {exc}")
            continue
        text = complete(model, tok, args.prompt, n=100, seed=args.seed)
        rate, n_words = english_word_rate(text)
        print(f"\033[1m{label}\033[0m")
        for line in head(text, 3).split("\n"):
            print(f"    {line}")
        print(f"    \033[2mparses as Python: {python_signature_valid(text)}"
              f"   dictionary-word rate: {rate:.3f} over {n_words} words\033[0m\n")

    print("注意这个词率，并且不要信任它。在 480 次生成中，filler")
    print("模型得分 0.9556——上面的单个样本太短了，无法读出可靠的词率，")
    print("这本身就是教训——")
    print("比真实 Python 源码高，后者充满了标识符和数字。")
    print("它发出的每个 token 都是真正的英语单词；没有一个是代码。")
    print("一个退化模型得分最高的指标，并没有在衡量质量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
