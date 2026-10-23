"""生成质量评估：将"可识别的 Python 和英语"转化为数字。

设计动机（why）
-------------
P5 门控要求两件事：更低的 held-out loss，以及生成的样本是"可识别的 Python
和英语，而非 Markov 汤"。前者是测量，后者是意见——除非你把它变成测量。

两个独立信号，因为它们的失败模式不同：
- python_signature_valid：检查第一行是否能被 CPython 解析器接受
- english_word_rate：检查字母序列中有多少是真正的英语单词

与 kimi3-master 的关系：
  本文件忠实复刻 sample_quality.py 的核心逻辑，添加详尽中文注释。
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

__all__ = [
    "python_signature_valid",
    "english_word_rate",
    "load_word_list",
]

#: 匹配两个或更多字母组成的序列。单字母跳过，因为 "a"/"i" 是真单词，
#: 但每个散乱字符在损坏的 token 中也是，会膨胀恰恰最不该膨胀的模型的比率。
_WORD = re.compile(r"[A-Za-z]{2,}")

#: 默认词表路径（Linux/macOS 标准）
DEFAULT_WORD_LIST = Path("/usr/share/dict/words")


@lru_cache(maxsize=4)
def load_word_list(path=None) -> frozenset:
    """加载换行分隔的词表，全部小写。

    如果词表不存在，返回空集合——调用者降级为"无法评分英语"而非崩溃。

    Args:
        path: 词表文件路径（默认 /usr/share/dict/words）

    Returns:
        frozenset of lowercase words
    """
    p = Path(path) if path else DEFAULT_WORD_LIST
    if not p.exists():
        return frozenset()
    words = (line.strip().lower() for line in p.read_text(errors="ignore").splitlines())
    return frozenset(w for w in words if w)


def python_signature_valid(text: str) -> bool:
    """第一行文本是否能在某个合法上下文中被解析为 Python？

    关键细节：
    - 先 strip 再检查，因为 ast.parse 对前导空白会报 IndentationError
    - 如果行尾是 ":"，追加 "\n    pass" 使其成为合法的复合语句
    - 不检查作用域（"return x" 在函数外是 compile 错误而非 parse 错误）

    Args:
        text: 生成的文本

    Returns:
        第一行是否可解析为 Python
    """
    stripped = text.split("\n", 1)[0].rstrip().lstrip()
    if not stripped:
        return False
    candidate = f"{stripped}\n    pass" if stripped.endswith(":") else stripped
    try:
        ast.parse(candidate)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def english_word_rate(text: str, vocab=None) -> tuple:
    """文本中字母序列有多少是字典单词。

    下划线先分割标识符，所以 get_setting_ids 贡献三个真单词而非一个非单词。

    Args:
        text: 待评分文本
        vocab: 词表（可选，默认加载系统词表）

    Returns:
        (rate, n_words) 元组
    """
    vocab = load_word_list() if vocab is None else vocab
    if not vocab:
        return (float("nan"), 0)
    words = _WORD.findall(text)
    if not words:
        return (0.0, 0)
    hits = sum(1 for w in words if w.lower() in vocab)
    return (hits / len(words), len(words))
