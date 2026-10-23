"""PI0.5 探针脚本公共工具：sys.path 挂载 + 统一输出格式。

为什么需要这个文件
------------------
探针脚本的目的是「把实测数字固化下来，供分析文档的『实测证据』小节直接引用」。
既然要被文档引用，各脚本的输出格式就必须统一（否则回填时格式五花八门），
而且必须能从任意工作目录启动（否则 CI/手动跑的结果会不一致）。
因此这里集中做两件事：

1. 把 `Model-Viz/src` 挂到 sys.path。项目没有 pip install -e，
   `conftest.py` 只在 pytest 下生效，脚本单独跑时必须自己挂。
2. 提供 banner/section/kv/table 等打印助手，风格对齐
   `scripts/deepseek_v4/_probe_common.py`，保持全项目探针输出一致。

环境约束（写代码时必须记住）
--------------------------
实际解释器是 Python 3.7.9 + torch 1.11.0+cpu，没有 jax / transformers。
所以：不用 3.8+ 语法（海象运算符、位置参数标记 /、typing.Literal 等），
不用 torch 2.x 才有的 API（nn.RMSNorm、torch.compile 等）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量：相对本文件定位，因此脚本可从任意工作目录启动
# ---------------------------------------------------------------------------
# 本文件位于 Model-Viz/scripts/pi05/，parents[2] 即 Model-Viz/
MODEL_VIZ_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MODEL_VIZ_ROOT.parent
SRC_ROOT = MODEL_VIZ_ROOT / "src"
# 官方 openpi 仅做静态阅读（需 jax，本环境跑不了），此处仅记录路径供文档引用
REF_OPENPI = PROJECT_ROOT / "Reference" / "PI" / "openpi-main"


def add_src_to_path():
    """把 Model-Viz/src 加入 sys.path，使 `import models.vla...` 可用。"""
    p = str(SRC_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return SRC_ROOT


# ---------------------------------------------------------------------------
# 打印助手：统一所有探针脚本的输出风格
# ---------------------------------------------------------------------------
_WIDTH = 74


def banner(title):
    """打印一级标题（脚本开头用）。"""
    print("=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def section(title):
    """打印二级标题（每个检查项用）。"""
    print()
    print("-- " + title + " " + "-" * max(0, _WIDTH - len(title) - 4))


def kv(key, value, width=40):
    """打印「键: 值」对，键左对齐到固定宽度，便于肉眼纵向对照。"""
    print("  {0:<{1}} {2}".format(key, width, value))


def note(text):
    """打印一行解释性说明（用于把「这个数字为什么重要」写进输出）。"""
    print("     -> " + text)


def table(headers, rows):
    """打印一张等宽对齐的表格（ASCII，可直接粘进 Markdown 代码块）。

    Args:
        headers: 表头序列
        rows:    每行的单元格序列，元素会被 str() 化
    """
    rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    print("  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def shape_of(t):
    """安全地取张量 shape 的字符串（None 或非张量也不会抛异常）。"""
    if t is None:
        return "None"
    shape = getattr(t, "shape", None)
    return str(tuple(shape)) if shape is not None else type(t).__name__


def count_params(module):
    """统计模块可训练参数量。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def fmt_num(n):
    """把大整数格式化成带千分位 + human-readable 后缀，便于文档表格阅读。"""
    if n >= 1e9:
        return "{0:,} ({1:.2f} B)".format(n, n / 1e9)
    if n >= 1e6:
        return "{0:,} ({1:.2f} M)".format(n, n / 1e6)
    if n >= 1e3:
        return "{0:,} ({1:.2f} K)".format(n, n / 1e3)
    return str(n)
