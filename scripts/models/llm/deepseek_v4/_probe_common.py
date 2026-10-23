"""探针脚本公共工具：把两个参考仓库挂到 sys.path，并提供统一的打印助手。

为什么需要这个文件
------------------
两个参考仓库都不是 pip 安装进环境的，各自的导入根还不一样：
  - deepseek-v4-mini-pytorch：包根是仓库根目录，模块以 `src.xxx` 形式导入
    （其 pyproject 声明 pythonpath = [".", "src"]）
  - nano-deepseek-v4：包根是仓库根目录，模块以 `nano_deepseek_v4.xxx` 形式导入
把这段 sys.path 处理集中在一处，避免每个探针脚本重复且写错。

同时提供若干打印助手，使所有探针脚本的输出格式统一，
便于把输出整段回填进分析文档的「实测证据」小节。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# 路径常量：相对本文件定位，因此脚本可从任意工作目录启动
# ---------------------------------------------------------------------------
# 本文件位于 Model-Viz/scripts/deepseek_v4/，parents[3] 即 Model-Viz-Impl/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REF_ROOT = PROJECT_ROOT / "Reference" / "DeepSeekV4"
REF_MINI = REF_ROOT / "deepseek-v4-mini-pytorch-main"
REF_NANO = REF_ROOT / "nano-deepseek-v4-main"
# SGLang 仅做静态阅读（需 GPU + 自定义内核），此处仅记录路径供文档引用
REF_SGLANG = PROJECT_ROOT / "Reference" / "sglang-main-2026-08-03-KIMI-K3"


def add_mini_to_path() -> Path:
    """把主参考仓库（deepseek-v4-mini-pytorch）加入 sys.path 并返回其根目录。"""
    if not REF_MINI.is_dir():
        raise FileNotFoundError(f"主参考仓库不存在: {REF_MINI}")
    p = str(REF_MINI)
    if p not in sys.path:
        sys.path.insert(0, p)
    return REF_MINI


def add_nano_to_path() -> Path:
    """把次参考仓库（nano-deepseek-v4）加入 sys.path 并返回其根目录。"""
    if not REF_NANO.is_dir():
        raise FileNotFoundError(f"次参考仓库不存在: {REF_NANO}")
    p = str(REF_NANO)
    if p not in sys.path:
        sys.path.insert(0, p)
    return REF_NANO


# ---------------------------------------------------------------------------
# 打印助手：统一所有探针脚本的输出风格
# ---------------------------------------------------------------------------
_WIDTH = 74


def banner(title: str) -> None:
    """打印一级标题（脚本开头用）。"""
    print("=" * _WIDTH)
    print(title)
    print("=" * _WIDTH)


def section(title: str) -> None:
    """打印二级标题（每个检查项用）。"""
    print()
    print("-- " + title + " " + "-" * max(0, _WIDTH - len(title) - 4))


def kv(key: str, value: Any, width: int = 34) -> None:
    """打印「键: 值」对，键左对齐到固定宽度，便于肉眼纵向对照。"""
    print(f"  {key:<{width}} {value}")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """打印一张等宽对齐的表格（用于三方对照 / 逐层调度等）。

    Args:
        headers: 表头
        rows:    每行的单元格序列，元素会被 str() 化
    """
    rows = [[str(c) for c in r] for r in rows]
    headers = [str(h) for h in headers]
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(c))
    line = "  " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  " + "-+-".join("-" * w for w in widths))
    for r in rows:
        print("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))


def shape_of(t: Any) -> str:
    """安全地取张量 shape 的字符串（None 或非张量也不会抛异常）。"""
    if t is None:
        return "None"
    shape = getattr(t, "shape", None)
    return str(tuple(shape)) if shape is not None else type(t).__name__


def count_params(module: Any) -> int:
    """统计模块可训练参数量。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
