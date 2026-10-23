"""文档引用校验器：核对 DeepSeekV4 文档里的 file:line 引用是否真实存在。

为什么需要这个文件
------------------
分析文档（D1-D4）里写了大量 `file:line` 引用（指向三个参考仓库的真实代码行）。
随着参考仓库更新或手写行号漂移，这些引用可能失效，误导读者跳转到错误位置。
本脚本扫描 `Model-Viz/docs/DeepSeekV4/` 下所有文档，解析 `文件名:行号` 模式，
逐一验证：
  1. 该文件是否存在；
  2. 该行号是否在该文件总行数范围内（引用真实存在的行）。

用法
----
    python scripts\\deepseek_v4\\99_check_doc_refs.py

退出码
------
0 = 所有引用有效；1 = 存在失效引用（会列出每一条：文档 / 引用 / 问题）

说明
----
- 引用模式：`xxx.py:NNN` 或 `xxx.py:NNN` 形式（文件:数字）。
- 文件路径可能带相对前缀（如 `modeling.py:839`），本脚本尝试：
    1. 直接当作相对路径（相对当前工作目录）；
    2. 在三个参考仓库根目录下递归查找该文件名。
- 对无法解析路径但文件名存在于参考仓库的，按仓库内行号校验。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
DOC_DIR = Path(__file__).resolve().parents[2] / "docs" / "DeepSeekV4"
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # Model-Viz 项目根（含 src/）
REF_ROOT = Path(__file__).resolve().parents[3] / "Reference"  # 参考仓库根（Model-Viz-Impl 下）
# 三个参考仓库根目录（用于递归查找引用文件）
REF_ROOTS = [
    REF_ROOT / "DeepSeekV4" / "deepseek-v4-mini-pytorch-main",
    REF_ROOT / "DeepSeekV4" / "nano-deepseek-v4-main",
    REF_ROOT / "sglang-main-2026-08-03-KIMI-K3",
]

# 文档中 file:line 引用模式：<可选路径>/<文件名>.py:数字
_REF_RE = re.compile(r"([\w./\\-]+\.py):(\d+)")


def _resolve_file(name: str) -> Path | None:
    """解析 `xxx.py:NNN` 引用的实际文件。

    查找规则：
      - 带路径前缀（如 `modules/mqa.py`、`deepseek_v4/config.py`）：在本仓库
        `src/models/llm/deepseek_v4/` 下按完整相对路径匹配（D5 最终实现文档
        引用本仓库实现代码）。
      - 裸文件名（如 `modeling.py`、`deepseek_components.py`）：只在三个参考
        仓库根目录下递归查找（D1-D3 导读文档引用参考仓库代码，避免与 src 下
        同名文件冲突）。
    """
    name_posix = name.replace("\\", "/")
    if "/" in name_posix:
        # 带路径：先试本仓库 src 下的完整相对路径
        src_root = PROJECT_ROOT / "src"
        pkg_root = src_root / "models" / "llm" / "deepseek_v4"
        # `deepseek_v4/config.py` -> pkg_root/config.py（剥掉包名前缀）
        # `modules/mqa.py`        -> pkg_root/modules/mqa.py
        candidates = [src_root / name_posix, pkg_root / name_posix]
        if name_posix.startswith("deepseek_v4/"):
            candidates.append(pkg_root / name_posix[len("deepseek_v4/"):])
        # 兼容 `models/llm/deepseek_v4/...` 全路径
        candidates.append(src_root / name_posix.removeprefix("models/llm/"))
        for p in candidates:
            if p.is_file():
                return p
        # 参考仓库里也可能有子路径引用，回退递归
        for root in REF_ROOTS:
            for p in root.rglob(name_posix):
                if p.is_file():
                    return p
        return None
    # 裸文件名：只在参考仓库查找
    for root in REF_ROOTS:
        for p in root.rglob(name_posix):
            if p.is_file():
                return p
    return None


def _line_count(path: Path) -> int:
    """统计文件总行数（避免读入内存）。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f)


def check_doc(doc: Path) -> tuple[int, list[str]]:
    """扫描单个文档，返回 (引用总数, 问题列表)。"""
    issues: list[str] = []
    total = 0
    with open(doc, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, 1):
            for m in _REF_RE.finditer(line):
                total += 1
                raw_name, num_str = m.group(1), m.group(2)
                num = int(num_str)
                resolved = _resolve_file(raw_name)
                if resolved is None:
                    issues.append(f"{doc.name}:{ln} 引用 '{raw_name}' 文件未找到")
                    continue
                if num > _line_count(resolved):
                    issues.append(
                        f"{doc.name}:{ln} 引用 '{raw_name}:{num}' 超出文件"
                        f"总行数 {_line_count(resolved)}"
                    )
    return total, issues


def main() -> int:
    if not DOC_DIR.is_dir():
        print(f"[FAIL] 文档目录不存在: {DOC_DIR}")
        return 1

    print("=" * 66)
    print(f"DeepSeekV4 文档引用校验器（扫描 {DOC_DIR}）")
    print("=" * 66)

    all_issues: list[str] = []
    grand_total = 0
    for doc in sorted(DOC_DIR.glob("*.md")):
        total, issues = check_doc(doc)
        grand_total += total
        status = "OK " if not issues else "FAIL"
        print(f"  [{status}] {doc.name}: {total} 处引用，{len(issues)} 处问题")
        all_issues.extend(issues)

    print("\n" + "=" * 66)
    if all_issues:
        print(f"共 {grand_total} 处引用，{len(all_issues)} 处失效：")
        for i, msg in enumerate(all_issues, 1):
            print(f"  {i}. {msg}")
        return 1
    print(f"全部 {grand_total} 处引用有效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
