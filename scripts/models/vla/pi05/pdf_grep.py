# -*- coding: utf-8 -*-
"""PI 系列论文关键词检索脚本。

用途：解决「RTC（实时约束 / real-time chunking）与 action chunk 部分执行
（receding horizon）这两个机制，究竟出自 pi0.5 还是 pi0.7」这一归属问题。

做法：把 Reference/Paper/PI 下的 PDF 逐页抽成纯文本，按关键词做上下文检索，
打印「论文 → 页码 → 命中句子」三元组，从而用一手证据判定机制归属。

用法：
    python scripts/pi05/pdf_grep.py <关键词1> <关键词2> ...
    python scripts/pi05/pdf_grep.py --dump <论文名片段>   # 导出整篇纯文本
"""
from __future__ import annotations

import pathlib
import re
import sys

from pypdf import PdfReader

# 论文目录（相对仓库根 Model-Viz-Impl）
PAPER_DIR = pathlib.Path(__file__).resolve().parents[3] / "Reference" / "Paper" / "PI"


def load_pages(pdf_path: pathlib.Path) -> list[str]:
    """抽取 PDF 每页文本，返回 list[str]（索引 0 = 第 1 页）。"""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # 个别页可能解析失败，跳过但保留占位
            pages.append("")
            print(f"  [warn] 解析失败: {exc}", file=sys.stderr)
    return pages


def normalize(text: str) -> str:
    """把换行/连字符断词压平，便于跨行匹配关键词。"""
    text = text.replace("-\n", "")       # 行末连字符断词
    text = re.sub(r"\s+", " ", text)      # 多空白压成单空格
    return text


def grep(keywords: list[str], name_filter: str | None = None) -> None:
    """在所有论文中检索关键词，打印命中上下文。"""
    pdfs = sorted(PAPER_DIR.glob("*.pdf"))
    for pdf in pdfs:
        if name_filter and name_filter.lower() not in pdf.name.lower():
            continue
        print("=" * 78)
        print(f"[PAPER] {pdf.name}")
        print("=" * 78)
        pages = load_pages(pdf)
        for kw in keywords:
            hits = 0
            pat = re.compile(re.escape(kw), re.IGNORECASE)
            for pno, raw in enumerate(pages, start=1):
                flat = normalize(raw)
                for m in pat.finditer(flat):
                    # 取命中点前后 260 字符作为上下文
                    lo = max(0, m.start() - 260)
                    hi = min(len(flat), m.end() + 260)
                    print(f"  <{kw}> p{pno}: ...{flat[lo:hi]}...")
                    hits += 1
                    if hits >= 6:       # 每个关键词每篇最多 6 条，避免刷屏
                        break
                if hits >= 6:
                    break
            if hits == 0:
                print(f"  <{kw}> 未命中")
        print()


def dump(name_filter: str) -> None:
    """导出匹配论文的整篇纯文本（供人工细读）。"""
    for pdf in sorted(PAPER_DIR.glob("*.pdf")):
        if name_filter.lower() in pdf.name.lower():
            for pno, raw in enumerate(load_pages(pdf), start=1):
                print(f"\n----- p{pno} -----")
                print(raw)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--dump":
        dump(args[1])
    else:
        grep(args)
