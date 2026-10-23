# -*- coding: utf-8 -*-
"""DeepSeek 系列论文关键词检索脚本（MoE 知识点专用，可通用）。

用途：在 `Reference/Paper/DeepSeek` 下的 PDF 中做一手取证，回答诸如
「sqrt(softplus) 打分、aux-loss-free 偏置、hash 路由、SwiGLU clamp、
Anticipatory Routing 究竟出自论文哪一节、原文怎么表述」这类归属问题。

做法：把 PDF 逐页抽成纯文本 → 压平换行/断词 → 按关键词做上下文检索，
打印「论文 → 页码 → 命中上下文」三元组。

用法：
    python scripts/deepseek_v4/pdf_grep_dsv4.py <关键词1> <关键词2> ...
    python scripts/deepseek_v4/pdf_grep_dsv4.py --paper V4 <关键词> ...
    python scripts/deepseek_v4/pdf_grep_dsv4.py --dump V4         # 导出整篇纯文本

说明：脚本只依赖 pypdf，CPU 可跑，无需 GPU / 网络。
"""
from __future__ import annotations

import pathlib
import re
import sys

from pypdf import PdfReader

# 论文目录（相对仓库根 Model-Viz-Impl；本文件在 Model-Viz/scripts/deepseek_v4/ 下）
PAPER_DIR = pathlib.Path(__file__).resolve().parents[3] / "Reference" / "Paper" / "DeepSeek"

# 每个关键词每篇最多打印多少条命中，避免刷屏
MAX_HITS_PER_KEYWORD = 8
# 命中点前后取多少字符作为上下文
CONTEXT_CHARS = 320


def load_pages(pdf_path: pathlib.Path) -> list[str]:
    """抽取 PDF 每页文本，返回 list[str]（索引 0 = 第 1 页）。"""
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # 个别页可能含异常字体，跳过但保留占位保证页码对齐
            pages.append("")
            print(f"  [warn] 解析失败: {exc}", file=sys.stderr)
    return pages


def normalize(text: str) -> str:
    """把换行 / 连字符断词压平，便于跨行匹配关键词。"""
    text = text.replace("-\n", "")        # 行末连字符断词（如 "rout-\ning"）
    text = re.sub(r"\s+", " ", text)      # 多空白压成单空格
    return text


def grep(keywords: list[str], name_filter: str | None = None) -> None:
    """在所有（或过滤后的）论文中检索关键词，打印命中上下文。"""
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
                    lo = max(0, m.start() - CONTEXT_CHARS)
                    hi = min(len(flat), m.end() + CONTEXT_CHARS)
                    print(f"  <{kw}> p{pno}: ...{flat[lo:hi]}...")
                    hits += 1
                    if hits >= MAX_HITS_PER_KEYWORD:
                        break
                if hits >= MAX_HITS_PER_KEYWORD:
                    break
            if hits == 0:
                print(f"  <{kw}> 未命中")
        print()


def dump(name_filter: str) -> None:
    """导出匹配论文的整篇纯文本（供人工细读 / 再检索）。"""
    for pdf in sorted(PAPER_DIR.glob("*.pdf")):
        if name_filter.lower() in pdf.name.lower():
            for pno, raw in enumerate(load_pages(pdf), start=1):
                print(f"\n----- p{pno} -----")
                print(raw)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--dump":
        dump(args[1])
    elif args and args[0] == "--paper":
        grep(args[2:], name_filter=args[1])
    else:
        grep(args)
