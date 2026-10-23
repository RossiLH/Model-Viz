# -*- coding: utf-8 -*-
"""MTP / EAGLE-3 交互式演示（Gradio 启动文件）。

13 个 Tab，详见 gradio_mtp_demo_doc.md。

运行:
    cd experiments/models/llm/kimi_k3/mtp
    python gradio_mtp_demo.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
_EXP_ROOT = str(Path(__file__).resolve().parents[5] / "experiments")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font
setup_cjk_font()

from _mtp_demo_ui import build_demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
