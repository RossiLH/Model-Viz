# -*- coding: utf-8 -*-
"""K3 PostTrain 交互式演示 — 启动入口。

运行：python gradio_posttrain_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

from plot_utils import setup_cjk_font
setup_cjk_font()

from _posttrain_demo_ui import build_demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
