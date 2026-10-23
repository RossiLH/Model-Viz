# -*- coding: utf-8 -*-
"""K3 Optimizer 交互式演示 — 启动入口。

运行：
  cd Model-Viz/experiments/models/llm/kimi_k3/optim
  .venv/Scripts/python.exe gradio_optim_demo.py
"""
from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")

from plot_utils import setup_cjk_font
setup_cjk_font()

from _optim_demo_ui import build_demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
