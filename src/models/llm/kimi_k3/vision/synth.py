# -*- coding: utf-8 -*-
"""离线合成多模态数据（测试/演示用，无下载依赖）。

论文的视觉语料（§3.1）涵盖：caption、交错图文文档、OCR、感知、视频、视觉编程，
并提供绝对 + 归一化 [0,1] 两种坐标监督格式（分辨率鲁棒的定位）。那些语料无法
随仓库分发，因此本模块生成一个**结构相同**的程序化替身：渲染好的场景配上图注与
坐标（两种格式都有）。

这个"结构"正是它的价值所在。论文大幅扩展程序化多模态数据 —— 代码片段与其渲染
产物的精确配对（SVG、3D 资产、网页、游戏、CAD）——正是因为这种配对是**精确且
机器可校验**的。同一思路让整个视觉通路在本项目里无需任何下载即可测试。

本实现逐位对齐 ``Reference/kimi3-master/kimi3/vision/synth.py``。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor

__all__ = ["Shape", "Scene", "render_scene", "SyntheticVisionSource"]

# 固定色板：RGB (0..1)。颜色名字同时也用作 caption 的词。
COLORS = {
    "red": (1.0, 0.2, 0.2),
    "green": (0.2, 0.8, 0.3),
    "blue": (0.25, 0.4, 0.95),
    "yellow": (0.95, 0.85, 0.2),
}


@dataclass
class Shape:
    kind: str     # 'square' | 'circle' | 'triangle'
    color: str    # 'red' | ...
    cx: int       # 中心 x（像素）
    cy: int       # 中心 y（像素）
    size: int     # 边长/直径（像素）

    def bbox(self) -> tuple[int, int, int, int]:
        """包围盒 (x0, y0, x1, y1)。"""
        h = self.size // 2
        return self.cx - h, self.cy - h, self.cx + h, self.cy + h


@dataclass
class Scene:
    pixels: Tensor              # (3, H, W)，像素值在 [0,1]
    shapes: list[Shape]
    caption: str
    height: int
    width: int

    def coordinates(self, normalized: bool) -> list[tuple[str, tuple[float, ...]]]:
        """坐标监督的两种格式。

        - ``normalized=False``：绝对像素坐标（x0,y0,x1,y1）；
        - ``normalized=True``：归一化到 [0,1]（除以宽/高），分辨率鲁棒。
        """
        out = []
        for s in self.shapes:
            x0, y0, x1, y1 = s.bbox()
            if normalized:
                box = (x0 / self.width, y0 / self.height, x1 / self.width, y1 / self.height)
            else:
                box = (float(x0), float(y0), float(x1), float(y1))
            out.append((f"{s.color} {s.kind}", box))
        return out


def render_scene(
    shapes: list[Shape], height: int, width: int, *, background: float = 0.1
) -> Tensor:
    """把形状光栅化到画布上。确定性、无第三方依赖。

    实现技巧：用 meshgrid 生成 x/y 网格，对每个形状算一个布尔 mask，
    ``torch.where(mask, color, canvas)`` 就地覆盖。
    """
    canvas = torch.full((3, height, width), background)
    ys = torch.arange(height).view(-1, 1)
    xs = torch.arange(width).view(1, -1)
    for shape in shapes:
        colour = torch.tensor(COLORS[shape.color]).view(3, 1, 1)
        half = shape.size // 2
        if shape.kind == "square":
            # 方形：x 与 y 都在 [cx±half, cy±half] 内。
            mask = (
                (xs >= shape.cx - half) & (xs <= shape.cx + half)
                & (ys >= shape.cy - half) & (ys <= shape.cy + half)
            )
        elif shape.kind == "circle":
            # 圆：到中心的欧氏距离 <= half。
            mask = ((xs - shape.cx) ** 2 + (ys - shape.cy) ** 2) <= half**2
        else:  # triangle（上宽下窄的等腰三角）
            # y 在 [cy-half, cy+half]，且 x 距中心不超过"当前高度/2"（向底部变宽）。
            mask = (
                (ys >= shape.cy - half) & (ys <= shape.cy + half)
                & ((xs - shape.cx).abs() <= (ys - (shape.cy - half)) // 2)
            )
        canvas = torch.where(mask.unsqueeze(0), colour.expand(3, height, width), canvas)
    return canvas


class SyntheticVisionSource:
    """确定性场景流：每个 sample 有像素、caption、绝对/归一化坐标。

    用作视觉通路的最小可运行数据源 —— 不下载、可复现（固定 seed）。
    """

    KINDS = ("square", "circle", "triangle")

    def __init__(self, n: int = 64, height: int = 56, width: int = 56, seed: int = 0):
        self.n = n
        self.height = height
        self.width = width
        self.seed = seed

    def __iter__(self) -> Iterator[Scene]:
        rng = random.Random(self.seed)
        for _ in range(self.n):
            count = rng.randint(1, 3)
            shapes = []
            for _ in range(count):
                size = rng.randint(8, 18)
                shapes.append(
                    Shape(
                        kind=rng.choice(self.KINDS),
                        color=rng.choice(list(COLORS)),
                        cx=rng.randint(size, self.width - size),
                        cy=rng.randint(size, self.height - size),
                        size=size,
                    )
                )
            parts = [f"a {s.color} {s.kind}" for s in shapes]
            caption = "the image shows " + " and ".join(parts) + "."
            yield Scene(
                pixels=render_scene(shapes, self.height, self.width),
                shapes=shapes,
                caption=caption,
                height=self.height,
                width=self.width,
            )