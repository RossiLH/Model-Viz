# -*- coding: utf-8 -*-
"""XTML 对话模板可视化：区域划分、往返验证、语法状态机。"""
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _ax_title, save,
)

from models.llm.kimi_k3.posttrain.test_helpers import (
    sample_conversation, verify_xtml_roundtrip,
)
from models.llm.kimi_k3.posttrain.xtml import (
    serialize, END_OF_MSG, allowed_next_tokens,
)

_ZONE_COLORS = {
    "全局声明": "#e74c3c",
    "历史消息": "#3498db",
    "单次选项": "#f39c12",
    "生成前缀": "#2ecc71",
}


def _classify_segment(seg: str, idx: int, total: int) -> str:
    stripped = seg.strip()
    if idx == 0:
        if stripped.startswith("[open]message"):
            return "历史消息"
        return "全局声明"
    if 'role="option"' in stripped:
        return "单次选项"
    if idx == total - 1:
        return "生成前缀"
    return "历史消息"


def fig_xtml_zones(kind: str = "weather", thinking: bool = True,
                   with_tools: bool = True, effort: str = "high") -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    conv = sample_conversation(kind)
    if not with_tools:
        conv.tools = []
        conv.tool_choice = None
    if effort in ("low", "medium", "high"):
        conv.thinking_effort = effort

    text = serialize(conv, thinking=thinking, add_generation_prefix=True)
    total_len = len(text)

    parts = text.split(END_OF_MSG)
    n_segs = len(parts)

    zones = []
    seg_lengths = []
    seg_labels = []
    for i, part in enumerate(parts):
        zone = _classify_segment(part, i, n_segs)
        zones.append(zone)
        seg_lengths.append(len(part))
        seg_labels.append(f"段 {i + 1}  {zone}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=120,
                                    gridspec_kw={"height_ratios": [1, 3]})

    left = 0
    for i, (zone, length) in enumerate(zip(zones, seg_lengths)):
        if length == 0:
            if i < len(zones) - 1:
                eom_len = len(END_OF_MSG)
                ax1.barh(0, eom_len, left=left, color="#95a5a6",
                         edgecolor="white", linewidth=0.5)
                left += eom_len
            continue
        ax1.barh(0, length, left=left, color=_ZONE_COLORS[zone],
                 edgecolor="white", linewidth=1)
        if length > total_len * 0.03:
            ax1.text(left + length / 2, 0, zone, ha="center", va="center",
                     fontsize=9, color="white", fontweight="bold",
                     fontproperties=fp)
        left += length
        if i < len(zones) - 1:
            eom_len = len(END_OF_MSG)
            ax1.barh(0, eom_len, left=left, color="#95a5a6",
                     edgecolor="white", linewidth=0.5)
            left += eom_len

    ax1.set_xlim(0, max(left, 1))
    ax1.set_yticks([])
    ax1.set_xticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)
    _ax_title(ax1, "XTML 文本区域划分（堆叠视图）")

    y_pos = list(range(n_segs))
    colors = [_ZONE_COLORS[z] for z in zones]
    ax2.barh(y_pos, seg_lengths, color=colors, edgecolor="white", linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(seg_labels, fontproperties=fp, fontsize=9)
    ax2.set_xlabel("字符数", fontproperties=fp)
    ax2.invert_yaxis()
    max_len = max(seg_lengths) if seg_lengths else 1
    for i, v in enumerate(seg_lengths):
        ax2.text(v + max_len * 0.01, i, str(v), va="center", fontsize=8)
    _ax_title(ax2, "各段消息长度分布")

    legend_patches = [mpatches.Patch(color=c, label=z)
                      for z, c in _ZONE_COLORS.items()]
    ax2.legend(handles=legend_patches, loc="lower right",
               prop=fp, fontsize=9)

    fig.tight_layout()

    zone_order = []
    seen = set()
    for z in zones:
        if z not in seen:
            zone_order.append(z)
            seen.add(z)

    return fig, _text(
        f"总长度：{total_len} 字符",
        f"段数：{n_segs} 段（含 END_OF_MSG 分隔符）",
        f"区域顺序：{' -> '.join(zone_order)}",
        f"thinking：{thinking}，tools：{with_tools}，effort：{effort}",
    )


def fig_roundtrip(kind: str = "multi", thinking: bool = True) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    conv = sample_conversation(kind)
    # add_generation_prefix=True 与 fig_xtml_zones 保持一致（推理时的真实文本形态）。
    # verify_xtml_roundtrip 内部会剔除生成前缀解析出的空桩后再比对。
    result = verify_xtml_roundtrip(conv, thinking=thinking,
                                   add_generation_prefix=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    border_color = "#27ae60" if result.ok else "#e74c3c"
    status_text = "PASS" if result.ok else "FAIL"
    bg_color = "#eafaf1" if result.ok else "#fdedec"

    rect = mpatches.FancyBboxPatch(
        (0.3, 0.3), 9.4, 9.4,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        facecolor=bg_color, edgecolor=border_color, linewidth=4,
    )
    ax.add_patch(rect)

    ax.text(5, 7.2, status_text, ha="center", va="center",
            fontsize=56, fontweight="bold", color=border_color,
            fontproperties=fp)

    ax.text(5, 5.6, "XTML 往返一致性验证", ha="center", va="center",
            fontsize=14, color="#2c3e50", fontweight="bold",
            fontproperties=fp)

    info_items = [
        ("消息数", str(result.messages)),
        ("字符数", str(result.text_len)),
        ("工具声明", "有" if result.tools else "无"),
        ("thinking_effort", result.thinking_effort or "None"),
        ("thinking 模式", "开启" if thinking else "关闭"),
    ]

    n_items = len(info_items)
    box_w = 8.5 / n_items
    for i, (label, value) in enumerate(info_items):
        x = 0.75 + i * box_w
        y = 3.5
        inner = mpatches.FancyBboxPatch(
            (x, y - 1.0), box_w - 0.15, 1.8,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor="white", edgecolor=border_color, linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(inner)
        ax.text(x + (box_w - 0.15) / 2, y + 0.4, value,
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="#2c3e50", fontproperties=fp)
        ax.text(x + (box_w - 0.15) / 2, y - 0.4, label,
                ha="center", va="center", fontsize=9,
                color="#7f8c8d", fontproperties=fp)

    if not result.ok and result.mismatch:
        mismatch_lines = ["不匹配项："]
        for m in result.mismatch[:4]:
            mismatch_lines.append(f"  - {m}")
        mismatch_text = "\n".join(mismatch_lines)
        ax.text(5, 1.0, mismatch_text, ha="center", va="center",
                fontsize=9, color="#c0392b", fontproperties=fp,
                linespacing=1.5)

    fig.tight_layout()

    return fig, _text(
        f"验证结果：{status_text}",
        f"消息数：{result.messages}（已剔除生成前缀产生的空桩）",
        f"字符数：{result.text_len}",
        f"工具声明：{'有' if result.tools else '无'}",
        f"thinking_effort：{result.thinking_effort}",
        *([f"不匹配项：{'；'.join(result.mismatch)}"] if result.mismatch else []),
    )


def fig_grammar_fsm(thinking: bool = True) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    states = {
        "None":     (1.5, 3.5, "#95a5a6"),
        "think":    (4.5, 5.5, "#3498db"),
        "response": (4.5, 1.5, "#2ecc71"),
        "tools":    (7.8, 3.5, "#e67e22"),
    }

    state_r = 0.8

    for name, (x, y, color) in states.items():
        circle = mpatches.Circle((x, y), state_r, facecolor=color,
                                 edgecolor="white", linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", fontproperties=fp)

    def _arrow(x1, y1, x2, y2, label, label_offset=(0, 0), color="#34495e", rad=0.1):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    lw=1.8, connectionstyle=f"arc3,rad={rad}"))
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
                color=color, fontproperties=fp,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=color, alpha=0.9))

    start_token_thinking = allowed_next_tokens(None, thinking=True)[0]
    start_token_instruct = allowed_next_tokens(None, thinking=False)[0]

    if thinking:
        _arrow(1.5 + state_r, 3.9, 4.5 - state_r, 5.2,
               start_token_thinking, label_offset=(0, 0.3))
        _arrow(4.5 + state_r * 0.3, 5.5 - state_r,
               4.5 + state_r * 0.3, 1.5 + state_r,
               allowed_next_tokens("think", thinking=True)[0],
               label_offset=(0.7, 0), color="#3498db", rad=0.0)
    else:
        _arrow(1.5 + state_r, 3.1, 4.5 - state_r, 1.8,
               start_token_instruct, label_offset=(0, -0.3))

    _arrow(4.5 + state_r, 1.8, 7.8 - state_r, 3.2,
           "[open]tools", label_offset=(0.2, -0.25), color="#e67e22")

    tools_tokens = allowed_next_tokens("tools", thinking=True)
    _arrow(7.8 - state_r, 3.8, 4.5 + state_r, 5.2,
           tools_tokens[0], label_offset=(-0.2, 0.3), color="#e67e22", rad=-0.1)

    ax.annotate("", xy=(7.8 + state_r * 0.2, 3.5 + state_r * 0.9),
                xytext=(7.8 + state_r, 3.5 + state_r * 0.3),
                arrowprops=dict(arrowstyle="->", color="#e74c3c",
                                lw=1.8, connectionstyle="arc3,rad=0.8"))
    ax.text(8.9, 4.6, END_OF_MSG, ha="center", va="center", fontsize=7.5,
            color="#e74c3c", fontproperties=fp,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.9))

    ax.text(5, 0.4,
            f"起始差异：thinking 模式 = {start_token_thinking}   |   "
            f"instruct 模式 = {start_token_instruct}",
            ha="center", va="center", fontsize=9,
            color="#7f8c8d", fontproperties=fp,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa",
                      edgecolor="#bdc3c7"))

    _ax_title(ax, "XTML 语法状态机 (FSM)")

    fig.tight_layout()

    return fig, _text(
        "4 个状态：None -> think -> response -> tools",
        f"thinking 起始：{start_token_thinking}",
        f"instruct 起始：{start_token_instruct}",
        f"tools 合法出口：{', '.join(tools_tokens)}",
    )


if __name__ == "__main__":
    fig, _ = fig_xtml_zones(kind="multi", thinking=True, with_tools=True, effort="high")
    p1 = save(fig, "plot_xtml_zones.png")
    print("saved:", p1)

    fig, _ = fig_roundtrip(kind="multi", thinking=True)
    p2 = save(fig, "plot_xtml_roundtrip.png")
    print("saved:", p2)

    fig, _ = fig_grammar_fsm(thinking=True)
    p3 = save(fig, "plot_grammar_fsm.png")
    print("saved:", p3)
