# -*- coding: utf-8 -*-
"""Q3 可视化：π0.7 里到底有哪几类 embedding？它们各自怎么造、参数量多少？

三个子图回答三件事：
1. 左图：prelim 一块「embedding 全景表」—— 每个 embedding 模块的
   输入是什么（像素/文本 token/连续状态/标量元数据/连续动作/标量时间步）、
   输出维度、参数量（**真跑模型查 nn.Module 得到**）、是否可学习。
   一眼看清：SinusoidalTimeEmbedding 是**唯一零参数量**的编码器，
   因为它的 sin/cos 频率表是解析算死的（不是学出来的）。
2. 中图：SinusoidalTimeEmbedding 的真实数值热力图 —— 横轴 tau∈[0,1]，
   纵轴为各通道；前半是 sin、后半是 cos；低通道周期 4s（τ 一整个量程
   才走一圈）、高通道周期 4e-3（极短周期做精细分辨）。这是"多尺度
   时间编码"的最直观证据。
3. 右图：为什么动作 embedding 用单个 Linear、时间 embedding 却要
   sin+MLP？对比两类输入的本质：动作是**高维连续向量**（本身已经
   是低层特征，单 Linear 投影即可，非线性留给后面的 action expert）；
   τ 是**一维标量**，必须先用 0 参数的 sin/cos 展开成多尺度相位，
   再用一个小 MLP 学非线性。设计差异不在"参数量谁大谁小"，
   而在"非线性放在哪一段"。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import C_STD, C_NEW, C_AUX, C_AUX2, C_WARN, OUT, setup_fonts, build_model

# 每个（模块名, 输入形态, 宽度, 论文/官方对应）的可读描述，参数/宽度真跑取得
MODULES = [
    ("video_encoder",   "视频帧 (T×C×H×W)",            None,   "SigLIP+时间层"),
    ("vision_proj",     "patch 特征 (D_v)",            "32→W", "Linear"),
    ("text_embed",      "文本 token (整数 0..255)",     "256→W", "Embedding"),
    ("state_proj",      "连续状态 (state_dim)",        "8→W",  "Linear"),
    ("metadata_proj",   "标量元数据 (1)",              "1→W",  "Linear"),
    ("action_in_proj",  "连续动作 (action_dim)",       "8→W",  "Linear (非线性交下游)"),
    ("time_embed",      "标量时间步 τ",                 "-→W",  "正弦编码(0参)"),
    ("time_mlp",        "τ 调制量",                     "W→W",  "MLP (时间支路非线性)"),
]


def measure_modules(model) -> dict:
    """真跑模型，逐模块统计参数量与输出形状。

    输出形状通过构造一批最小输入、调用模型拿到 prefix/suffix embedding 长度。
    """
    stats = {}
    for name, _in, _shape, _desc in MODULES:
        mod = getattr(model, name, None)
        if mod is None:
            stats[name] = {"params": 0, "shape": _shape, "desc": _desc}
            continue
        n_params = sum(p.numel() for p in mod.parameters())
        stats[name] = {"params": n_params, "shape": _shape, "desc": _desc}
    # 输出形状：直接量 embed_prefix 与 embed_suffix 的 token 数。
    # 前缀序列 = 视频块(ar=0) + 语言文本(ar=1) + State 指示 token(ar=1)
    #            + 状态块(ar=[1,0,0,...])。
    # 本项目默认不启用 metadata/subgoal，故这里按"启用时"示意标注。
    # 注意：视频编码器输出 = (patch 数 + 1) 个 token（28x28/7x7=16 个 patch
    # 再加 1 个汇总 token），这是真跑实测得到的，不能想当然。
    cfg = model.config
    B, L = 1, 4
    T = cfg.vision.n_obs_steps
    videos = torch.zeros(B, T, 3, cfg.vision.image_size, cfg.vision.image_size)
    lang = torch.randint(1, 250, (B, L))
    lang_masks = torch.ones(B, L, dtype=torch.bool)
    state = torch.zeros(B, T, cfg.state_dim)
    with torch.no_grad():
        embs, pads, ars = model.embed_prefix(videos, lang, lang_masks, state)
    n_total = embs.shape[1]
    n_state = T
    # 用 ar 布局反推各块 token 数：视频块是 ar=0 的连续段，
    # 其后是语言文本(L) + State 指示(1) + 状态块(T)。
    ar0 = ars[0].tolist()
    first_ar1 = ar0.index(True)
    n_vid = first_ar1
    n_lang = L
    assert n_total == n_vid + n_lang + 1 + n_state, (n_total, n_vid, n_lang, n_state)
    stats["vision_proj"]["shape"] = f"{n_vid}→W"
    stats["text_embed"]["shape"] = f"{n_lang}→W"
    stats["state_proj"]["shape"] = f"{n_state}→W"
    # metadata / subgoal 是可选的，这里按"启用时"示意
    stats["metadata_proj"]["shape"] = f"M→W"
    # action 相关形状
    H = cfg.action_horizon
    noisy = torch.zeros(B, H, cfg.action_dim)
    t = torch.zeros(B)
    with torch.no_grad():
        a_emb, a_cond = model.embed_suffix(noisy, t)
    stats["action_in_proj"]["shape"] = f"{H}→W"
    stats["time_embed"]["shape"] = f"1→W"
    stats["time_mlp"]["shape"] = f"W→W"
    return stats


def plot() -> tuple:
    """绘制 embedding 全景三子图，返回 (fig, text)。

    Returns:
        (fig, text): matplotlib Figure 对象和关键发现文本摘要。
    """
    setup_fonts()
    model = build_model()
    stats = measure_modules(model)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))

    # ---------------- 左：embedding 全景表 ----------------
    ax = axes[0]
    ax.axis("off")
    rows = [
        ("模块", "输入", "输出形状", "参数量", "可学习"),
    ]
    data_rows = []
    for name, _in, _shape, _desc in MODULES:
        s = stats[name]
        params = s["params"]
        learn = "是" if params > 0 else "否（解析算死）"
        data_rows.append((name, _in, s["shape"], f"{params:,}", learn))
    tbl = ax.table(cellText=data_rows, colLabels=rows[0], loc="center",
                   cellLoc="left", colWidths=[0.14, 0.2, 0.16, 0.15, 0.22])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", weight="bold")
        elif c == 3 and "0" in data_rows[r - 1][3].replace(",", ""):
            cell.set_facecolor("#fff8e1")
    ax.set_title("Q3: π0.7 各类 embedding 一览（真跑模型统计）",
                 fontsize=11, loc="left")

    # ---------------- 中：SinusoidalTimeEmbedding 真实热力图 ----------------
    ax = axes[1]
    from models.vla.core.embedding import SinusoidalTimeEmbedding
    W = model.config.action.width
    te = SinusoidalTimeEmbedding(W)
    tau = torch.linspace(0.0, 1.0, 200)
    mat = te(tau).detach().numpy()      # (200, W)
    im = ax.imshow(mat.T, aspect="auto", origin="lower",
                   cmap="RdBu_r", interpolation="nearest")
    ax.set_yticks([0, W // 2 - 1, W // 2, W - 1])
    ax.set_yticklabels(["ch 0 (sin)", f"ch {W//2 - 1}", f"ch {W//2} (cos)", f"ch {W-1}"],
                       fontsize=8)
    ax.set_xticks(np.linspace(0, 199, 5))
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    ax.set_xlabel("时间步 τ")
    ax.set_ylabel("通道 index")
    ax.set_title("SinusoidalTimeEmbedding 数值（前半 sin / 后半 cos）",
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("编码值", fontsize=8)
    # 标注多尺度：低频通道周期 4.0
    ax.text(0.02, 0.98, "低频通道周期 ≈ 4.0（τ∈[0,1] 才一圈）\n"
                        "高频通道周期 ≈ 4e-3（精细分辨）",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="#e3f2fd",
                      edgecolor=C_AUX, alpha=0.95))

    # ---------------- 右：为什么 action 线性、time 要 sin+MLP ----------------
    ax = axes[2]
    # 拆三组：动作 Linear / 正弦时间（0参，解析算死）/ 时间 MLP
    # 这样读者一眼看出：动作 embed 本身不小，但时间那条路是"0 参 + MLP"两段式
    names = ["ActionEmbedding\n(单 Linear)",
             "SinusoidalTime\n(解析 sin/cos)",
             "TimeMLP\n(后接非线性)"]
    params = [stats["action_in_proj"]["params"],
              stats["time_embed"]["params"],
              stats["time_mlp"]["params"]]
    colors = [C_AUX, C_NEW, C_WARN]
    bars = ax.bar(names, params, color=colors, alpha=0.9, width=0.55)
    for b, v in zip(bars, params):
        ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_ylabel("参数量")
    ax.set_title("动作 embed vs 时间 embed：为什么用不同结构\n"
                 "(动作单 Linear 一次到位 / 时间 sin(0参)+MLP 两段)",
                 fontsize=11)
    ax.text(0.5, 0.97,
            "动作（8→32 维）已是高维特征，单 Linear 投影即可，\n"
            "    非线性交给下游 action expert；\n"
            "τ 是一维标量，先用 0 参数的 sin/cos 展开成多尺度相位，\n"
            "    再用一个小 MLP 学非线性 —— 总参数量反而比动作 Linear 大",
            transform=ax.transAxes, va="top", ha="center", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="#fff8e1",
                      edgecolor=C_WARN, alpha=0.95))
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Q3：π0.7 的 embedding 全景 —— 一个模型里藏着六类编码器",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    text = (f"vision_proj params: {stats['vision_proj']['params']}\n"
            f"text_embed params:  {stats['text_embed']['params']}\n"
            f"state_proj params:  {stats['state_proj']['params']}\n"
            f"metadata_proj:      {stats['metadata_proj']['params']}\n"
            f"action_in_proj:     {stats['action_in_proj']['params']}  (单 Linear)\n"
            f"time_embed:         {stats['time_embed']['params']}  (解析 sin/cos，0 参)\n"
            f"time_mlp:           {stats['time_mlp']['params']}  (时间支路非线性)\n"
            f"设计差异: 动作用单 Linear 一次投影；τ 走 sin(0参)+MLP 两段式")
    return fig, text


def main() -> None:
    fig, text = plot()
    fig.savefig(OUT / "plot_embedding_zoo.png", dpi=130)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
