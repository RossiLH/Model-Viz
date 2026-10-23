# -*- coding: utf-8 -*-
"""AttnRes（Attention Residuals）教学演示 —— Block 形式的"深度维 softmax 注意力"。

运行:
    python experiments/attnres/demo_attn_res.py
产出:
    experiments/attnres/demo_attn_res.png   (2x2 四联图)
    experiments/attnres/demo_attn_res.json  (关键数值，供文档引用)

演示内容（对应论文 Eq.8-10 与本项目 attn_res.py 的真实行为）:
  1. w=0 初始化 == 均匀混合 == 标准残差: 训练前 alpha 恒等均匀, 与"等权相加"无差别;
  2. 学习后 alpha 的"选择性读取": 热力图展示每个 layer 实际重点读哪些源
     (embedding 永远是最强源 —— 这正是 test_embedding_is_always_a_source 的现象);
  3. Block 形式内存有界: 已折叠块数 + 至多 1 个 partial, 源数不随 n_units 线性增长
     (对比 Full 形式 O(L) 源, 见 test_memory_is_bounded_by_block_count);
  4. 训练曲线对照: 同样层权重, AttnRes 能按内容路由到 embedding, 收敛快于纯残差。

任务设计: 给定随机输入 x, 模型必须"找回"输入本身 (目标 = embedding)。
纯残差把 6 层输出等权加起来 —— 输入信息被逐层稀释; AttnRes 则学会把权重
集中在 embedding 源上, 从而直接读回 x。这是理解"选择性读取"的最小任务。
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from plot_utils import setup_cjk_font  # noqa: E402  (experiments/attnres/plot_utils.py)

setup_cjk_font()

from models.llm.kimi_k3.modules.attn_res import AttentionResiduals  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent


def build_problem(dim: int, n_units: int, block_units: int, seed: int = 0):
    """构造共享层权重 + AttnRes 模块。

    层权重 l 与 nn.Sequential 共享, 使得 AttnRes 与标准残差在同一组 F_l 上比较,
    唯一的变量是"跨层路由"方式。queries 按论文规定 zero 初始化 (w=0 -> 均匀)。
    """
    torch.manual_seed(seed)
    layers = torch.nn.ModuleList(
        [torch.nn.Sequential(torch.nn.Linear(dim, dim), torch.nn.GELU()) for _ in range(n_units)]
    )
    attnres = AttentionResiduals(dim, n_units, block_units)
    # 所有层 + attnres 参数可训练; queries 保持 zero 初始化 (论文 canonical)。
    return layers, attnres


def forward_attnres(layers, attnres, emb: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """按本项目模型集成方式逐 unit 前向 (model.py:158-171 的简化版)。"""
    state = attnres.init_state(emb)
    alphas = []
    for i, layer in enumerate(layers):
        alphas.append(attnres.attention_weights(state, i))   # (n_sources, b, t)
        h = attnres.mix(state, i)
        out = layer(h)
        state = attnres.add_output(state, out)
    out = attnres.finalize(state)
    return out, alphas


def forward_residual(layers, emb: torch.Tensor) -> torch.Tensor:
    """标准残差基线: h = h + F_l(h)  (PreNorm 的"纯加法"形式)。"""
    h = emb
    for layer in layers:
        h = h + layer(h)
    return h


def trace_state_growth(n_units: int, block_units: int, dim: int = 16, seed: int = 1):
    """Block vs Full 的"可见源数"随 unit 的演进 (Full 用 block_units=1 复原)。"""
    rows = []
    for label, bu in [("Block(3)", block_units), ("Full(1)", 1)]:
        layers, attnres = build_problem(dim, n_units, bu, seed=seed)
        emb = torch.randn(1, 1, dim)
        state = attnres.init_state(emb)
        curve = [len(state.sources())]
        for i, layer in enumerate(layers):
            state = attnres.add_output(state, layer(attnres.mix(state, i)))
            curve.append(len(state.sources()))
        rows.append({"label": label, "sources_per_unit": curve})
    return rows


def fig_demo_attn_res(
    dim: int = 16,
    n_units: int = 6,
    block_units: int = 3,
    steps: int = 260,
    batch: int = 16,
    lr: float = 0.02,
    seed: int = 0,
) -> tuple[plt.Figure, str]:
    """生成 AttnRes 2x2 四联图并返回 (fig, text)。

    用于 Gradio demo 直接调用，避免维护两份代码。

    Args:
        dim: 隐藏维度
        n_units: 子层数量
        block_units: 块大小
        steps: 训练步数
        batch: 批大小
        lr: 学习率
        seed: 随机种子
    Returns:
        (fig, text): matplotlib Figure 和结论文本
    """
    torch.set_num_threads(2)
    torch.manual_seed(seed)

    layers, attnres = build_problem(dim, n_units, block_units)
    opt = torch.optim.Adam(list(layers.parameters()) + list(attnres.parameters()), lr=lr)
    opt_res = torch.optim.Adam(list(layers.parameters()), lr=lr)

    probe = torch.randn(batch, 1, dim)
    _, alphas_before = forward_attnres(layers, attnres, probe)

    loss_a, loss_r = [], []
    for step in range(steps):
        x = torch.randn(batch, 1, dim)
        opt.zero_grad()
        out_a, _ = forward_attnres(layers, attnres, x)
        la = torch.nn.functional.mse_loss(out_a, x)
        la.backward()
        opt.step()
        loss_a.append(la.item())
        opt_res.zero_grad()
        out_r = forward_residual(layers, x)
        lr_ = torch.nn.functional.mse_loss(out_r, x)
        lr_.backward()
        opt_res.step()
        loss_r.append(lr_.item())

    _, alphas_after = forward_attnres(layers, attnres, probe)
    growth = trace_state_growth(n_units, block_units)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    ax = axes[0][0]
    w0 = alphas_before[-1][:, 0, 0].detach().cpu()
    ax.plot(range(1, len(w0) + 1), w0, "o-", color="#888")
    ax.set_title("(a) 训练前 alpha（w=0 → 均匀 = 标准残差）")
    ax.set_xlabel("source 编号")
    ax.set_ylabel("alpha 权重")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    w1 = alphas_after[-1][:, 0, 0].detach().cpu()
    ax.bar(range(1, len(w1) + 1), w1, color=["#d62728" if i == 1 else "#1f77b4" for i in range(len(w1))])
    ax.set_title("(b) 训练后 alpha：学会重点读 embedding")
    ax.set_xlabel("source 编号（1 = embedding）")
    ax.set_ylabel("alpha 权重")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    for row in growth:
        ax.plot(range(len(row["sources_per_unit"])), row["sources_per_unit"], "o-",
                label=row["label"])
    ax.axhline(n_units // block_units + 2, color="red", ls="--", alpha=0.6,
               label="Block 上界 (n_units//block_units+2)")
    ax.set_title("(c) 可见源数随 unit 演进：Block 内存有界 vs Full 线性")
    ax.set_xlabel("unit 编号")
    ax.set_ylabel("source 数（= 内存占用量）")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    ax.semilogy(range(steps), loss_a, label="AttnRes (softmax 路由)")
    ax.semilogy(range(steps), loss_r, label="标准残差 (等权相加)")
    ax.set_title("(d) copy 任务训练曲线")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE (log)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    text = (
        f"训练后最后一 unit 的 alpha (source 1 = embedding):\n"
        f"  {'  '.join(f'{v:.3f}' for v in w1.tolist())}\n\n"
        f"最终 MSE: AttnRes={loss_a[-1]:.4f}  标准残差={loss_r[-1]:.4f}\n"
        f"Block 源数演进 (n_units={n_units}, block_units={block_units}): "
        f"{growth[0]['sources_per_unit']}"
    )
    return fig, text


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)

    # ------------------------------ 超参 ------------------------------
    DIM, N_UNITS, BLOCK_UNITS = 16, 6, 3
    STEPS, BATCH, LR = 260, 16, 0.02

    fig, text = fig_demo_attn_res(DIM, N_UNITS, BLOCK_UNITS, STEPS, BATCH, LR)
    png = OUT / "demo_attn_res.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    print("saved", png)
    print(text)

    # 计算数值用于 JSON 输出
    layers, attnres = build_problem(DIM, N_UNITS, BLOCK_UNITS)
    probe = torch.randn(BATCH, 1, DIM)
    _, alphas_before = forward_attnres(layers, attnres, probe)
    _, alphas_after = forward_attnres(layers, attnres, probe)
    growth = trace_state_growth(N_UNITS, BLOCK_UNITS)

    w0 = alphas_before[-1][:, 0, 0].detach().cpu()
    w1 = alphas_after[-1][:, 0, 0].detach().cpu()

    payload = {
        "config": {"dim": DIM, "n_units": N_UNITS, "block_units": BLOCK_UNITS,
                   "steps": STEPS, "batch": BATCH, "lr": LR},
        "alpha_before_last_unit": [round(float(v), 4) for v in w0.tolist()],
        "alpha_after_last_unit": [round(float(v), 4) for v in w1.tolist()],
        "sources_growth_block": growth[0]["sources_per_unit"],
        "sources_growth_full": growth[1]["sources_per_unit"],
    }
    with open(OUT / "demo_attn_res.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved", OUT / "demo_attn_res.json")


if __name__ == "__main__":
    main()
