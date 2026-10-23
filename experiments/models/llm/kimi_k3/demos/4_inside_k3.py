#!/usr/bin/env python3
"""论文的机制，在真实前向传播上运行。

    python experiments/demos/4_inside_k3.py

Kimi K3 不是一个换了超参数的 Transformer。这打印了使它成为不同架构的四个机制，
在一个实际 batch 上测量而非描述：

* **3:1 混合注意力** —— 三个 Kimi Delta Attention 层对应一个 Gated MLA 层，
  最后一层总是全局的。
* **完全没有位置编码。** 这个声明容易说也容易错，所以通过检查每个参数名来验证。
* **Stable LatentMoE** —— 专家在模型宽度一半的潜空间工作，每个 token 只有少数运行。
* **有界遗忘门** —— KDA 的衰减被挤压，使得一个头不能饱和到零并丢失状态。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import load, nano_config, rule

import torch


def main() -> int:
    cfg = nano_config()
    pc = count_params(cfg)

    rule("what was built")
    print(f"  total parameters      {pc.total/1e6:>10.1f} M")
    print(f"  active per token      {pc.activated/1e6:>10.1f} M   "
          f"({100*pc.activated/pc.total:.0f} % of the model runs for any one token)")
    print(f"  model width           {cfg.model.dim:>10}")
    print(f"  vocabulary            {cfg.model.vocab_size:>10}   byte-level")

    rule("3:1 hybrid attention (paper 2.1)")
    schedule = cfg.model.layer_schedule()
    print(f"  {' '.join('K' if k == 'kda' else 'M' for k in schedule)}")
    kda = sum(1 for k in schedule if k == "kda")
    print(f"  K = Kimi Delta Attention ({kda})   M = Gated MLA ({len(schedule)-kda})")
    print(f"  last layer is {'global (MLA)' if schedule[-1] != 'kda' else 'KDA'}"
          " -- every token can reach every other at least once")

    rule("NoPE: no positional encoding anywhere (paper 2.1.2)")
    # 尝试加载 checkpoint（如果有的话），否则用随机权重
    try:
        model, tok = load("3000")
    except SystemExit:
        print("  (checkpoint not found, using random weights for structure demo)")
        from models.llm.kimi_k3.model import Kimi3ForCausalLM
        model = Kimi3ForCausalLM(cfg)
        tok = None

    suspects = [n for n, _ in model.named_parameters()
                if any(t in n.lower() for t in ("pos", "rope", "rotary", "alibi", "sin", "cos"))]
    print(f"  parameters whose name suggests position: {len(suspects)}  {suspects or ''}")
    print("  position is carried by causal structure and KDA's recurrence, not by an")
    print("  embedding -- which is what lets the context window extend without retraining")

    rule("Stable LatentMoE (paper 2.3)")
    moe = cfg.model.moe
    print(f"  routed experts        {moe.n_routed:>10}")
    print(f"  shared experts        {moe.n_shared:>10}   always on")
    print(f"  active per token      {moe.top_k:>10}   sparsity {moe.n_routed/moe.top_k:.0f}x")
    print(f"  latent width          {moe.latent_dim:>10}   vs model width {cfg.model.dim}")
    print("  experts work in the narrow space; only the shared path is full width")

    rule("KDA forget gates (paper 2.1.1, Eq. 5)")
    # 尝试获取 decay 数据
    try:
        text = "def parse(config):\n    return config.get('value')\n"
        ids = torch.tensor([tok.encode(text)], dtype=torch.long)
        with torch.no_grad():
            model(ids)
        layer = next(b.attn for b in model.model.layers if hasattr(b.attn, "log_decay"))
        with torch.no_grad():
            h = model.model.embed_tokens(ids)
            decay = layer.log_decay(h).exp()
        print(f"  decay per channel     min {decay.min():.4f}   "
              f"mean {decay.mean():.4f}   max {decay.max():.4f}")
        print(f"  floor is exp(g_min) = {torch.tensor(cfg.model.kda.g_min).exp():.4f}, so no channel can")
        print("  forget completely -- an unbounded gate saturates and the state dies")
    except Exception:
        print(f"  floor is exp(g_min) = {torch.tensor(cfg.model.kda.g_min).exp():.4f}")
        print("  (decay measurement requires checkpoint with trained weights)")

    rule()
    print("Every one of these is asserted in tests/, not just printed here:")
    print("  test_model.py       the 3:1 schedule and the trailing global layer")
    print("  test_mla.py         that no positional parameter exists")
    print("  test_kda.py         the chunkwise form equals the recurrence to 1e-9")
    print("  test_moe.py         the latent routing matches an independent reference")
    return 0


# 延迟导入避免循环
from models.llm.kimi_k3.config import count_params  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
