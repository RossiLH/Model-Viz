"""Test all tab functions in gradio_moe_demo."""
import sys, os
sys.path.insert(0, "src")
sys.path.insert(0, "tests/models/llm/kimi_k3/modules")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(2)

import gradio_moe_demo as g

tabs = [
    ("tab_principle_four", lambda: g.tab_principle_four()),
    ("tab_demo_four", lambda: g.tab_demo_four(n_experts=4, top_k=1, n_batches=5, batch=50, hot_experts="0", bins=64)),
    ("tab_forward_contract", lambda: g.tab_forward_contract(dim=16, batch=2, seq_len=4, seed=0)),
    ("tab_latent_bottleneck", lambda: g.tab_latent_bottleneck(dim=32)),
    ("tab_rmsnorm_scale", lambda: g.tab_rmsnorm_scale(dim=16, scale_factor=10, seed=0)),
    ("tab_gradient_sparsity", lambda: g.tab_gradient_sparsity(dim=16, top_k=1, seed=0)),
    ("tab_dispatch_correctness", lambda: g.tab_dispatch_correctness(n_experts=4, dim_in=8, hidden=12, dim_out=8, n_tokens=16, seed=0)),
    ("tab_qb_semantics", lambda: g.tab_qb_semantics(n_experts=8, top_k=3, dim=16, n_tokens=32, seed=0)),
    ("tab_qb_balance", lambda: g.tab_qb_balance(n_experts=8, top_k=2, dim=16, bins=500, tokens=1024, seed=0)),
    ("tab_qb_engineering", lambda: g.tab_qb_engineering(dim=16, n_experts=8, top_k=2, n_tokens=32, seed=0)),
    ("tab_paper_scale", lambda: g.tab_paper_scale(dim=32, n_experts=64, top_k=4, bins=500, tokens=2048, seed=0)),
    ("tab_situ_glu", lambda: g.tab_situ_glu(beta_gate=4.0, beta_up=25.0, x_range=10.0, input_scale=10.0, seed=0)),
]

for name, fn in tabs:
    try:
        fig, text = fn()
        print(f"OK  {name} -> text length={len(text)}")
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception as e:
        import traceback
        print(f"FAIL {name}:")
        traceback.print_exc()
        print()
