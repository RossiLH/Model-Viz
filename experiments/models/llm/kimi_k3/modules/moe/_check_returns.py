"""Check return values of all measure functions."""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests/models/llm/kimi_k3/modules")
import torch
torch.set_num_threads(2)

import test_moe as tm
import test_router_qb as trq
import test_situ_glu as tsg
import test_norm as tn

checks = [
    ("tm.measure_forward_shape_and_routing", lambda: tm.measure_forward_shape_and_routing(dim=16, batch=2, seq=4, seed=0)),
    ("tm.measure_eq11_reference_consistency", lambda: tm.measure_eq11_reference_consistency(dim=16, batch=2, seq=4, seed=0)),
    ("tm.measure_expert_gradient_sparsity", lambda: tm.measure_expert_gradient_sparsity(dim=16, batch=1, seq=3, top_k=1, seed=0)),
    ("tm.measure_latent_bottleneck_dimensions", lambda: tm.measure_latent_bottleneck_dimensions(dim=32)),
    ("tm.measure_rmsnorm_scale_sensitivity", lambda: tm.measure_rmsnorm_scale_sensitivity(dim=16, batch=2, seq=4, factor=50.0, seed=0)),
    ("tm.measure_grouped_dispatch_vs_dense", lambda: tm.measure_grouped_dispatch_vs_dense(n_experts=4, dim_in=8, hidden=12, dim_out=8, n_tokens=16, seed=0)),
    ("tm.measure_dispatch_permutation_invariance", lambda: tm.measure_dispatch_permutation_invariance(n_experts=4, dim_in=8, hidden=12, dim_out=8, n_tokens=16, seed=0)),
    ("trq.measure_cutoff_semantics", lambda: trq.measure_cutoff_semantics(dim=16, n_experts=8, top_k=3, n_tokens=32, seed=0)),
    ("trq.measure_bias_steering", lambda: trq.measure_bias_steering(dim=16, n_experts=8, top_k=2, n_tokens=32, bias_std=2.0, seed=0)),
    ("trq.measure_weights_normalization", lambda: trq.measure_weights_normalization(dim=16, n_experts=8, top_k=3, n_tokens=32)),
    ("trq.measure_one_step_balance", lambda: trq.measure_one_step_balance(dim=16, n_experts=8, top_k=2, bins=500, tokens=1024, seed=0)),
    ("trq.measure_monotonic_convergence", lambda: trq.measure_monotonic_convergence(dim=16, n_experts=8, top_k=2, bins=500, tokens=1024, seed=0)),
    ("trq.measure_histogram_vs_exact", lambda: trq.measure_histogram_vs_exact(dim=16, n_experts=8, top_k=2, bins=500, tokens=1024, seed=0)),
    ("trq.measure_deferred_update", lambda: trq.measure_deferred_update(dim=16, n_experts=8, top_k=2, n_tokens=32, seed=0)),
    ("trq.measure_mean_centring_invariance", lambda: trq.measure_mean_centring_invariance(dim=16, n_experts=8, top_k=2, n_tokens=32, shift=7.25, seed=0)),
    ("trq.measure_bias_mean_centred", lambda: trq.measure_bias_mean_centred(dim=16, n_experts=8, top_k=2, bins=500, tokens=1024, seed=0)),
    ("trq.measure_inference_frozen", lambda: trq.measure_inference_frozen(dim=16, n_experts=8, top_k=2, n_tokens=32, seed=0)),
    ("trq.measure_disabled_router", lambda: trq.measure_disabled_router(dim=16, n_experts=8, top_k=2, tokens=1024, seed=0)),
    ("trq.measure_paper_scale_performance", lambda: trq.measure_paper_scale_performance(dim=32, n_experts=64, top_k=4, bins=500, tokens=2048, seed=0)),
    ("tsg.measure_situ_vs_swiglu_curve", lambda: tsg.measure_situ_vs_swiglu_curve(x_range=(-10, 10), beta_gate=4.0, beta_up=25.0)),
    ("tsg.measure_situ_bound", lambda: tsg.measure_situ_bound(beta_gate=4.0, beta_up=25.0)),
    ("tsg.measure_origin_match", lambda: tsg.measure_origin_match(beta_gate=4.0, beta_up=25.0)),
    ("tsg.measure_saturation_gradients", lambda: tsg.measure_saturation_gradients(beta_gate=4.0, beta_up=25.0)),
    ("tsg.measure_module_bound", lambda: tsg.measure_module_bound(dim_in=16, hidden=32, beta_gate=4.0, beta_up=25.0, input_scale=10.0)),
    ("tsg.measure_swiglu_explodes_vs_situ", lambda: tsg.measure_swiglu_explodes_vs_situ(dim_in=8, hidden=16, beta_gate=4.0, beta_up=25.0, input_std=40.0, seed=0)),
]

for name, fn in checks:
    try:
        result = fn()
        keys = list(result.keys()) if isinstance(result, dict) else str(type(result))
        print(f"OK  {name} -> keys: {keys}")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {e}")
