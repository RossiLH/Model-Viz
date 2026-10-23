"""Quick test to check which measure functions work."""
import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 6))
_SRC = os.path.join(_ROOT, "src")
_TEST = os.path.join(_ROOT, "tests", "models", "llm", "kimi_k3", "modules")
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_SRC, _TEST, _THIS]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
torch.set_num_threads(2)

import test_moe as _tm
import test_router_qb as _trq
import test_situ_glu as _tsg
import test_norm as _tn
print("All imports OK")

tests = [
    ("tab_forward_contract shapes", lambda: _tm.measure_forward_shape_and_routing(dim=32, batch=3, seq=6, seed=0)),
    ("tab_forward_contract ref", lambda: _tm.measure_eq11_reference_consistency(dim=32, batch=3, seq=4, seed=0)),
    ("tab_forward_contract grad", lambda: _tm.measure_expert_gradient_sparsity(dim=32, batch=1, seq=3, top_k=1, seed=0)),
    ("tab_latent_bottleneck", lambda: _tm.measure_latent_bottleneck_dimensions(dim=32)),
    ("tab_rmsnorm_scale", lambda: _tm.measure_rmsnorm_scale_sensitivity(dim=32, batch=4, seq=8, factor=50.0, seed=0)),
    ("tab_gradient_sparsity", lambda: _tm.measure_expert_gradient_sparsity(dim=16, batch=1, seq=3, top_k=1, seed=0)),
    ("tab_dispatch dense", lambda: _tm.measure_grouped_dispatch_vs_dense(n_experts=6, dim_in=8, hidden=12, dim_out=8, n_tokens=20, seed=0)),
    ("tab_dispatch perm", lambda: _tm.measure_dispatch_permutation_invariance(n_experts=6, dim_in=8, hidden=12, dim_out=8, n_tokens=20, seed=0)),
    ("tab_qb_cutoff", lambda: _trq.measure_cutoff_semantics(dim=16, n_experts=8, top_k=3, n_tokens=32, seed=0)),
    ("tab_qb_bias", lambda: _trq.measure_bias_steering(dim=16, n_experts=8, top_k=2, n_tokens=32, bias_std=2.0, seed=0)),
    ("tab_qb_norm", lambda: _trq.measure_weights_normalization(dim=16, n_experts=8, top_k=3, n_tokens=32)),
    ("tab_qb_balance", lambda: _trq.measure_one_step_balance(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, seed=0)),
    ("tab_qb_mono", lambda: _trq.measure_monotonic_convergence(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, seed=0)),
    ("tab_qb_hist", lambda: _trq.measure_histogram_vs_exact(dim=32, n_experts=12, top_k=3, bins=4000, tokens=8192, seed=0)),
    ("tab_qb_deferred", lambda: _trq.measure_deferred_update(dim=16, n_experts=8, top_k=2, n_tokens=128, seed=0)),
    ("tab_qb_mc", lambda: _trq.measure_mean_centring_invariance(dim=16, n_experts=8, top_k=3, n_tokens=64, shift=7.25, seed=0)),
    ("tab_qb_bmc", lambda: _trq.measure_bias_mean_centred(dim=16, n_experts=8, top_k=2, bins=1000, tokens=2048, seed=0)),
    ("tab_qb_frozen", lambda: _trq.measure_inference_frozen(dim=16, n_experts=8, top_k=2, n_tokens=128, seed=0)),
    ("tab_qb_disabled", lambda: _trq.measure_disabled_router(dim=32, n_experts=16, top_k=2, tokens=4096, seed=0)),
    ("tab_paper_scale", lambda: _trq.measure_paper_scale_performance(dim=64, n_experts=896, top_k=16, bins=1000, tokens=8192, seed=0)),
    ("tab_situ_curve", lambda: _tsg.measure_situ_vs_swiglu_curve(x_range=(-10, 10), beta_gate=4.0, beta_up=25.0)),
    ("tab_situ_bound", lambda: _tsg.measure_situ_bound(beta_gate=4.0, beta_up=25.0)),
    ("tab_situ_origin", lambda: _tsg.measure_origin_match(beta_gate=4.0, beta_up=25.0)),
    ("tab_situ_grad", lambda: _tsg.measure_saturation_gradients(beta_gate=4.0, beta_up=25.0)),
    ("tab_situ_module", lambda: _tsg.measure_module_bound(dim_in=16, hidden=32, beta_gate=4.0, beta_up=25.0, input_scale=10.0)),
    ("tab_situ_explode", lambda: _tsg.measure_swiglu_explodes_vs_situ(dim_in=8, hidden=16, beta_gate=4.0, beta_up=25.0, input_std=40.0, seed=0)),
]

for name, fn in tests:
    try:
        result = fn()
        print(f"  OK  {name}")
    except Exception as e:
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
