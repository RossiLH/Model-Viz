# -*- coding: utf-8 -*-
"""SiTU-GLU 有界激活的单元测试（对应 modules/situ_glu.py，论文 §2.3.2 / Eq. 12）。

背景回顾
========
SwiGLU（``silu(x) * x``）的 gate 与 up 两个因子都是**无界**的：gate/up 投影坐标
偶尔同时落在很大的值上时，乘积会产生激活离群点（outlier），在低精度（FP8/BF16）
下溢出或严重损失精度。SiTU-GLU 给两个因子各套一个光滑软截断：

    SiTU-GLU(g, u) = [beta_g·tanh(g/beta_g)·sigmoid(g)] * [beta_u·tanh(u/beta_u)]

        gate 分支 = softcap(g) * sigmoid(g)   （有界 Swish：sigmoid 压负响应，
                                               softcap 切断大正激活）
        up   分支 = softcap(u)                 （有界线性因子）

核心性质（论文附录 B，本文件逐条验证）：
    1. **有界**：``|SiTU-GLU(x)| <= beta_g * beta_u``。论文取 beta_g=4、beta_u=25，
       于是上界 = 100，特意选得小于 FP8 E4M3 上限 448，保证低精度激活不溢出。
    2. **原点近似 SwiGLU**：``tanh(z) = z - z^3/3 + ...``，softcap 到一阶与恒等
       相同，所以在原点附近 SiTU-GLU 与 SwiGLU 形状一致。
    3. **光滑可导**：相比硬 clamp，softcap 在饱和区仍有非零梯度（tanh' > 0），
       不会掐断训练梯度。
    4. **负尾部归零**：gate 分支的 sigmoid 让负输入快速趋近 0，保留 Swish 的稀疏性。

设计说明：测试尽量用**标量函数**（``scalar_situ``）直接驱动 ``situ_glu`` 的纯函数
接口，避免模块层（projection 权重）的干扰，聚焦数学性质本身；再用两个模块级测试
（``SiTUGLU``/``SwiGLU``）验证"有界 vs 无界"在同一参数结构下的消融对比。
"""
import math

import pytest
import torch
import torch.nn.functional as F

from models.llm.kimi_k3.modules.situ_glu import SiTUGLU, SwiGLU, situ_glu, softcap

# 论文默认参数：gate 截断 beta=4，up 截断 beta=25，二者乘积恰为上界 100。
# 100 < FP8 E4M3 表示上限(448)，这正是论文选择该数值的目的（FP8 激活不溢出）。
BETA_GATE = 4.0
BETA_UP = 25.0
BOUND = BETA_GATE * BETA_UP


def scalar_situ(x: torch.Tensor) -> torch.Tensor:
    """标量 SiTU-GLU：把同一输入同时当作 gate 与 up 传入（``SiTU-GLU(x, x)``）。

    用途：用纯函数接口快速扫描 SiTU-GLU 在整个数值轴上的行为，是多数性质测试的操作面。
    （gate==up 虽是退化配置，但用于验证"输入幅度→输出"的代数性质完全足够。）
    """
    return situ_glu(x, x, BETA_GATE, BETA_UP)


def scalar_swiglu(x: torch.Tensor) -> torch.Tensor:
    """标量 SwiGLU 基线（``silu(x) * x``），作为与 SiTU-GLU 对比的无界参照。"""
    return F.silu(x) * x


# ---------------------------------------------------------------------------
# 可复用测量函数（供 gradio demo 等外部调用，返回 dict，不做 assert）
# ---------------------------------------------------------------------------

def measure_situ_bound(beta_gate=BETA_GATE, beta_up=BETA_UP):
    """测量 SiTU-GLU 在大正输入下的输出上界。

    返回 dict:
      - bound: float（理论上界 beta_g*beta_u）
      - actual_max: float（x=1e3 时的实际输出）
      - rel_error: float（相对误差 |actual-bound|/bound）
    """
    bound = beta_gate * beta_up
    y = situ_glu(torch.tensor(1e3), torch.tensor(1e3), beta_gate, beta_up)
    actual = float(y.item())
    return {
        "bound": bound,
        "actual_max": actual,
        "rel_error": abs(actual - bound) / bound,
    }


def measure_situ_vs_swiglu_curve(n=401, x_range=(-10, 10), beta_gate=BETA_GATE, beta_up=BETA_UP):
    """测量 SiTU-GLU 与 SwiGLU 在给定数值区间上的曲线。

    返回 dict:
      - x: list[float]
      - situ: list[float]
      - swiglu: list[float]
      - bound: float
    """
    x = torch.linspace(*x_range, n)
    s = situ_glu(x, x, beta_gate, beta_up)
    w = F.silu(x) * x
    return {
        "x": x.tolist(),
        "situ": s.tolist(),
        "swiglu": w.tolist(),
        "bound": beta_gate * beta_up,
    }


def measure_origin_match(n=401, x_range=(-0.1, 0.1), beta_gate=BETA_GATE, beta_up=BETA_UP):
    """测量原点附近 SiTU-GLU 与 SwiGLU 的相对误差。"""
    x = torch.linspace(*x_range, n)
    situ, swi = situ_glu(x, x, beta_gate, beta_up), F.silu(x) * x
    denom = swi.abs().clamp_min(1e-8)
    max_rel = float(((situ - swi).abs() / denom).max().item())
    return {"max_rel_error": max_rel}


def measure_saturation_gradients(beta_gate=BETA_GATE, beta_up=BETA_UP):
    """测量饱和区 softcap 梯度 vs 硬 clamp 梯度。

    返回 dict:
      - z_vals: list[float]
      - softcap_grad: list[float]
      - hard_clamp_grad: list[float]
      - situ_has_finite_grad: bool
    """
    z_vals = torch.tensor([20.0, 50.0, 100.0], requires_grad=True)
    situ_glu(z_vals, z_vals, beta_gate, beta_up).sum().backward()
    soft_grad = z_vals.grad.detach().clone()

    z2 = torch.tensor([20.0, 50.0, 100.0], requires_grad=True)
    (torch.clamp(z2, max=beta_gate) * torch.sigmoid(z2)).sum().backward()
    hard_grad = z2.grad.detach().clone()

    return {
        "z_vals": z_vals.detach().tolist(),
        "softcap_grad": soft_grad.tolist(),
        "hard_clamp_grad": hard_grad.tolist(),
        "situ_has_finite_grad": bool(torch.isfinite(soft_grad).all().item()),
        "soft_better_than_hard": bool((soft_grad > hard_grad).all().item()),
    }


def measure_module_bound(dim_in=16, hidden=32, beta_gate=BETA_GATE, beta_up=BETA_UP,
                         param_std=3.0, input_scale=10.0, batch=4, seq=7):
    """测量 SiTUGLU 模块在极端参数和输入下的输出有界性。"""
    ffn = SiTUGLU(dim_in, hidden, beta_gate=beta_gate, beta_up=beta_up)
    with torch.no_grad():
        for p in ffn.parameters():
            p.normal_(std=param_std)
    out = ffn(torch.randn(batch, seq, dim_in) * input_scale)
    return {
        "shape": tuple(out.shape),
        "all_finite": bool(torch.isfinite(out).all().item()),
        "max_abs": float(out.abs().max().item()),
    }


def measure_swiglu_explodes_vs_situ(dim_in=8, hidden=16, beta_gate=BETA_GATE, beta_up=BETA_UP,
                                   input_std=40.0, n_tokens=64, seed=0):
    """测量同权重同输入下 SwiGLU 离群 vs SiTU-GLU 守界。"""
    torch.manual_seed(seed)
    x = torch.randn(n_tokens, dim_in) * input_std
    situ = SiTUGLU(dim_in, hidden, beta_gate=beta_gate, beta_up=beta_up)
    swi = SwiGLU(dim_in, hidden)
    with torch.no_grad():
        for a, b in ((situ.gate_proj, swi.gate_proj), (situ.up_proj, swi.up_proj)):
            b.weight.copy_(a.weight)
        swi.down_proj.weight.copy_(situ.down_proj.weight)
    with torch.no_grad():
        h_situ = situ_glu(situ.gate_proj(x), situ.up_proj(x), beta_gate, beta_up)
        h_swi = F.silu(swi.gate_proj(x)) * swi.up_proj(x)
    return {
        "situ_max_abs": float(h_situ.abs().max().item()),
        "swiglu_max_abs": float(h_swi.abs().max().item()),
        "ratio_swiglu_over_situ": float(h_swi.abs().max().item() / h_situ.abs().max().item()),
        "bound": beta_gate * beta_up,
    }


def test_output_is_bounded() -> None:
    """测试目的：验证 SiTU-GLU 的输出在任意输入幅度下永久有界。

    具体功能：
    - 在 [-1e4, 1e4] 上取 20 万点，外加两个极端值 ±1e30（模拟 FP8 场景下的
      巨大离群输入）；
    - 断言输出绝对值恒 <= BOUND=100（+1e-4 容忍 float32 累积误差）；
    - 断言输出全部有限（isfinite），即任何输入都不会产生 inf/nan。

    为什么重要：这正是 SiTU-GLU 相比 SwiGLU 的**核心卖点**——只有输出有界，
    FP8 低精度推理才不会因离群点溢出。若此测试失败，说明 softcap 没有真正尽责。
    """
    x = torch.cat(
        [
            torch.linspace(-1e4, 1e4, 200_001),
            torch.tensor([float(1e30), float(-1e30)]),
        ]
    )
    y = scalar_situ(x)
    assert torch.all(y.abs() <= BOUND + 1e-4)
    assert torch.isfinite(y).all()


def test_approaches_the_bound_for_large_positive_inputs() -> None:
    """测试目的：验证大正输入下输出**渐近逼近**上界 BOUND。

    具体功能：对 x=1e3（早已进入饱和区）求值，输出应 ≈ beta_g·beta_u = 100
    （相对误差 1e-6）。softcap 的数学极限是 ``|softcap(z)| -> beta``，两个因子
    都饱和时乘积 -> beta_g·beta_u，即 BOUND。

    为什么重要：确认上界不仅是"不超过"，而且是**可达/可逼近的**，说明截断是把
    极大值平滑拉回有限区间而不是单纯掐断。
    """
    assert float(scalar_situ(torch.tensor(1e3))) == pytest.approx(BOUND, rel=1e-6)


def test_swiglu_is_unbounded_by_contrast() -> None:
    """测试目的：对照实验，证明同规模输入下 SwiGLU 确实无界。

    具体功能：``silu(1e3) * 1e3 ≈ 1e3``，远大于 100。把 SwiGLU 的爆炸行为
    显式钉死为 >1e5，与上面 SiTU-GLU 的 100 形成鲜明对比。

    为什么重要：为"SiTU-GLU 有界是有意设计、而非巧合"提供反例证据；同时说明
    为什么需要 SiTU-GLU——同样的数值在 SwiGLU 里就是离群点。
    """
    assert float(scalar_swiglu(torch.tensor(1e3))) > 1e5


def test_matches_swiglu_near_the_origin() -> None:
    """测试目的：验证原点附近 SiTU-GLU 与 SwiGLU 形状一致。

    具体功能：在 [-0.1, 0.1] 上逐点比较两者，用**相对误差**（除以 SwiGLU 幅值，
    clamp_min 防除零）度量差异，断言最大相对误差 < 1e-3。

    为什么重要：SiTU-GLU 应在激活进入饱和区之前与应被它替换的 SwiGLU 基本
    行为一致——这样"换激活函数"才不改变模型正常区的动力学，只修掉离群点。
    """
    x = torch.linspace(-0.1, 0.1, 401)
    situ, swi = scalar_situ(x), scalar_swiglu(x)
    denom = swi.abs().clamp_min(1e-8)
    assert float(((situ - swi).abs() / denom).max()) < 1e-3


def test_softcap_expansion_error_scales_as_predicted() -> None:
    """测试目的：用泰勒展开的一阶理论预言校验 softcap 的小量误差。

    具体功能：``softcap(z,beta) = beta·tanh(z/beta) ≈ z - z^3/(3·beta^2)``，
    故相对误差 ≈ ``z^2/(3·beta^2)``。对 z = 0.4/0.2/0.1（随 beta_gate 缩放），
    断言实测相对误差与预言值吻合（相对容差 5%）。

    为什么重要：这同时验证了两件事——(a) softcap 的实现就是 tanh 软截断本身
    （没引入奇怪的额外项）；(b) 误差随 z 缩小按 z^2 快速消失，即原点附近
    "近似恒等"是渐近成立的，误差衰减率符合理论。
    """
    for z in (0.4, 0.2, 0.1):
        err = abs(float(softcap(torch.tensor(z), BETA_GATE)) - z) / z
        predicted = z**2 / (3 * BETA_GATE**2)
        assert err == pytest.approx(predicted, rel=0.05)


def test_recovers_swiglu_as_caps_grow() -> None:
    """测试目的：验证极限情形——beta 足够大时 SiTU-GLU 退化为 SwiGLU。

    具体功能：把两个截断参数都设为 1e6（远大于输入范围 [-5,5]），此时 softcap
    在输入范围上近似恒等（``beta·tanh(z/beta) ≈ z``），gate 分支只剩 sigmoid，
    整体应 ≈ ``sigmoid(x)·x`` = ``silu(x)`` 的正分支；断言逐点 allclose(SiTU,
    SwiGLU)。

    为什么重要：这是一个"连续性"测试——证明 SiTU-GLU 与 SwiGLU 属于**同一个
    参数化族**，beta→∞ 是同族内的一族退化解，而非两个不相关的函数。这让"有界性
    只是加了约束"的论述更严谨。
    """
    x = torch.linspace(-5, 5, 201)
    loose = situ_glu(x, x, 1e6, 1e6)
    assert torch.allclose(loose, scalar_swiglu(x), atol=1e-4)


def test_negative_tail_vanishes_like_swiglu() -> None:
    """测试目的：验证负区间上 SiTU-GLU 保持 Swish 的"零化"特性。

    具体功能：在 [-10, -3] 上：
    - 输出单调递增（``situ.diff() > 0``，负区间随 x 增大输出增大）；
    - 最负端输出 < 0.01（非常接近 0，稀疏性）；
    - 输出始终 > 0（sigmoid + softcap 正号保证）；
    - 输出 <= 同点 SwiGLU（softcap 对负向因子的缩放使其比 SwiGLU 更保守）。

    为什么重要：SiTU-GLU 不止是有界，还必须**保留 Swish 的稀疏激活**（负输入
    被压到接近 0），否则会引入大量非零小激活、破坏 MoE 的稀疏特性。
    """
    x = torch.linspace(-10, -3, 100)
    situ, swi = scalar_situ(x), scalar_swiglu(x)
    assert torch.all(situ.diff() > 0)
    assert float(situ[0]) < 0.01
    assert torch.all(situ > 0)
    assert torch.all(situ <= swi)


def test_gradients_survive_saturation() -> None:
    """测试目的：验证饱和区仍有非零梯度（"光滑可导"性质），并优于硬 clamp。

    具体功能：
    - 对 x=[20, 50]（gate 已远超 beta_gate=4 的饱和点）反向传播，断言梯度非空、
      全部 >0 且有限——SiTU-GLU 即使在饱和区也在"继续学习"；
    - 对照组：用 ``clamp(x, max=beta_gate) * sigmoid(x)`` 模拟硬截断版激活做同样
      反向传播，断言其在 x=50 处的梯度**小于** SiTU-GLU 的梯度（硬 clamp 在
      截断区的梯度恒为 0，softcap 则保留 tanh' > 0 的小梯度）。

    为什么重要：硬 clamp 上线虽然也"有界"，但在饱和区梯度为 0，训练时这些通道
    直接退化；softcap 用光滑截断换取"有界 + 可导"兼得，本测试把这个差异钉死。
    """
    x = torch.tensor([20.0, 50.0], requires_grad=True)
    scalar_situ(x).sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad.abs() > 0)
    assert torch.isfinite(x.grad).all()

    hard = torch.tensor([20.0, 50.0], requires_grad=True)
    (torch.clamp(hard, max=BETA_GATE) * torch.sigmoid(hard)).sum().backward()
    assert hard.grad is not None
    assert float(hard.grad[1]) < float(x.grad[1])


def test_module_shapes_and_bound() -> None:
    """测试目的：模块层（``SiTUGLU``）的形状约定与极端权重下仍保持有界。

    具体功能：构造 dim_in=16 -> hidden=32 -> dim_out=16 的 SiTUGLU，把所有参数
    用 std=3.0 的正态随机覆盖（制造比正常训练更大的权重），输入放大 10 倍；
    断言：输出形状 (4,7,16) 正确、输出全部有限。

    为什么重要：验证"有界性"不仅是数学函数的性质，在**带投影权重的模块**里、
    即使权重被刻意放大（模拟训练中的病态初始化）输出依然不会爆炸——这是 SiTU-GLU
    能安全替换 MoE 专家内 FFN 的前提。
    """
    ffn = SiTUGLU(16, 32, beta_gate=BETA_GATE, beta_up=BETA_UP)
    with torch.no_grad():
        for p in ffn.parameters():
            p.normal_(std=3.0)
    out = ffn(torch.randn(4, 7, 16) * 10)
    assert out.shape == (4, 7, 16)
    assert torch.isfinite(out).all()


def test_module_hidden_activation_respects_bound() -> None:
    """测试目的：验证隐藏层激活（gate/up 的 SiTU-GLU 输出）在差初始化下守界。

    具体功能：把 gate_proj / up_proj 权重用 std=50 正态覆盖（极端的病态初始化），
    输入也放大到 ±50，直接调用纯函数 ``situ_glu`` 计算隐藏激活，断言其幅值上限
    仍 <= BOUND + 1e-3。

    为什么重要：这道测试是"最深一层"的边界防御——即使投影把 token 推到极大
    坐标系，**SiTU-GLU 的数学上界依然生效**，保证后面接的 down_proj 不会收到
    爆炸输入。这也是 FP8 量化安全的关键（见模块 docstring）。
    """
    ffn = SiTUGLU(8, 16, beta_gate=BETA_GATE, beta_up=BETA_UP)
    with torch.no_grad():
        ffn.gate_proj.weight.normal_(std=50.0)
        ffn.up_proj.weight.normal_(std=50.0)
    x = torch.randn(3, 5, 8) * 50
    with torch.no_grad():
        hidden = situ_glu(ffn.gate_proj(x), ffn.up_proj(x), BETA_GATE, BETA_UP)
    assert float(hidden.abs().max()) <= BOUND + 1e-3


def test_swiglu_baseline_explodes_where_situ_does_not() -> None:
    """测试目的：在**同一参数结构**上做"有界 vs 无界"的公平消融。

    具体功能：构造参数完全相同的 SiTUGLU 与 SwiGLU，把 SiTUGLU 的三组权重逐一
    拷贝给 SwiGLU（消除初始化差异），用大输入 x（~N(0,40²)）分别算两边的隐藏
    激活，断言：SiTU 侧幅值 <= BOUND=100，而 SwiGLU 侧幅值 > 10·BOUND=1000。

    为什么重要：这是该项目"before/after 对比"理念的直接体现——**同样的权重、
    同样的输入**，只因激活不同，SwiGLU 就产生 >10 倍离群点而 SiTU-GLU 守界。
    说明有界性是激活函数带来的，而非权重巧合。
    """
    torch.manual_seed(0)
    x = torch.randn(64, 8) * 40
    situ = SiTUGLU(8, 16, beta_gate=BETA_GATE, beta_up=BETA_UP)
    swi = SwiGLU(8, 16)
    with torch.no_grad():
        for a, b in ((situ.gate_proj, swi.gate_proj), (situ.up_proj, swi.up_proj)):
            b.weight.copy_(a.weight)
        swi.down_proj.weight.copy_(situ.down_proj.weight)
    with torch.no_grad():
        h_situ = situ_glu(situ.gate_proj(x), situ.up_proj(x), BETA_GATE, BETA_UP)
        h_swi = F.silu(swi.gate_proj(x)) * swi.up_proj(x)
    assert float(h_situ.abs().max()) <= BOUND
    assert float(h_swi.abs().max()) > 10 * BOUND


def test_config_defaults_give_the_papers_bound() -> None:
    """测试目的：验证默认配置文件给出的 beta 乘积恰为论文的上界 100。

    具体功能：实例化默认 ``MoEConfig``，断言 ``situ_beta_gate * situ_beta_up == 100.0``，
    且 beta_gate≈4.0、beta_up≈25.0。

    为什么重要：整个测试文件的常量（BETA_GATE=4 / BETA_UP=25 / BOUND=100）必须是
    **配置的真实默认值**，否则测试与产品配置脱节。一旦论文数值被改，本测试会立即
    报警，确保"有界性证明"始终作用在真实部署参数上。
    """
    from models.llm.kimi_k3.config import MoEConfig

    cfg = MoEConfig()
    assert cfg.situ_beta_gate * cfg.situ_beta_up == 100.0
    assert math.isclose(cfg.situ_beta_gate, 4.0)
    assert math.isclose(cfg.situ_beta_up, 25.0)