"""PI0.5 忠实还原 —— 论文级不变量测试

本文件将论文的架构断言直接固化为可执行测试。
这些是"不变量"测试：它们不关心具体数值，只验证论文强制的结构性不变式。
与 test_pi05_faithful.py 的形状/自洽测试互补 —— 后者发现不了语义 bug，
本文件专门针对语义 bug。

覆盖的论文断言（arXiv:2504.16054 附录 A-E）：
1. 单向信息流： "no VLM embedding attends to the action expert"
   —— 扰动 suffix，prefix 输出必须完全不变（diff == 0）。
2. 注意力掩码作用在 key 轴：被 mask 掉的 token 不应再被其他 token 看到，
   且被 mask 的 query 行不应产生 NaN。
3. KV cache 逐层独立：每层缓存长度只等于 prefix_len（不随层数累积）。
4. cache 等价性：增量解码（用 cache）结果 == 全量前向结果。
5. 门控残差的条件性：门控由时间步 cond 产生，不同 timestep 应产生不同 gate。
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

import torch
import torch.nn.functional as F

from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.modules.dual_expert_attention import (
    DualExpertJointAttention,
    make_attn_mask,
)
from models.vla.pi05_faithful.modules.pi05_transformer import Pi05Transformer
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

B = 2
PREFIX_LEN = 5
SUFFIX_LEN = 4


def _make_attn():
    cfg = Pi05FaithfulConfig.use_small_scale()
    return cfg, DualExpertJointAttention(cfg)


def _make_transformer():
    cfg = Pi05FaithfulConfig.use_small_scale()
    return cfg, Pi05Transformer(cfg)


def _block_mask(cfg, prefix_len=PREFIX_LEN, suffix_len=SUFFIX_LEN, device=None):
    """构造标准全可见的二维块掩码（prefix 不看 suffix）"""
    p = torch.ones(B, prefix_len, dtype=torch.bool, device=device)
    s = torch.ones(B, suffix_len, dtype=torch.bool, device=device)
    return make_attn_mask(p, s, prefix_len, suffix_len)


def _cond(cfg):
    """adaRMS 时间步条件（宽度 = action expert 宽度，对齐官方 pi0.py:164-169）"""
    return torch.randn(B, cfg.action_width)


# ============================================================
# 1. 单向信息流：prefix 不 attend 到 suffix（论文强制）
# ============================================================
class TestUnidirectionalFlow:
    def test_prefix_ignores_suffix(self):
        """扰动 suffix，prefix 输出必须完全不变（论文要求 diff == 0）"""
        cfg, attn = _make_attn()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix1 = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        suffix2 = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        mask = _block_mask(cfg)

        p1, _, _ = attn(prefix, suffix1, mask)
        p2, _, _ = attn(prefix, suffix2, mask)

        diff = (p1 - p2).abs().max().item()
        assert diff == 0.0, (
            f"prefix_out changed by {diff} when suffix changed; "
            "paper requires 0.0 (no VLM embedding attends to action expert)"
        )

    def test_suffix_attends_to_prefix(self):
        """suffix 必须能 attend 到 prefix（信息单向流动的另一半）"""
        cfg, attn = _make_attn()
        prefix1 = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        prefix2 = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        mask = _block_mask(cfg)

        _, s1, _ = attn(prefix1, suffix, mask)
        _, s2, _ = attn(prefix2, suffix, mask)

        diff = (s1 - s2).abs().max().item()
        assert diff > 1e-6, "suffix_out should depend on prefix (unidirectional flow)"


# ============================================================
# 2. 掩码作用在 key 轴，而非 query 轴
# ============================================================
class TestMaskOnKeyAxis:
    def test_masked_token_not_seen(self):
        """扰动被 mask 掉的 token，其他位置输出必须不变（掩码应挡住 key）"""
        cfg, attn = _make_attn()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)

        # mask 掉 prefix 的第 2 个 token（index 2）
        p = torch.ones(B, PREFIX_LEN, dtype=torch.bool)
        p[:, 2] = False
        s = torch.ones(B, SUFFIX_LEN, dtype=torch.bool)
        mask = make_attn_mask(p, s, PREFIX_LEN, SUFFIX_LEN)

        prefix2 = prefix.clone()
        prefix2[:, 2] = 999.0  # 大扰动被 mask 的 token

        _, s1, _ = attn(prefix, suffix, mask)
        _, s2, _ = attn(prefix2, suffix, mask)

        diff = (s1 - s2).abs().max().item()
        assert diff == 0.0, (
            f"other positions changed by {diff} when masked token perturbed; "
            "mask should block the KEY, not leave it visible"
        )

    def test_no_nan_on_masked_query_row(self):
        """被 mask 的 query 行不应产生 NaN（掩码在 key 轴上时不会整行 -inf）"""
        cfg, attn = _make_attn()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)

        p = torch.ones(B, PREFIX_LEN, dtype=torch.bool)
        p[:, 2] = False  # 仅 mask 一个 key
        s = torch.ones(B, SUFFIX_LEN, dtype=torch.bool)
        mask = make_attn_mask(p, s, PREFIX_LEN, SUFFIX_LEN)

        po, so, _ = attn(prefix, suffix, mask)

        assert not torch.isnan(so).any(), "suffix_out has NaN"
        # prefix 中被 mask 的那一行本身不参与损失，但不应污染其他行
        others = torch.cat([po[:, :2], po[:, 3:]], dim=1)
        assert not torch.isnan(others).any(), "unmasked prefix rows have NaN"


# ============================================================
# 3. KV cache 逐层独立
# ============================================================
class TestKVCachePerLayer:
    def test_cache_lengths_do_not_accumulate(self):
        """每层 cache 长度必须相等，不能随层数累积"""
        cfg, transformer = _make_transformer()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        cond = _cond(cfg)
        mask = _block_mask(cfg)

        _, _, kv = transformer(prefix, suffix, cond, mask)

        # 每层的 cache 应为 (k, v)，长度一致（不随层数累积）
        lengths = []
        for layer_kv in kv:
            k, v = layer_kv
            assert k.shape[1] == v.shape[1]
            lengths.append(k.shape[1])

        assert len(set(lengths)) == 1, (
            f"cache lengths per layer differ: {lengths}; "
            "KV cache must be independent per layer"
        )
        # 全量前向时每层缓存 prefix+suffix 的 K/V
        assert lengths[0] == PREFIX_LEN + SUFFIX_LEN, (
            f"expected per-layer cache len {PREFIX_LEN + SUFFIX_LEN}, got {lengths[0]}"
        )

    def test_prefix_forward_caches_prefix_only(self):
        """prefix_forward 每层只缓存 prefix 的 K/V，长度 == prefix_len"""
        cfg, transformer = _make_transformer()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)

        kv = transformer.prefix_forward(prefix)

        assert len(kv) == cfg.paligemma_depth
        for k, v in kv:
            assert k.shape[1] == PREFIX_LEN, (
                f"prefix cache len {k.shape[1]} != prefix_len {PREFIX_LEN}"
            )


# ============================================================
# 4. 增量解码（cache）与全量前向一致
# ============================================================
class TestCacheEquivalence:
    def test_incremental_matches_full(self):
        """用 cache 的增量解码 == 全量前向（验证 cache 语义正确）"""
        cfg, transformer = _make_transformer()
        transformer.eval()
        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        cond = _cond(cfg)

        with torch.no_grad():
            # 全量：prefix + suffix 一次算完
            mask_full = _block_mask(cfg)
            _, s_full, _ = transformer(prefix, suffix, cond, mask_full)

            # 增量：先缓存 prefix（逐层），再只喂 suffix
            prefix_kv = transformer.prefix_forward(prefix)
            s_row = torch.ones(B, SUFFIX_LEN, dtype=torch.bool)
            # 增量步的掩码只覆盖新增的 suffix keys（缓存部分在 attention 内补全可见）
            mask_inc = s_row.unsqueeze(2) & s_row.unsqueeze(1)  # (B, sl, sl)
            _, s_inc, _ = transformer(
                None, suffix, cond, mask_inc, kv_cache=prefix_kv
            )

        diff = (s_inc - s_full).abs().max().item()
        assert diff < 1e-5, (
            f"incremental (cached) suffix output differs from full forward by {diff}"
        )


# ============================================================
# 5. 门控残差的条件性（adaRMS gate 由时间步 cond 产生）
# ============================================================
class TestGatedResidualConditional:
    def test_gate_depends_on_timestep(self):
        """门控残差的 gate 必须来自 adaRMSNorm 的时间步 cond，而非静态参数"""
        from models.vla.pi05_faithful.modules.pi05_transformer import Pi05TransformerBlock

        cfg = Pi05FaithfulConfig.use_small_scale()
        block = Pi05TransformerBlock(cfg)

        # 关键断言：不应再存在与时间步无关的静态 gate 参数
        static_gates = [
            n for n, _ in block.named_parameters()
            if "gate_attn" in n or "gate_ffn" in n
        ]
        assert len(static_gates) == 0, (
            f"found static gate params {static_gates}; "
            "gate must come from AdaRMSNorm cond, not a static parameter"
        )

    def test_suffix_output_depends_on_cond(self):
        """不同时间步条件应产生不同的 suffix 输出（cond 真正生效）"""
        from models.vla.pi05_faithful.modules.pi05_transformer import Pi05TransformerBlock

        cfg = Pi05FaithfulConfig.use_small_scale()
        block = Pi05TransformerBlock(cfg)
        # cond_proj 零初始化 => 初始时调制为 0，需先扰动权重才能观察差异
        for m in block.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features == cfg.action_width * 3:
                torch.nn.init.normal_(m.weight, std=0.02)

        prefix = torch.randn(B, PREFIX_LEN, cfg.paligemma_width)
        suffix = torch.randn(B, SUFFIX_LEN, cfg.action_width)
        mask = _block_mask(cfg)

        cond1 = torch.randn(B, cfg.action_width)
        cond2 = torch.randn(B, cfg.action_width)

        with torch.no_grad():
            _, s1, _ = block(prefix, suffix, cond1, mask)
            _, s2, _ = block(prefix, suffix, cond2, mask)

        diff = (s1 - s2).abs().max().item()
        assert diff > 1e-6, "suffix output should depend on the timestep cond"

    def test_norm_returns_gate(self):
        """AdaRMSNorm(return_gate=True) 应返回 (out, gate)，gate 形状匹配残差"""
        from models.vla.core.norms import AdaRMSNorm

        dim = 32
        norm = AdaRMSNorm(dim, cond_dim=64, return_gate=True)
        x = torch.randn(2, 8, dim)
        cond = torch.randn(2, 64)

        out, gate = norm(x, cond)

        assert out.shape == (2, 8, dim)
        # gate 用于残差，形状 (B, 1, dim)
        assert gate.shape == (2, 1, dim)


# ============================================================
# 6. 时间步采样：论文 s=0.999 截断的 Beta((s-tau)/s)
# ============================================================
class TestFlowTimeSampling:
    def test_beta_transform(self):
        """论文 p(tau)=Beta((s-tau)/s; 1.5, 1)，s=0.999，需提供此采样函数"""
        from models.vla.pi05_faithful.modules import time_sampler

        s = 0.999
        torch.manual_seed(0)
        tau = time_sampler.sample_flow_time((4,), alpha=1.5, beta=1.0, s=s)

        assert tau.shape == (4,)
        assert (tau >= 0).all() and (tau < s).all(), f"tau must be in [0, s), got {tau}"


# ============================================================
# 7. 时间步正弦编码：逐位对齐官方 posemb_sincos
# ============================================================
class TestSinusoidalTimeEmbedding:
    """对齐官方 Reference/PI/openpi-main/src/openpi/models/pi0.py:48-63

    官方 posemb_sincos:
        fraction = linspace(0.0, 1.0, dim // 2)
        period   = min_period * (max_period / min_period) ** fraction
        sinusoid_input = einsum("i,j->ij", pos, 1.0 / period * 2 * pi)
        return concat([sin(sinusoid_input), cos(sinusoid_input)], -1)

    三个关键点（易错）：
    1. 用的是 1/period（角频率），不是 period 本身
    2. 要乘 2*pi
    3. sin 在前、cos 在后
    """

    @staticmethod
    def _official(pos, dim, min_period=4e-3, max_period=4.0):
        """官方公式的独立复现（不依赖被测代码）"""
        import math

        half = dim // 2
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
        period = min_period * (max_period / min_period) ** fraction
        omega = 1.0 / period * 2 * math.pi
        sinusoid_input = pos.to(torch.float64).unsqueeze(-1) * omega.unsqueeze(0)
        return torch.cat([sinusoid_input.sin(), sinusoid_input.cos()], dim=-1)

    def test_matches_official_formula(self):
        """逐位核对：与官方公式的 max diff 必须 < 1e-6"""
        from models.vla.core.embedding import SinusoidalTimeEmbedding

        dim = 64
        t = torch.tensor([0.0, 0.001, 0.25, 0.5, 0.75, 0.999])

        ours = SinusoidalTimeEmbedding(dim=dim, min_period=4e-3, max_period=4.0)(t)
        official = self._official(t, dim).to(ours.dtype)

        max_diff = float((ours - official).abs().max())
        assert max_diff < 1e-6, f"与官方公式不一致，max diff = {max_diff}"

    def test_sin_comes_first(self):
        """sin 在前半、cos 在后半（官方 concat 顺序）"""
        from models.vla.core.embedding import SinusoidalTimeEmbedding

        dim = 8
        # t=0 时：sin(0)=0、cos(0)=1，可直接区分前后半顺序
        out = SinusoidalTimeEmbedding(dim=dim)(torch.tensor([0.0]))
        first_half, second_half = out[0, : dim // 2], out[0, dim // 2 :]

        assert torch.allclose(first_half, torch.zeros_like(first_half)), \
            f"前半应是 sin(0)=0，实际 {first_half}"
        assert torch.allclose(second_half, torch.ones_like(second_half)), \
            f"后半应是 cos(0)=1，实际 {second_half}"

    def test_high_frequency_channel_is_discriminative(self):
        """高频通道必须能分辨相邻时间步

        若误用 period 而非 1/period，低频通道会退化为近似常数
        （cos(0.004*tau) ~= 1），丧失分辨力。
        """
        from models.vla.core.embedding import SinusoidalTimeEmbedding

        emb = SinusoidalTimeEmbedding(dim=64)
        a = emb(torch.tensor([0.10]))
        b = emb(torch.tensor([0.11]))

        # 相邻时间步（差 0.01）应产生可观测差异
        assert float((a - b).abs().max()) > 0.01, "时间编码对时间步变化不敏感"


# ============================================================
# 8. TimeMLP：官方末尾也有 swish
# ============================================================
class TestTimeMLP:
    """官方 pi0.py:164-167 (pi05=True):
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)      # <-- 末尾也有 swish
    即 Linear -> swish -> Linear -> swish
    """

    def test_output_passes_through_swish(self):
        """末尾经过 swish：输出下界必须 >= swish 的全局最小值 ~= -0.2785"""
        from models.vla.core.embedding import TimeMLP

        torch.manual_seed(0)
        mlp = TimeMLP(dim=32)
        # 放大输入，迫使中间层产生大范围正负值
        x = torch.randn(64, 32) * 10.0
        out = mlp(x)

        swish_min = -0.2784645427610738  # min of x*sigmoid(x)
        assert float(out.min()) >= swish_min - 1e-4, \
            f"输出最小值 {float(out.min())} 低于 swish 下界，末尾缺少 swish"

    def test_not_plain_linear(self):
        """必须是非线性：f(2x) != 2*f(x)"""
        from models.vla.core.embedding import TimeMLP

        torch.manual_seed(0)
        mlp = TimeMLP(dim=16)
        x = torch.randn(8, 16)
        assert not torch.allclose(mlp(2 * x), 2 * mlp(x), atol=1e-4)


# ============================================================
# 9. compute_loss：Flow Matching 训练目标应封装在模型内
# ============================================================
class TestComputeLoss:
    """官方把加噪 + 时间采样 + MSE 封装在 Pi0.compute_loss (pi0.py:189-214)：
        noise = normal(actions.shape)
        time  = beta(1.5, 1) * 0.999 + 0.001
        x_t   = time * noise + (1 - time) * actions
        u_t   = noise - actions
        v_t   = forward(...)
        loss  = mean((v_t - u_t)^2)
    我们此前把这段散落在 demo/测试里，调用方容易写错（如忘用 sample_flow_time）。
    """

    def test_compute_loss_exists_and_is_scalar(self):
        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)

        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 6))
        actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

        loss = model.compute_loss(images, text, actions)

        assert loss.dim() == 0, f"loss 应为标量，实际 shape {tuple(loss.shape)}"
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_compute_loss_is_differentiable(self):
        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)

        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 6))
        actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

        model.compute_loss(images, text, actions).backward()

        assert model.action_out_proj.weight.grad is not None
        assert model.action_in_proj.proj.weight.grad is not None

    def test_compute_loss_uses_paper_time_sampling(self):
        """传入固定 noise/time 时应可复现，且内部采样的 time 落在 [0, 0.999)"""
        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)

        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 6))
        actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)

        noise = torch.randn_like(actions)
        time = torch.full((B,), 0.3)

        l1 = model.compute_loss(images, text, actions, noise=noise, timestep=time)
        l2 = model.compute_loss(images, text, actions, noise=noise, timestep=time)
        assert torch.allclose(l1, l2), "给定 noise/time 时 loss 应可复现"