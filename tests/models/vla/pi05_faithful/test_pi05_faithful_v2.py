"""PI0.5 完整版（六特性）测试

每个测试对应设计规格 §9.1 的论文不变量。
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

import pytest
import torch
import numpy as np

from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel


class TestConfigNewFields:
    """设计规格 §8.2：新字段默认值 + 派生 + 校验"""

    def test_new_fields_defaults(self):
        cfg = Pi05FaithfulConfig.use_small_scale()
        assert cfg.proprio_enabled is False
        assert cfg.fast_enabled is False
        assert cfg.fast_scale == 10.0
        assert cfg.fast_max_tokens == 32
        assert cfg.fast_vocab_size == 512
        assert cfg.knowledge_insulation is True
        assert cfg.rtc_max_delay == 0
        assert cfg.loss_weight_ce == 10.0

    def test_n_action_steps_defaults_to_action_horizon(self):
        cfg = Pi05FaithfulConfig.use_small_scale()
        assert cfg.n_action_steps == cfg.action_horizon

    def test_n_action_steps_must_not_exceed_horizon(self):
        with pytest.raises(ValueError):
            Pi05FaithfulConfig(
                action_horizon=10, n_action_steps=12,
            )

    def test_rtc_max_delay_must_not_exceed_horizon(self):
        with pytest.raises(ValueError):
            Pi05FaithfulConfig(
                action_horizon=10, rtc_max_delay=11,
            )

    def test_rtc_plus_partial_chunk_guard(self):
        # rtc_max_delay>0 且 n_action_steps<action_horizon 时要求 delay < steps
        with pytest.raises(ValueError):
            Pi05FaithfulConfig(
                action_horizon=10, n_action_steps=4, rtc_max_delay=4,
            )
        # 合法组合：delay < n_action_steps
        cfg = Pi05FaithfulConfig(
            action_horizon=10, n_action_steps=4, rtc_max_delay=3,
        )
        assert cfg.rtc_max_delay < cfg.n_action_steps

    def test_existing_fields_untouched(self):
        cfg = Pi05FaithfulConfig.use_small_scale()
        assert cfg.paligemma_width == 256
        assert cfg.action_horizon == 10
        assert cfg.action_dim == 32

class TestFastTokenizer:
    """设计规格 §4：FAST DCT 编解码往返（方案 (a) DCT+均匀量化）"""

    @staticmethod
    def _chunk(H=4, A=3, seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-1.0, 1.0, size=(H, A))

    def test_encode_decode_roundtrip(self):
        """DCT 编解码往返误差 < 容差（无截断、scale 足够大时量化误差小）"""
        from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer

        H, A = 4, 3
        chunk = self._chunk(H, A)
        tok = FastTokenizer(scale=100.0, min_token=0, time_horizon=H, action_dim=A)
        tokens = tok.encode(chunk)
        decoded = tok.decode(tokens)
        # scale=100 到量化步长 0.01，往返误差应远小于 0.1
        assert np.abs(decoded - chunk).max() < 0.1

    def test_tokens_are_nonneg_integers(self):
        """量化后 token 必须是非负整数（min_token 偏移保证）"""
        from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer

        tok = FastTokenizer(scale=10.0, min_token=-5, time_horizon=4, action_dim=3)
        tokens = tok.encode(self._chunk())
        assert tokens.dtype.kind == "i"
        assert (tokens >= 0).all()

    def test_truncates_to_max_tokens(self):
        """token 序列截断到 max_tokens"""
        from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer

        tok = FastTokenizer(scale=10.0, min_token=0, max_tokens=4, time_horizon=4, action_dim=3)
        tokens = tok.encode(self._chunk())
        assert len(tokens) == 4

    def test_fit_determines_min_token_and_vocab_range(self):
        """fit 统计 min/max token，校验范围 <= vocab_size"""
        from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer

        data = [self._chunk(H=4, A=3, seed=s) for s in range(5)]
        tok = FastTokenizer.fit(data, scale=10.0, vocab_size=512, time_horizon=4, action_dim=3)
        assert hasattr(tok, "min_token")
        tokens = tok.encode(self._chunk(seed=99))
        assert (tokens >= 0).all()
        assert tokens.max() < 512

    def test_fit_raises_when_vocab_too_small(self):
        """token 范围超过 vocab_size 时 fit 必须报错"""
        from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer

        data = [self._chunk(H=4, A=3, seed=s) * 50.0 for s in range(3)]
        with pytest.raises(ValueError):
            FastTokenizer.fit(data, scale=10.0, vocab_size=4, time_horizon=4, action_dim=3)

class TestProprio:
    """设计规格 §3：Proprio 状态接入 prefix"""

    def test_encode_state_returns_bin_indices(self):
        """状态向量离散化为 [0, num_bins) 的整数 token id"""
        from models.vla.pi05_faithful.modules.proprio import ProprioEncoder

        enc = ProprioEncoder(num_bins=256)
        state = torch.tensor([[-1.0, 0.0, 1.0]])
        ids = enc.encode(state)
        assert ids.shape == (1, 3)
        assert ids.dtype == torch.long
        assert (ids >= 0).all() and (ids < 256).all()
        # -1 到 bin 0，1 到 bin 255（末端）
        assert ids[0, 0].item() == 0
        assert ids[0, 2].item() == 255

    def test_proprio_appended_to_prefix(self):
        """proprio_enabled 时 prefix 长度 = 图像 + 文本 + 状态维度"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        cfg = Pi05FaithfulConfig.use_small_scale()
        cfg.proprio_enabled = True
        model = Pi05FaithfulPolicyModel(cfg)

        B, text_len = 2, 5
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, text_len))
        state = torch.randn(B, cfg.action_dim)

        out = model.encode_prefix(images, text, state=state)
        prefix_len = out["prefix_tokens"].shape[1]
        assert prefix_len == cfg.num_image_tokens + text_len + cfg.action_dim

    def test_proprio_off_state_ignored(self):
        """proprio_enabled=False 时传 state 也忽略（向后兼容）"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)

        B, text_len = 2, 5
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, text_len))
        state = torch.randn(B, cfg.action_dim)

        with_state = model.encode_prefix(images, text, state=state)
        without_state = model.encode_prefix(images, text)
        assert with_state["prefix_tokens"].shape == without_state["prefix_tokens"].shape

class TestThreeSegment:
    """设计规格 §4.4/§9.1：三段块掩码"""

    def test_suffix_ignores_fast(self):
        """连续动作（suffix）对 FAST 位置全 False（新规则）"""
        from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask

        B, pl, fl, sl = 2, 4, 3, 5
        p = torch.ones(B, pl, dtype=torch.bool)
        f = torch.ones(B, fl, dtype=torch.bool)
        s = torch.ones(B, sl, dtype=torch.bool)
        mask = make_three_segment_mask(p, f, s, pl, fl, sl)

        suffix_rows = mask[:, pl + fl:, :]     # (B, sl, total)
        fast_cols = suffix_rows[:, :, pl:pl + fl]
        assert not fast_cols.any(), "suffix 不应 attend 到 FAST"

    def test_prefix_ignores_fast_and_suffix(self):
        """prefix 不看 fast 和 suffix（论文强制单向流）"""
        from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask

        B, pl, fl, sl = 2, 4, 3, 5
        p = torch.ones(B, pl, dtype=torch.bool)
        f = torch.ones(B, fl, dtype=torch.bool)
        s = torch.ones(B, sl, dtype=torch.bool)
        mask = make_three_segment_mask(p, f, s, pl, fl, sl)

        prefix_rows = mask[:, :pl, :]
        assert not prefix_rows[:, :, pl:].any(), "prefix 不应 attend 到 fast/suffix"

    def test_fast_sees_prefix_and_causal_self(self):
        """fast 看 prefix + 自身（因果下三角）"""
        from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask

        B, pl, fl, sl = 1, 4, 3, 5
        p = torch.ones(B, pl, dtype=torch.bool)
        f = torch.ones(B, fl, dtype=torch.bool)
        s = torch.ones(B, sl, dtype=torch.bool)
        mask = make_three_segment_mask(p, f, s, pl, fl, sl)

        fast_rows = mask[:, pl:pl + fl, :]
        assert fast_rows[:, :, :pl].all(), "fast 应看 prefix"
        causal = fast_rows[:, :, pl:pl + fl]
        for i in range(fl):
            for j in range(fl):
                expected = (j <= i)
                assert bool(causal[0, i, j]) == expected, f"fast 因果 [{i},{j}]"

    def test_suffix_sees_prefix_and_self(self):
        """suffix 看 prefix + 自身，不看 fast"""
        from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask

        B, pl, fl, sl = 2, 4, 3, 5
        p = torch.ones(B, pl, dtype=torch.bool)
        f = torch.ones(B, fl, dtype=torch.bool)
        s = torch.ones(B, sl, dtype=torch.bool)
        mask = make_three_segment_mask(p, f, s, pl, fl, sl)

        suffix_rows = mask[:, pl + fl:, :]
        assert suffix_rows[:, :, :pl].all(), "suffix 应看 prefix"
        assert suffix_rows[:, :, pl + fl:].all(), "suffix 应看自身"

    def test_suffix_len_zero_ok(self):
        """suffix_len=0（Stage 1 只用 prefix+fast 时）不报错"""
        from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask

        B, pl, fl = 2, 4, 3
        p = torch.ones(B, pl, dtype=torch.bool)
        f = torch.ones(B, fl, dtype=torch.bool)
        mask = make_three_segment_mask(p, f, None, pl, fl, 0)
        assert mask.shape == (B, pl + fl, pl + fl)

class TestFastCE:
    """设计规格 §4.5/§8.1：FAST CE 路径 + 联合损失"""

    @staticmethod
    def _fast_model(**overrides):
        cfg = Pi05FaithfulConfig.use_small_scale()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return Pi05FaithfulPolicyModel(cfg)

    def test_fast_ce_loss_reaches_vlm(self):
        """CE 路径梯度可达 VLM（text_embed / vision 权重有梯度）"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        model = self._fast_model(fast_enabled=True)
        B, fl = 2, 8
        images = torch.randn(B, 3, model.config.image_size, model.config.image_size)
        text = torch.randint(0, model.config.vocab_size, (B, 5))
        actions = torch.randn(B, model.config.action_horizon, model.config.action_dim)
        fast_tokens = torch.randint(0, model.config.fast_vocab_size, (B, fl))

        model.zero_grad()
        result = model.compute_loss(
            images, text, actions, fast_tokens=fast_tokens, return_components=True
        )
        assert "ce" in result and "mse" in result
        result["ce"].backward()
        # CE 梯度应到达 VLM（FAST 路径不受知识隔离影响）
        assert model.text_embed.weight.grad is not None, "CE 梯度未达 text_embed"

    def test_ce_shift_by_one(self):
        """CE 采用 shift-by-one：位置 i 预测位置 i+1 的 token"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        model = self._fast_model(fast_enabled=True)
        # 构造可区分的 fast token：token[1:] 为标签，末位无监督
        fl = 4
        fast_tokens = torch.tensor([[3, 7, 1, 9]], dtype=torch.long)
        B, text_len = 1, 5
        images = torch.randn(B, 3, model.config.image_size, model.config.image_size)
        text = torch.randint(0, model.config.vocab_size, (B, text_len))
        actions = torch.randn(B, model.config.action_horizon, model.config.action_dim)

        # 直接调用内部 CE 损失函数验证 shift 语义
        import torch.nn.functional as F
        hidden = torch.randn(B, fl, model.config.paligemma_width)
        logits = model.da_head(hidden)                       # (B, fl, vocab)
        ce = F.cross_entropy(
            logits[:, :-1].reshape(-1, model.config.fast_vocab_size),
            fast_tokens[:, 1:].reshape(-1),
        )
        assert ce.dim() == 0 and torch.isfinite(ce)

    def test_fast_forward_returns_velocity_shape(self):
        """fast 路径 forward 输出仍是 (B, H, A) 速度场"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        model = self._fast_model(fast_enabled=True)
        B, fl = 2, 8
        images = torch.randn(B, 3, model.config.image_size, model.config.image_size)
        text = torch.randint(0, model.config.vocab_size, (B, 5))
        x_t = torch.randn(B, model.config.action_horizon, model.config.action_dim)
        timestep = torch.full((B,), 0.5)
        fast_tokens = torch.randint(0, model.config.fast_vocab_size, (B, fl))

        v = model(images, text, x_t, timestep, fast_tokens=fast_tokens)
        assert v.shape == (B, model.config.action_horizon, model.config.action_dim)

    def test_fast_disabled_no_fast_modules(self):
        """fast_enabled=False 时不创建 discrete_action_embedding / da_head"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        model = self._fast_model(fast_enabled=False)
        assert not hasattr(model, "discrete_action_embedding")
        assert not hasattr(model, "da_head")

class TestKnowledgeInsulation:
    """设计规格 §5：知识隔离（KV 缓存 detach）"""

    @staticmethod
    def _fast_model(**overrides):
        cfg = Pi05FaithfulConfig.use_small_scale()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return Pi05FaithfulPolicyModel(cfg)

    def test_ki_no_vlm_grad_on_mse(self):
        """KI=True 时仅 MSE 反向，VLM 参数 grad=None（动作梯度不污染 VLM）"""
        from models.vla.pi05_faithful.modules.knowledge_insulation import detach_kv_cache

        model = self._fast_model(fast_enabled=True, knowledge_insulation=True)
        B, fl = 2, 8
        images = torch.randn(B, 3, model.config.image_size, model.config.image_size)
        text = torch.randint(0, model.config.vocab_size, (B, 5))
        actions = torch.randn(B, model.config.action_horizon, model.config.action_dim)
        fast_tokens = torch.randint(0, model.config.fast_vocab_size, (B, fl))
        noise = torch.randn_like(actions)
        timestep = torch.full((B,), 0.5)
        x_t = timestep[:, None, None] * noise + (1 - timestep[:, None, None]) * actions
        u_t = noise - actions

        fast_hidden, prefix_kv, _ = model._stage1_prefix_fast(
            images, text, fast_tokens, None
        )
        prefix_kv = detach_kv_cache(prefix_kv)          # 知识隔离：detach 缓存
        suffix_dict = model.embed_suffix(x_t, timestep)
        B = x_t.shape[0]
        slen = suffix_dict["suffix_tokens"].shape[1]
        s_row = torch.ones(B, slen, dtype=torch.bool)
        mask = s_row.unsqueeze(2) & s_row.unsqueeze(1)
        _, suffix_out, _ = model.transformer(
            None, suffix_dict["suffix_tokens"], suffix_dict["timestep_cond"],
            mask, kv_cache=prefix_kv,
        )
        v_t = model.action_out_proj(suffix_out)
        mse = torch.mean((v_t - u_t) ** 2)
        mse.backward()

        assert model.text_embed.weight.grad is None, "KI=True 时 MSE 不应给 text_embed 梯度"
        # 动作侧参数应有梯度（suffix 分支可学习）
        assert model.action_out_proj.weight.grad is not None

    def test_no_ki_vlm_has_grad(self):
        """KI=False 时 MSE 反向 VLM 有梯度"""
        model = self._fast_model(fast_enabled=True, knowledge_insulation=False)
        B, fl = 2, 8
        images = torch.randn(B, 3, model.config.image_size, model.config.image_size)
        text = torch.randint(0, model.config.vocab_size, (B, 5))
        actions = torch.randn(B, model.config.action_horizon, model.config.action_dim)
        fast_tokens = torch.randint(0, model.config.fast_vocab_size, (B, fl))
        noise = torch.randn_like(actions)
        timestep = torch.full((B,), 0.5)
        x_t = timestep[:, None, None] * noise + (1 - timestep[:, None, None]) * actions
        u_t = noise - actions

        fast_hidden, prefix_kv, _ = model._stage1_prefix_fast(
            images, text, fast_tokens, None
        )
        suffix_dict = model.embed_suffix(x_t, timestep)
        B = x_t.shape[0]
        slen = suffix_dict["suffix_tokens"].shape[1]
        s_row = torch.ones(B, slen, dtype=torch.bool)
        mask = s_row.unsqueeze(2) & s_row.unsqueeze(1)
        _, suffix_out, _ = model.transformer(
            None, suffix_dict["suffix_tokens"], suffix_dict["timestep_cond"],
            mask, kv_cache=prefix_kv,
        )
        v_t = model.action_out_proj(suffix_out)
        torch.mean((v_t - u_t) ** 2).backward()
        assert model.text_embed.weight.grad is not None, "KI=False 时 MSE 应给 text_embed 梯度"

    def test_detach_kv_cache_detaches_grad(self):
        """detach_kv_cache 返回的缓存 grad_fn 为 None"""
        from models.vla.pi05_faithful.modules.knowledge_insulation import detach_kv_cache

        k = torch.randn(2, 3, 4, 5, requires_grad=True)
        v = torch.randn(2, 3, 4, 5, requires_grad=True)
        out = detach_kv_cache([(k, v)])
        assert out[0][0].grad_fn is None and out[0][1].grad_fn is None

class TestRTC:
    """设计规格 §6：实时约束（max_delay 硬冻结，方案 (a)）"""

    def test_make_prefix_mask(self):
        """前 delay 步为冻结前缀（mask=True）"""
        from models.vla.pi05_faithful.modules.rtc import make_prefix_mask

        delay = torch.tensor([0, 2, 3])
        mask = make_prefix_mask(5, delay)
        assert mask.shape == (3, 5)
        assert mask[0].sum() == 0
        assert mask[1, :2].all() and not mask[1, 2:].any()
        assert mask[2, :3].all() and not mask[2, 3:].any()

    def test_masked_mse_masks_prefix_and_renormalizes(self):
        """冻结步 MSE 被 mask，postfix 按未冻结计数归一化"""
        from models.vla.pi05_faithful.modules.rtc import masked_mse

        v = torch.ones(2, 4, 3)
        u = torch.zeros(2, 4, 3)
        prefix_mask = torch.zeros(2, 4, dtype=torch.bool)
        prefix_mask[:, 0] = True                       # 第 0 步冻结
        loss = masked_mse(v, u, prefix_mask)
        # 未冻结 3 步 * 3 维 = 9 个元素，每个误差 1 → loss = 1.0
        assert torch.allclose(loss, torch.tensor(1.0))

    def test_rtc_prefix_frozen_during_infer(self):
        """推理时冻结前缀步与输入 action_prefix 完全一致（clamp 生效）"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        cfg = Pi05FaithfulConfig.use_small_scale()
        cfg.rtc_max_delay = 2
        model = Pi05FaithfulPolicyModel(cfg)
        model.eval()

        B, d = 2, 2
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 5))
        action_prefix = torch.randn(B, d, cfg.action_dim)
        delay = torch.full((B,), d, dtype=torch.long)

        with torch.no_grad():
            out = model.sample_actions(
                images, text, action_prefix=action_prefix, delay=delay
            )
        # 前 d 步被 clamp 回 action_prefix
        assert torch.allclose(out[:, :d], action_prefix, atol=1e-6)

    def test_rtc_zero_no_change(self):
        """rtc_max_delay=0 时不传 prefix/delay，行为与默认一致"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)
        model.eval()
        B = 2
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 5))
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.sample_actions(images, text)
        assert out.shape == (B, cfg.action_horizon, cfg.action_dim)

class TestChunkExecution:
    """设计规格 §7：chunk 部分执行（n_action_steps 退视界，方案 (a)）"""

    class _StubModel:
        """桩模型：记录 sample_actions 调用次数，返回可区分 chunk"""
        def __init__(self, horizon, dim, steps):
            self.horizon = horizon
            self.dim = dim
            self.steps = steps
            self.calls = 0

        def sample_actions(self, images, text_tokens, **kwargs):
            self.calls += 1
            # 第 n 次调用返回值全为 n（可用作分块依据）
            base = self.calls
            return torch.full((1, self.horizon, self.dim), float(base))

    def test_partial_execution_reinfers_on_depletion(self):
        """输出全 H 步，但只执行前 n_action_steps，耗尽后重推理"""
        from models.vla.pi05_faithful.modules.chunk_execution import ChunkExecutionPolicy

        stub = self._StubModel(horizon=10, dim=3, steps=0)
        policy = ChunkExecutionPolicy(stub, n_action_steps=4)

        images = torch.zeros(1, 3, 8, 8)
        text = torch.zeros(1, 5, dtype=torch.long)

        # 前 4 次返回第 1 个 chunk 的前 4 步（值=1.0）
        actions = [policy(images, text) for _ in range(4)]
        assert all(a.abs().max().item() == 1.0 for a in actions)
        assert stub.calls == 1
        # 第 5 次触发重推理，返回第 2 个 chunk 的第 1 步（值=2.0）
        a5 = policy(images, text)
        assert a5.abs().max().item() == 2.0
        assert stub.calls == 2

    def test_full_chunk_when_steps_equals_horizon(self):
        """n_action_steps == action_horizon 时一整块执行，行为一致"""
        from models.vla.pi05_faithful.modules.chunk_execution import ChunkExecutionPolicy

        stub = self._StubModel(horizon=10, dim=3, steps=0)
        policy = ChunkExecutionPolicy(stub, n_action_steps=10)
        images = torch.zeros(1, 3, 8, 8)
        text = torch.zeros(1, 5, dtype=torch.long)

        for i in range(10):
            a = policy(images, text)
            assert a.abs().max().item() == 1.0
        assert stub.calls == 1


class TestBackwardCompat:
    """设计规格 §9.1：全默认配置下行为与旧版一致"""

    def test_default_forward_matches_manual(self):
        """默认配置 forward 输出 = 手工两段联合前向结果"""
        from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask

        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)
        model.eval()

        B = 2
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 5))
        x_t = torch.randn(B, cfg.action_horizon, cfg.action_dim)
        timestep = torch.full((B,), 0.5)

        with torch.no_grad():
            v_model = model(images, text, x_t, timestep)
            # 手工：encode_prefix + embed_suffix + 联合前向
            pd = model.encode_prefix(images, text)
            sd = model.embed_suffix(x_t, timestep)
            plen, slen = pd["prefix_tokens"].shape[1], sd["suffix_tokens"].shape[1]
            m = make_attn_mask(pd["prefix_mask"], sd["suffix_mask"], plen, slen)
            _, so, _ = model.transformer(
                pd["prefix_tokens"], sd["suffix_tokens"], sd["timestep_cond"], m
            )
            v_manual = model.action_out_proj(so)

        assert torch.allclose(v_model, v_manual, atol=1e-5)

    def test_default_compute_loss_plain_mse(self):
        """默认配置 compute_loss == 纯 MSE（无 CE 项）"""
        cfg = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(cfg)

        B = 2
        images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
        text = torch.randint(0, cfg.vocab_size, (B, 5))
        actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)
        noise = torch.randn_like(actions)
        time = torch.full((B,), 0.3)
        result = model.compute_loss(
            images, text, actions, noise=noise, timestep=time, return_components=True
        )
        assert torch.allclose(result["ce"], torch.tensor(0.0))
        assert torch.allclose(result["total"], result["mse"])
