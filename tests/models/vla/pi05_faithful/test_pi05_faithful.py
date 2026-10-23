"""PI0.5 忠实还原测试"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))

import torch
from models.vla.pi05_faithful.config import Pi05FaithfulConfig


class TestPi05FaithfulConfig:
    """配置数据类测试"""

    def test_small_scale_config(self):
        """测试小规模配置创建"""
        config = Pi05FaithfulConfig.use_small_scale()
        assert config.paligemma_width == 256
        assert config.paligemma_depth == 4
        assert config.action_width == 128
        assert config.action_depth == 4
        assert config.head_dim == 32
        assert config.paligemma_num_heads == 4
        assert config.paligemma_num_kv_heads == 1
        assert config.action_horizon == 10
        assert config.action_dim == 32
        assert config.num_inference_steps == 5

    def test_full_scale_config(self):
        """测试完整规模配置创建（对齐论文附录 A-E）"""
        config = Pi05FaithfulConfig.use_full_scale()
        assert config.paligemma_width == 2048
        assert config.paligemma_depth == 18
        assert config.action_width == 1024
        assert config.action_depth == 18
        assert config.head_dim == 256
        # 论文附录 A-E: num_heads=18, num_kv_heads=1
        assert config.paligemma_num_heads == 18
        assert config.paligemma_num_kv_heads == 1
        # 论文附录 A-E: VLM mlp_dim=16384，action expert mlp_dim=4096
        assert config.paligemma_mlp_dim == 16384
        assert config.action_mlp_dim == 4096
        assert config.action_horizon == 50
        assert config.action_dim == 32
        assert config.num_inference_steps == 10

    def test_config_derived_values(self):
        """测试配置派生值"""
        config = Pi05FaithfulConfig.use_small_scale()
        # mlp_dim 默认为 width * 4
        assert config.paligemma_mlp_dim == config.paligemma_width * 4
        assert config.action_mlp_dim == config.action_width * 4


class TestStateTokenizer:
    """状态离散化测试"""

    def test_tokenize_with_state(self):
        """测试带状态的 tokenization"""
        from models.vla.pi05_faithful.modules.state_tokenizer import StateTokenizer
        
        tokenizer = StateTokenizer()
        state = torch.tensor([0.5, -0.5, 0.0, 1.0, -1.0])
        result = tokenizer.tokenize("pick up the cup", state)
        
        assert "Task: pick up the cup" in result
        assert "State:" in result
        assert "Action:" in result
        # 状态值应该被离散化到 0-255
        assert all(c.isdigit() or c.isspace() for c in result.split("State:")[1].split(";")[0])

    def test_tokenize_without_state(self):
        """测试无状态的 tokenization"""
        from models.vla.pi05_faithful.modules.state_tokenizer import StateTokenizer
        
        tokenizer = StateTokenizer()
        result = tokenizer.tokenize("pick up the cup", None)
        
        assert "Task: pick up the cup" in result
        assert "Action:" in result
        assert "State:" not in result

    def test_discretization_range(self):
        """测试离散化范围在 0-255"""
        from models.vla.pi05_faithful.modules.state_tokenizer import StateTokenizer
        
        tokenizer = StateTokenizer()
        # 测试边界值
        state = torch.tensor([-1.0, 0.0, 1.0])
        result = tokenizer.tokenize("test", state)
        
        # 提取状态部分
        state_part = result.split("State:")[1].split(";")[0].strip()
        values = [int(x) for x in state_part.split()]
        
        assert all(0 <= v <= 255 for v in values)


class TestSimplifiedSigLIP:
    """视觉编码器测试"""

    def test_forward_shape(self):
        """测试输出维度"""
        from models.vla.pi05_faithful.modules.vision_encoder import SimplifiedSigLIP
        from models.vla.pi05_faithful.config import Pi05FaithfulConfig
        
        config = Pi05FaithfulConfig.use_small_scale()
        encoder = SimplifiedSigLIP(config)
        
        # 输入: (B, C, H, W)
        images = torch.randn(2, 3, config.image_size, config.image_size)
        tokens = encoder(images)
        
        # 输出: (B, num_patches, width)
        # 官方 PaliGemma 路径无 CLS token（pool_type="none"），故无 +1
        assert tokens.shape == (2, config.num_image_tokens, config.paligemma_width)

    def test_gradient_flow(self):
        """测试梯度能够反向传播"""
        from models.vla.pi05_faithful.modules.vision_encoder import SimplifiedSigLIP
        from models.vla.pi05_faithful.config import Pi05FaithfulConfig
        
        config = Pi05FaithfulConfig.use_small_scale()
        encoder = SimplifiedSigLIP(config)
        
        images = torch.randn(1, 3, config.image_size, config.image_size)
        tokens = encoder(images)
        loss = tokens.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


class TestDualExpertJointAttention:
    """联合注意力测试"""

    def test_forward_shapes(self):
        """测试输出维度"""
        from models.vla.pi05_faithful.modules.dual_expert_attention import (
            DualExpertJointAttention, make_attn_mask,
        )

        config = Pi05FaithfulConfig.use_small_scale()
        attn = DualExpertJointAttention(config)

        B = 2
        prefix_len = 10
        suffix_len = 5

        prefix = torch.randn(B, prefix_len, config.paligemma_width)
        suffix = torch.randn(B, suffix_len, config.action_width)
        # 二维块掩码（prefix 不看 suffix）
        p = torch.ones(B, prefix_len, dtype=torch.bool)
        s = torch.ones(B, suffix_len, dtype=torch.bool)
        mask = make_attn_mask(p, s, prefix_len, suffix_len)

        prefix_out, suffix_out, kv_cache = attn(prefix, suffix, mask)

        assert prefix_out.shape == (B, prefix_len, config.paligemma_width)
        assert suffix_out.shape == (B, suffix_len, config.action_width)
        assert kv_cache is not None

    def test_with_kv_cache(self):
        """测试 KV cache 支持"""
        from models.vla.pi05_faithful.modules.dual_expert_attention import (
            DualExpertJointAttention, make_attn_mask,
        )

        config = Pi05FaithfulConfig.use_small_scale()
        attn = DualExpertJointAttention(config)

        B = 1
        prefix_len = 8
        suffix_len = 4

        prefix = torch.randn(B, prefix_len, config.paligemma_width)
        suffix = torch.randn(B, suffix_len, config.action_width)
        p = torch.ones(B, prefix_len, dtype=torch.bool)
        s = torch.ones(B, suffix_len, dtype=torch.bool)
        mask = make_attn_mask(p, s, prefix_len, suffix_len)

        # 第一次前向（无 cache）
        prefix_out1, suffix_out1, kv_cache = attn(prefix, suffix, mask)

        # 第二次前向（有 cache，只处理 suffix）
        suffix2 = torch.randn(B, suffix_len, config.action_width)
        s2 = torch.ones(B, suffix_len, dtype=torch.bool)
        # 增量掩码仅覆盖新增 suffix keys，缓存部分在内部补全可见
        mask2 = s2.unsqueeze(2) & s2.unsqueeze(1)
        prefix_out2, suffix_out2, kv_cache2 = attn(None, suffix2, mask2, kv_cache=kv_cache)

        # prefix_out2 应为 None（不重新处理 prefix）
        assert prefix_out2 is None
        assert suffix_out2.shape == (B, suffix_len, config.action_width)

    def test_gradient_flow(self):
        """测试梯度能够反向传播"""
        from models.vla.pi05_faithful.modules.dual_expert_attention import (
            DualExpertJointAttention, make_attn_mask,
        )

        config = Pi05FaithfulConfig.use_small_scale()
        attn = DualExpertJointAttention(config)

        B = 1
        prefix = torch.randn(B, 5, config.paligemma_width, requires_grad=True)
        suffix = torch.randn(B, 3, config.action_width, requires_grad=True)
        p = torch.ones(B, 5, dtype=torch.bool)
        s = torch.ones(B, 3, dtype=torch.bool)
        mask = make_attn_mask(p, s, 5, 3)

        prefix_out, suffix_out, _ = attn(prefix, suffix, mask)
        loss = prefix_out.sum() + suffix_out.sum()
        loss.backward()

        assert prefix.grad is not None
        assert suffix.grad is not None


class TestPi05FaithfulPolicyModel:
    """主模型测试"""

    def test_forward_pass(self):
        """测试完整前向传播"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)

        B = 2
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 10))
        noisy_actions = torch.randn(B, config.action_horizon, config.action_dim)
        timestep = torch.rand(B)

        output = model(images, text_tokens, noisy_actions, timestep)

        assert output.shape == (B, config.action_horizon, config.action_dim)

    def test_sample_actions(self):
        """测试推理采样"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)
        model.eval()

        B = 2
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 10))

        with torch.no_grad():
            actions = model.sample_actions(images, text_tokens)

        assert actions.shape == (B, config.action_horizon, config.action_dim)
        assert not torch.isnan(actions).any()
        assert not torch.isinf(actions).any()

    def test_gradient_flow(self):
        """测试梯度反向传播"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)

        B = 1
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 8))
        noisy_actions = torch.randn(B, config.action_horizon, config.action_dim, requires_grad=True)
        timestep = torch.rand(B)

        output = model(images, text_tokens, noisy_actions, timestep)
        loss = output.sum()
        loss.backward()

        assert noisy_actions.grad is not None

    def test_model_parameters(self):
        """测试模型参数数量"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)

        total = sum(p.numel() for p in model.parameters())
        assert total > 0


class TestPi05Transformer:
    """Transformer 测试"""

    def test_block_forward(self):
        """测试单个 block 前向传播"""
        from models.vla.pi05_faithful.modules.pi05_transformer import Pi05TransformerBlock
        from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask

        config = Pi05FaithfulConfig.use_small_scale()
        block = Pi05TransformerBlock(config)

        B = 2
        prefix = torch.randn(B, 10, config.paligemma_width)
        suffix = torch.randn(B, 5, config.action_width)
        # adaRMS 条件宽度 = action expert 宽度（官方 pi0.py:164-169）
        timestep_cond = torch.randn(B, config.action_width)
        p = torch.ones(B, 10, dtype=torch.bool)
        s = torch.ones(B, 5, dtype=torch.bool)
        mask = make_attn_mask(p, s, 10, 5)

        prefix_out, suffix_out, _ = block(prefix, suffix, timestep_cond, mask)

        assert prefix_out.shape == prefix.shape
        assert suffix_out.shape == suffix.shape

    def test_stack_forward(self):
        """测试多层 Transformer"""
        from models.vla.pi05_faithful.modules.pi05_transformer import Pi05Transformer
        from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask

        config = Pi05FaithfulConfig.use_small_scale()
        transformer = Pi05Transformer(config)

        B = 2
        prefix = torch.randn(B, 10, config.paligemma_width)
        suffix = torch.randn(B, 5, config.action_width)
        timestep_cond = torch.randn(B, config.action_width)
        p = torch.ones(B, 10, dtype=torch.bool)
        s = torch.ones(B, 5, dtype=torch.bool)
        mask = make_attn_mask(p, s, 10, 5)

        prefix_out, suffix_out, kv = transformer(prefix, suffix, timestep_cond, mask)

        assert prefix_out.shape == prefix.shape
        assert suffix_out.shape == suffix.shape
        assert len(kv) == config.paligemma_depth

    def test_gradient_flow(self):
        """测试梯度反向传播"""
        from models.vla.pi05_faithful.modules.pi05_transformer import Pi05Transformer
        from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask

        config = Pi05FaithfulConfig.use_small_scale()
        transformer = Pi05Transformer(config)

        B = 1
        prefix = torch.randn(B, 5, config.paligemma_width, requires_grad=True)
        suffix = torch.randn(B, 3, config.action_width, requires_grad=True)
        timestep_cond = torch.randn(B, config.action_width)
        p = torch.ones(B, 5, dtype=torch.bool)
        s = torch.ones(B, 3, dtype=torch.bool)
        mask = make_attn_mask(p, s, 5, 3)

        prefix_out, suffix_out, _ = transformer(prefix, suffix, timestep_cond, mask)
        loss = prefix_out.sum() + suffix_out.sum()
        loss.backward()

        assert prefix.grad is not None
        assert suffix.grad is not None


class TestPi05FaithfulE2E:
    """端到端测试"""

    def test_full_training_step(self):
        """测试完整训练步骤（Flow Matching）"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
        from models.vla.pi05_faithful.modules.time_sampler import sample_flow_time

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        B = 4
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 10))
        actions = torch.randn(B, config.action_horizon, config.action_dim)

        model.train()
        optimizer.zero_grad()

        # 采样噪声和时间步（对应官方 pi0.py + 论文附录 A-E 的 Beta((s-tau)/s) 截断采样）
        noise = torch.randn_like(actions)
        time_t = sample_flow_time((B,), alpha=1.5, beta=1.0, s=0.999)
        time_expanded = time_t[:, None, None]
        noisy_actions = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions

        # 前向 + Loss
        predicted_velocity = model(images, text_tokens, noisy_actions, time_t)
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_inference_pipeline(self):
        """测试完整推理流程"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)
        model.eval()

        B = 2
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 10))

        with torch.no_grad():
            actions = model.sample_actions(images, text_tokens, num_steps=3)

        assert actions.shape == (B, config.action_horizon, config.action_dim)
        assert not torch.isnan(actions).any()
        assert not torch.isinf(actions).any()

    def test_model_deterministic(self):
        """测试模型确定性（固定随机种子时相同输入相同输出）"""
        from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

        config = Pi05FaithfulConfig.use_small_scale()
        model = Pi05FaithfulPolicyModel(config)
        model.eval()

        B = 1
        images = torch.randn(B, 3, config.image_size, config.image_size)
        text_tokens = torch.randint(0, config.vocab_size, (B, 5))

        # 采样动作的初始噪声是随机的，固定种子保证可复现
        torch.manual_seed(42)
        with torch.no_grad():
            actions1 = model.sample_actions(images, text_tokens)
        torch.manual_seed(42)
        with torch.no_grad():
            actions2 = model.sample_actions(images, text_tokens)

        assert torch.allclose(actions1, actions2, atol=1e-6)
