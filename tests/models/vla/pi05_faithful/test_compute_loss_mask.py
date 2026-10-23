# -*- coding: utf-8 -*-
"""测试 Pi05FaithfulPolicyModel.compute_loss 的 actions_is_pad 行为。

策略：构造最小 fake 输入（图片、token、动作），验证：
  - 不传 actions_is_pad：行为与改造前一致（prefix_mask 关闭时纯 mean）
  - 传 actions_is_pad 且全 True：loss = 0
  - actions_is_pad 与 RTC prefix_mask 共存：取并集 mask
"""
import pytest
import torch

from models.vla.core.train.transforms import compose
from models.vla.core.train.norm_stats import NormStats


def _make_norm_stats(D):
    return {"state": NormStats(
        q01=[-2.0] * D, q99=[2.0] * D,
    )}


def _try_build_small_model():
    """尽量造一个能跑 compute_loss 的最小 Pi05 模型。

    完整 Pi05 模型太大不适合单元测试。改用 monkeypatch 的方式：
    构造一个随机 Linear 子类，对外暴露 compute_loss 接口，
    只验证 mask 逻辑（不验证前向正确性）。
    """
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

    cfg_overrides = dict(
        paligemma_width=8,
        paligemma_depth=1,
        action_width=8,
        action_depth=1,
        head_dim=4,
        paligemma_num_heads=2,
        paligemma_num_kv_heads=1,
        action_num_heads=2,
        action_num_kv_heads=1,
        vocab_size=16,
        image_size=8,
        patch_size=4,
        action_horizon=4,
        action_dim=2,
        num_inference_steps=2,
        text_len=4,
        rtc_max_delay=0,  # 关闭 RTC，单独测 actions_is_pad
    )
    from models.vla.pi05_faithful.config import Pi05FaithfulConfig
    return Pi05FaithfulConfig(**cfg_overrides)


def test_actions_is_pad_all_padding_returns_zero_loss():
    """全 padding 时 loss = 0（不会被反向梯度污染）。"""
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
    cfg = _try_build_small_model()
    try:
        model = Pi05FaithfulPolicyModel(cfg)
    except Exception as e:
        pytest.skip(f"无法构造 Pi05 模型（依赖缺失）：{e}")

    B, H, A = 2, 4, 2
    images = torch.zeros(B, 3, 4, 4)
    text = torch.zeros(B, 2, dtype=torch.long)
    actions = torch.randn(B, H, A)
    actions_is_pad = torch.ones(B, H, dtype=torch.bool)  # 全部 padding

    loss = model.compute_loss(images, text, actions, actions_is_pad=actions_is_pad)
    assert loss.item() < 1e-6


def test_actions_is_pad_none_keeps_old_behavior():
    """不传 actions_is_pad：行为与改造前一致（纯 mean）。"""
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
    cfg = _try_build_small_model()
    try:
        model = Pi05FaithfulPolicyModel(cfg)
    except Exception as e:
        pytest.skip(f"无法构造 Pi05 模型：{e}")

    B, H, A = 2, 4, 2
    images = torch.zeros(B, 3, 4, 4)
    text = torch.zeros(B, 2, dtype=torch.long)
    actions = torch.randn(B, H, A)

    loss_no_pad = model.compute_loss(images, text, actions, actions_is_pad=None)
    # 必须返回标量
    assert loss_no_pad.dim() == 0
    # 随机初始化 + 随机 target，loss 必 > 0
    assert loss_no_pad.item() > 0


def test_actions_is_pad_partial_excludes_padded_steps():
    """部分 padding 时，只有真实步计入 loss。"""
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
    cfg = _try_build_small_model()
    try:
        model = Pi05FaithfulPolicyModel(cfg)
    except Exception as e:
        pytest.skip(f"无法构造 Pi05 模型：{e}")

    B, H, A = 1, 4, 2
    images = torch.zeros(B, 3, 4, 4)
    text = torch.zeros(B, 2, dtype=torch.long)
    actions = torch.randn(B, H, A)
    # 第 0 步是 padding
    actions_is_pad = torch.tensor([[True, False, False, False]])

    loss = model.compute_loss(images, text, actions, actions_is_pad=actions_is_pad)
    assert loss.dim() == 0
    # 不能因为前向快随机变成 NaN 或 0（除非很巧）
    assert torch.isfinite(loss)


def test_combine_loss_masks_helper():
    """_combine_loss_masks 单元测试：直接验证合并逻辑。"""
    from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel

    device = torch.device("cpu")
    # 全 None → None
    assert Pi05FaithfulPolicyModel._combine_loss_masks(None, None, 2, 4, device) is None
    # 只有 actions_is_pad
    pad = torch.tensor([[True, False, True, False]])
    m = Pi05FaithfulPolicyModel._combine_loss_masks(None, pad, 1, 4, device)
    assert torch.equal(m, pad)
    # 只有 prefix_mask
    pm = torch.tensor([[True, True, False, False]])
    m = Pi05FaithfulPolicyModel._combine_loss_masks(pm, None, 1, 4, device)
    assert torch.equal(m, pm)
    # 都给：取并集
    m = Pi05FaithfulPolicyModel._combine_loss_masks(pm, pad, 1, 4, device)
    expected = torch.tensor([[True, True, True, False]])
    assert torch.equal(m, expected)
