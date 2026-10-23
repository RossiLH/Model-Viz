# -*- coding: utf-8 -*-
"""transforms.py 的单元测试。

覆盖以下不变式：
  - Normalize / Unnormalize 严格互逆（往返误差 < 1e-5）
  - 图像类 uint8 数据被跳过
  - PadStatesAndActions 在不足和超过两种情况下都正确
  - DeltaActions / AbsoluteActions 严格互逆
  - DeltaActions 缺 state 时是 no-op（不抛错）
  - ResizeImages 对 (H, W, 3) uint8 正确缩放，对非图像原样透传
  - compose 严格按列表顺序执行
"""

import numpy as np
import pytest

from models.vla.core.train.norm_stats import NormStats
from models.vla.core.train.transforms import (
    AbsoluteActions,
    DeltaActions,
    Normalize,
    PadStatesAndActions,
    ResizeImages,
    Unnormalize,
    _pad_last_dim,
    compose,
)


# ---------- 工具 ----------
def _make_stats(q01: list, q99: list, mean=None, std=None) -> NormStats:
    """构造一个分位数统计量，缺省 mean/std 用于 z-score 测试。"""
    return NormStats(
        q01=np.array(q01, dtype=np.float32),
        q99=np.array(q99, dtype=np.float32),
        mean=np.array(mean, dtype=np.float32) if mean is not None else None,
        std=np.array(std, dtype=np.float32) if std is not None else None,
    )


# ---------- Normalize / Unnormalize ----------

def test_normalize_unnormalize_roundtrip_quantile():
    """分位数归一化的最关键不变式：Normalize -> Unnormalize 严格互逆。"""
    stats = {"actions": _make_stats([0.0, -1.0], [1.0, 1.0])}
    norm = Normalize(norm_stats=stats, use_quantiles=True)
    unnorm = Unnormalize(norm_stats=stats, use_quantiles=True)

    x = np.array([[0.5, 0.0], [-0.5, 0.5], [0.0, -0.5]], dtype=np.float32)
    data = {"actions": x.copy()}
    data = unnorm(norm(data))
    np.testing.assert_allclose(data["actions"], x, atol=1e-5)


def test_normalize_unnormalize_roundtrip_zscore():
    """z-score 归一化也需要可逆，用于 PI0 训练场景。"""
    stats = {
        "state": _make_stats([], [],
                             mean=[0.0, 1.0], std=[2.0, 0.5])
    }
    norm = Normalize(norm_stats=stats, use_quantiles=False)
    unnorm = Unnormalize(norm_stats=stats, use_quantiles=False)

    x = np.array([[2.0, 1.5], [-4.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    data = {"state": x.copy()}
    data = unnorm(norm(data))
    np.testing.assert_allclose(data["state"], x, atol=1e-5)


def test_normalize_quantile_formula():
    """锁定分位数归一化公式：x=(x-q01)/(q99-q01)*2-1。"""
    stats = {"actions": _make_stats([0.0], [2.0])}
    norm = Normalize(norm_stats=stats, use_quantiles=True)
    out = norm({"actions": np.array([[0.0], [1.0], [2.0]])})
    # 0 -> -1, 1 -> 0, 2 -> +1
    np.testing.assert_allclose(out["actions"],
                               [[-1.0], [0.0], [1.0]], atol=1e-5)


def test_normalize_skips_uint8_images():
    """图像类数据（uint8）必须被跳过，归一化前后逐字节相等。"""
    stats = {"image": _make_stats([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])}
    norm = Normalize(norm_stats=stats, use_quantiles=True)
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = 200
    data = {"image": img.copy()}
    out = norm(data)
    # 图像没被改动，dtype 仍是 uint8
    assert out["image"].dtype == np.uint8
    np.testing.assert_array_equal(out["image"], img)


def test_normalize_strict_raises_on_missing_key():
    """strict=True 时缺关键字段必须立即抛错，方便排查配置错误。"""
    stats = {"actions": _make_stats([0.0], [1.0])}
    norm = Normalize(norm_stats=stats, use_quantiles=True, strict=True)
    with pytest.raises(KeyError):
        norm({"state": np.zeros((1, 1))})


def test_normalize_non_strict_skips_missing_key():
    """strict=False 时缺字段静默通过，管线不中断。"""
    stats = {"actions": _make_stats([0.0], [1.0])}
    norm = Normalize(norm_stats=stats, use_quantiles=True, strict=False)
    out = norm({"state": np.zeros((1, 1))})
    assert "state" in out
    np.testing.assert_array_equal(out["state"], np.zeros((1, 1)))


def test_unnormalize_preserves_padding_dims():
    """反归一化时，超过 stats 维度的 padding 部分必须原样保留。

    真实工程场景：训练时 state 被 pad 到 model_action_dim，但统计量是按
    原始维度算的；Unnormalize 必须只对真实维度做反变换，padding 不动。
    """
    stats = {"actions": _make_stats([0.0], [1.0])}  # stats 1 维
    unnorm = Unnormalize(norm_stats=stats, use_quantiles=True)
    # 数据 3 维，前 1 维是真实信号（值=0），后 2 维是 padding（值=0.5）
    x = np.array([[0.0, 0.5, 0.5]], dtype=np.float32)
    out = unnorm({"actions": x.copy()})
    # padding 维保持 0.5
    assert out["actions"][0, 1] == pytest.approx(0.5)
    assert out["actions"][0, 2] == pytest.approx(0.5)
    # 真实维：0 在 [-1, 1] 中点 → (0+1)/2 * (q99-q01+eps) + q01 = 0.5 + 5e-7
    assert out["actions"][0, 0] == pytest.approx(0.5, abs=1e-5)


# ---------- PadStatesAndActions ----------

def test_pad_states_and_actions_pad_and_truncate():
    """不足时 pad 到 model_action_dim，超过时截断。"""
    pad = PadStatesAndActions(model_action_dim=4)
    data = {
        "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),  # 3 -> 4 (pad 1 个 0)
        "actions": np.array([1, 2, 3, 4, 5], dtype=np.float32),  # 5 -> 4 (截断)
    }
    out = pad(data)
    assert out["state"].shape == (4,)
    np.testing.assert_array_equal(out["state"], [1.0, 2.0, 3.0, 0.0])
    assert out["actions"].shape == (4,)
    np.testing.assert_array_equal(out["actions"], [1, 2, 3, 4])


def test_pad_last_dim_preserves_shape():
    """_pad_last_dim 应对任意前置维数都正确。"""
    a = np.zeros((2, 3, 5), dtype=np.float32)
    out = _pad_last_dim(a, 7)
    assert out.shape == (2, 3, 7)
    out2 = _pad_last_dim(out, 4)
    assert out2.shape == (2, 3, 4)


# ---------- DeltaActions / AbsoluteActions ----------

def test_delta_absolute_roundtrip_chunk():
    """delta 与 absolute 互逆：absolute(delta(x)) == x。"""
    # 7 维动作：前 6 维做 delta（关节），第 7 维保持绝对（夹爪）
    mask = [True, True, True, True, True, True, False]
    state = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], dtype=np.float32)
    # chunk 内 3 步动作，前 6 维是增量、第 7 维是绝对夹爪目标
    actions_abs = np.array([
        [0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.8],
        [0.0,  0.0, 0.2,  0.0, 0.3,  0.0, 0.5],
        [0.1,  0.1, 0.1,  0.1, 0.1,  0.1, 0.2],
    ], dtype=np.float32)

    delta = DeltaActions(mask=mask)({"actions": actions_abs.copy(), "state": state})
    recovered = AbsoluteActions(mask=mask)({"actions": delta["actions"], "state": state})
    np.testing.assert_allclose(recovered["actions"], actions_abs, atol=1e-6)


def test_delta_absolute_roundtrip_batched():
    """批量数据 (B, H, A) 也要能往返。"""
    mask = [True, True, False]  # 关节 delta，夹爪绝对
    state = np.array([[0.1, 0.2, 0.5], [-0.1, -0.2, 0.3]], dtype=np.float32)
    actions_abs = np.array([
        [[0.2, 0.3, 0.7], [0.1, 0.2, 0.6]],
        [[-0.2, -0.3, 0.4], [0.0, -0.1, 0.5]],
    ], dtype=np.float32)

    delta = DeltaActions(mask=mask)({"actions": actions_abs.copy(), "state": state})
    recovered = AbsoluteActions(mask=mask)({"actions": delta["actions"], "state": state})
    np.testing.assert_allclose(recovered["actions"], actions_abs, atol=1e-6)


def test_delta_actions_noop_when_state_missing():
    """缺 state 时 delta 应该是空操作（不抛错，方便管线组合）。"""
    mask = [True, True, False]
    actions = np.array([[0.1, 0.2, 0.5]], dtype=np.float32)
    out = DeltaActions(mask=mask)({"actions": actions.copy()})
    np.testing.assert_array_equal(out["actions"], actions)


def test_delta_actions_mask_shorter_than_action_dim():
    """mask 比动作维度短时，超出 mask 范围的部分按绝对处理。"""
    # 2 维 mask + 实际 4 维动作：前 2 维 delta，后 2 维保持绝对
    mask = [True, True]
    state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    actions = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)

    out = DeltaActions(mask=mask)({"actions": actions, "state": state})
    # 前 2 维：减 state → [9, 18]
    # 后 2 维：mask 之外，保持原样 → [30, 40]
    np.testing.assert_allclose(out["actions"],
                               [[9.0, 18.0, 30.0, 40.0]], atol=1e-6)


# ---------- ResizeImages ----------

def test_resize_images_resizes_all_cameras():
    """所有相机视角都要被 resize 到目标尺寸。"""
    resize = ResizeImages(height=8, width=12)
    images = {
        "front": np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8),
        "wrist": np.random.randint(0, 256, (32, 24, 3), dtype=np.uint8),
    }
    out = resize({"image": images})
    assert out["image"]["front"].shape == (8, 12, 3)
    assert out["image"]["front"].dtype == np.uint8
    assert out["image"]["wrist"].shape == (8, 12, 3)
    assert out["image"]["wrist"].dtype == np.uint8


def test_resize_images_passthrough_non_image():
    """非 (H, W, 3) 数据要原样透传（避免破坏 mask / depth 等辅助字段）。"""
    resize = ResizeImages(height=8, width=8)
    depth = np.zeros((16, 16), dtype=np.float32)  # 缺通道维
    out = resize({"image": {"depth": depth}})
    assert out["image"]["depth"].shape == (16, 16)
    np.testing.assert_array_equal(out["image"]["depth"], depth)


def test_resize_images_skip_when_already_correct_size():
    """尺寸已经正确时跳过重采样（节省时间 + 避免精度损失）。"""
    resize = ResizeImages(height=4, width=4)
    img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
    out = resize({"image": {"front": img.copy()}})
    np.testing.assert_array_equal(out["image"]["front"], img)


# ---------- compose ----------

def test_compose_executes_in_order():
    """compose 必须严格按列表顺序执行：先 delta 再 normalize。"""
    mask = [True, True]
    state = np.array([1.0, 2.0], dtype=np.float32)
    actions = np.array([[2.0, 4.0]], dtype=np.float32)

    stats = {"actions": _make_stats([-1.0], [1.0])}  # 故意只 1 维，让顺序错就报错

    pipeline = compose([
        DeltaActions(mask=mask),
        Normalize(norm_stats=stats, use_quantiles=True),
    ])
    # 第一步 delta 后：actions = [[1, 2]]（减 state）
    # 第二步 normalize 用 1 维 stats 取最后一维 → [0.0]
    out = pipeline({"actions": actions, "state": state})
    assert out["actions"].shape == (1, 2)
    # 验证：最后一维是 actions[:, 1] = 2，归一化公式 (2 - (-1))/(1-(-1))*2-1 = 2.0
    assert out["actions"][0, 1] == pytest.approx(2.0, abs=1e-5)


def test_compose_empty_is_identity():
    """空 compose 应该是恒等函数。"""
    pipeline = compose([])
    data = {"x": np.array([1, 2, 3])}
    out = pipeline(data)
    np.testing.assert_array_equal(out["x"], data["x"])
