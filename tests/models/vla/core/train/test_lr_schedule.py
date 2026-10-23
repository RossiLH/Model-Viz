# -*- coding: utf-8 -*-
"""lr_schedule.py 的单元测试。

覆盖：
  - warmup 阶段线性增长
  - 衰减起点恰好 = peak
  - 中段余弦形状（端点和中间点数值）
  - 衰减后保持 end_value
  - attach_lr_scheduler 接到真 optimizer 后 step() 推进正确
"""
import math

import pytest
import torch

from models.vla.core.train.lr_schedule import (
    CosineDecayWithWarmupConfig,
    attach_lr_scheduler,
    warmup_cosine_decay_schedule,
)


def _make_opt_and_sched(peak_lr, warmup, decay, end):
    """便利构造：opt base lr = peak，scheduler 装上去。

    注意：必须用 attach_lr_scheduler（或者手动改 initial_lr），否则
    LambdaLR 拿到的 base 仍是 1.0（默认 base），导致 lr 计算错位。
    """
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=peak_lr)
    cfg = CosineDecayWithWarmupConfig(
        warmup_steps=warmup, peak_lr=peak_lr, decay_steps=decay, decay_lr=end
    )
    sched = attach_lr_scheduler(opt, cfg)
    return opt, sched


def _lr_at(scheduler, step: int) -> float:
    """强制把 scheduler 推到指定 step 并读取 lr。

    PyTorch LambdaLR 的 step() 才把 lr 写回 param_groups，且每次 +1；
    所以从初始 last_epoch=0 走到目标 step 需要 step() 调用 (target) 次。

    每次调用前必须用 _fresh_sched() 拿新 scheduler，避免 last_epoch 倒拨的麻烦。
    """
    cur = scheduler.last_epoch
    while cur < step:
        scheduler.step()
        cur += 1
    return scheduler.optimizer.param_groups[0]["lr"]


def _fresh_sched(peak_lr, warmup, decay, end):
    """构造一个全新的 opt+sched 组合，专供 _lr_at 一次性查询用。"""
    return _make_opt_and_sched(peak_lr, warmup, decay, end)[1]


def test_warmup_phase_is_linear():
    """前 warmup_steps 步 LR 应该线性从 init 涨到 peak。"""
    init = 0.01 / 11  # peak/(warmup+1)
    # step 0 → init
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 0) == pytest.approx(init, rel=1e-5)
    # step 5：init + (peak-init)*0.5
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    expected_mid = init + (0.01 - init) * (5 / 10)
    assert _lr_at(sched, 5) == pytest.approx(expected_mid, rel=1e-5)
    # step 10 → peak
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 10) == pytest.approx(0.01, rel=1e-5)


def test_decay_phase_cosine_shape():
    """中段余弦曲线在端点处分别取 peak / end。"""
    # 衰减起点 = step 10 → peak
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 10) == pytest.approx(0.01, rel=1e-5)
    # 衰减终点 = step 30 → end
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 30) == pytest.approx(0.0001, rel=1e-5)
    # 中点 step 20：cos(pi/2) = 0 → value = (peak + end) / 2
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    expected_mid = (0.01 + 0.0001) * 0.5
    assert _lr_at(sched, 20) == pytest.approx(expected_mid, rel=1e-4)


def test_after_decay_stays_at_end():
    """衰减结束后 LR 必须保持 end_value，不能继续下降。"""
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 30) == pytest.approx(0.0001, rel=1e-5)
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 100) == pytest.approx(0.0001, rel=1e-5)
    sched = _fresh_sched(peak_lr=0.01, warmup=10, decay=20, end=0.0001)
    assert _lr_at(sched, 10000) == pytest.approx(0.0001, rel=1e-5)


def test_attach_to_optimizer_and_step():
    """attach_lr_scheduler 接到真 optimizer，调用 step() 推进 LR。"""
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=1.0)
    cfg = CosineDecayWithWarmupConfig(
        warmup_steps=4, peak_lr=0.01, decay_steps=10, decay_lr=0.001
    )
    sched = attach_lr_scheduler(opt, cfg)
    # step 0 的 lr = init_value = peak/(warmup+1) = 0.01/5
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01 / 5, rel=1e-5)
    # 推进 3 次：第 3 步 (step 索引 3) 还在 warmup 段
    for _ in range(3):
        sched.step()
    # step 3/4 的 LR
    init = 0.01 / 5
    expected = init + (0.01 - init) * (3 / 4)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-5)


def test_warmup_zero_raises():
    """warmup_steps=0 是非法输入，必须抛错。"""
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.01)
    with pytest.raises(ValueError):
        warmup_cosine_decay_schedule(
            opt, init_value=0.001, peak_value=0.01, warmup_steps=0,
            decay_steps=10, end_value=0.0001,
        )


def test_attach_with_official_default_matches_optax():
    """用官方默认超参调用时，验证 init_value = peak/(warmup+1) 行为正确。"""
    cfg = CosineDecayWithWarmupConfig()  # 全部默认
    assert cfg.warmup_steps == 1000
    assert cfg.peak_lr == 2.5e-5
    assert cfg.decay_lr == 2.5e-6
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=0.1)  # base lr 故意不对，验证 attach 会覆盖
    sched = attach_lr_scheduler(opt, cfg)
    expected_init = 2.5e-5 / 1001
    assert opt.param_groups[0]["lr"] == pytest.approx(expected_init, rel=1e-5)
