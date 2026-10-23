"""混合 Muon+AdamW 优化器包装器。

将 Muon 和 AdamW 两个优化器组合为统一接口，支持：
- 统一的 step() / zero_grad() / state_dict() 接口
- 事务性更新（NaN 梯度跳过整步）
- 独立的 Muon LR 和 AdamW LR 控制
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class OptimizerStepReport:
    """单步优化报告。"""
    executed: bool
    adamw_lr: float
    muon_lr: float
    nonfinite_parameters: int
    update_metrics: dict[str, float]


class HybridMuonAdamW:
    """混合 Muon+AdamW 优化器。

    组合 KimiMuon 和 torch.optim.AdamW 为统一接口。
    step() 是事务性的：如果检测到非有限梯度，跳过整个步骤。
    """

    def __init__(
        self,
        muon: torch.optim.Optimizer,
        adamw: torch.optim.Optimizer,
    ):
        """
        Args:
            muon: Muon 优化器实例（处理 use_muon=True 的组）
            adamw: AdamW 优化器实例（处理 use_muon=False 的组）
        """
        self.muon = muon
        self.adamw = adamw
        self.step_number = 0
        self.last_step_report: OptimizerStepReport | None = None

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None) -> OptimizerStepReport:
        """事务性更新：NaN 梯度时跳过整步。"""
        if closure is not None:
            with torch.enable_grad():
                closure()

        # 收集所有有梯度的参数
        parameters = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        # 检查非有限梯度
        nonfinite = sum(
            not bool(torch.isfinite(p.grad).all()) for p in parameters
        )
        if nonfinite:
            report = OptimizerStepReport(
                executed=False,
                adamw_lr=self.adamw.param_groups[0]["lr"],
                muon_lr=self.muon.param_groups[0]["lr"],
                nonfinite_parameters=nonfinite,
                update_metrics={},
            )
            self.last_step_report = report
            return report

        # 正常更新
        self.muon.step()
        self.adamw.step()
        self.step_number += 1

        report = OptimizerStepReport(
            executed=True,
            adamw_lr=self.adamw.param_groups[0]["lr"],
            muon_lr=self.muon.param_groups[0]["lr"],
            nonfinite_parameters=0,
            update_metrics={},
        )
        self.last_step_report = report
        return report

    def set_lr(self, lr: float, muon_lr: float | None = None) -> None:
        """设置学习率。"""
        for group in self.adamw.param_groups:
            group["lr"] = lr
        for group in self.muon.param_groups:
            group["lr"] = lr if muon_lr is None else muon_lr

    def state_dict(self) -> dict:
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
            "step_number": self.step_number,
        }

    def load_state_dict(self, state: dict) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])
        self.step_number = state.get("step_number", 0)
