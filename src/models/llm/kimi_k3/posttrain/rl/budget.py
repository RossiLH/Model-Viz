"""Reasoning-effort control by token budget (paper section 4.1.2).

To tune reasoning effort while maximising token efficiency, K3 associates each
problem ``x`` with an initial token budget ``b_0(x)`` estimated from the
cold-start model, and overrides the task reward with ``-1`` for any trajectory
whose total token budget ``T(y)`` exceeds a scaled threshold ``tau * b_0(x)``.
The budget is per *problem*, not global: an easy question and a hard one get
different allowances, which is what makes the control meaningful.

What ``T(y)`` counts depends on the domain.  For general tasks it is the number
of *thinking* tokens; for agentic tasks it is the cumulative output tokens,
including both reasoning traces and tool-call arguments.

Training follows a stage-wise curriculum over the multiplier ``tau``.  A
*max-budget* variant is trained first with a relatively large ``tau`` -- while
still capping the maximum, to suppress excessive overthinking -- and ``tau`` is
then annealed to smaller values to obtain the **high** and **low** effort
experts.  Crossing three domains with three effort levels yields the nine expert
models that MOPD later consolidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models.llm.kimi_k3.posttrain.xtml import K3_SUPPORTED_EFFORT

__all__ = [
    "BudgetController",
    "EffortSchedule",
    "effort_experts",
    "DOMAINS",
    "EFFORTS",
]

#: The three RL domains of section 4.1.2.
DOMAINS: tuple[str, ...] = ("general", "general_agent", "coding_agent")
#: The three reasoning-effort levels, defined once by the chat template: the wire
#: format decides what an effort level *is*, and RL only selects among the levels
#: K3 ships.  ``xtml.EFFORT_LEVELS`` additionally reserves "medium".
EFFORTS: tuple[str, ...] = K3_SUPPORTED_EFFORT


def effort_experts() -> list[tuple[str, str]]:
    """9 个 domain × effort 专家组合。"""
    return [(d, e) for d in DOMAINS for e in EFFORTS]


@dataclass
class BudgetController:
    """对任务奖励施加预算覆盖。"""

    #: Per-problem base budgets, estimated from the cold-start model.
    base_budgets: dict[str, int] = field(default_factory=dict)
    tau: float = 4.0
    #: Absolute cap, applied even in the max-budget stage.
    hard_cap: int | None = None
    #: 'thinking' for general tasks, 'output' for agentic ones.
    count_mode: str = "thinking"

    def budget_for(self, problem_id: str) -> int:
        """获取问题的基础预算。"""
        if problem_id not in self.base_budgets:
            raise KeyError(f"no cold-start budget estimated for problem {problem_id!r}")
        return self.base_budgets[problem_id]

    def threshold(self, problem_id: str) -> float:
        """计算问题的预算阈值：tau * b_0(x)，可选硬上限。"""
        t = self.tau * self.budget_for(problem_id)
        return min(t, self.hard_cap) if self.hard_cap else t

    def token_count(self, *, thinking_tokens: int, output_tokens: int) -> int:
        """根据 count_mode 选择计数方式。"""
        return thinking_tokens if self.count_mode == "thinking" else output_tokens

    def apply(
        self,
        problem_id: str,
        task_reward: float,
        *,
        thinking_tokens: int = 0,
        output_tokens: int = 0,
    ) -> tuple[float, bool]:
        """返回 (reward, exceeded)。

        超预算时完全覆盖任务奖励——正确但过长的答案不会被部分奖励。
        """
        used = self.token_count(
            thinking_tokens=thinking_tokens, output_tokens=output_tokens
        )
        if used > self.threshold(problem_id):
            return -1.0, True
        return task_reward, False


@dataclass
class EffortSchedule:
    """阶段式退火 tau：先 max，再 high，再 low。"""

    stages: tuple[tuple[str, float], ...] = (("max", 8.0), ("high", 3.0), ("low", 1.0))
    hard_cap: int | None = None

    def tau_for(self, effort: str) -> float:
        """获取特定 effort 级别的 tau 值。"""
        for name, tau in self.stages:
            if name == effort:
                return tau
        raise KeyError(f"unknown effort level {effort!r}")

    def controller(self, effort: str, base_budgets: dict[str, int],
                   *, count_mode: str = "thinking") -> BudgetController:
        """为特定 effort 级别创建 BudgetController。"""
        return BudgetController(
            base_budgets=dict(base_budgets),
            tau=self.tau_for(effort),
            hard_cap=self.hard_cap,
            count_mode=count_mode,
        )

    def is_monotone(self) -> bool:
        """effort 级别必须有序：max 允许最多 token。"""
        taus = [tau for _, tau in self.stages]
        return taus == sorted(taus, reverse=True)
