r"""Multi-Teacher On-Policy Distillation (paper section 4.1.3, Eq. 15).

RL produces nine specialised policies -- three domains crossed with three
reasoning-effort levels.  MOPD consolidates them into one model.

For a given domain ``d`` and sampled effort level ``e``, optimisation is guided
by the corresponding teacher ``pi_teacher^(d,e)`` among the nine.  Given an input
``x`` and the prefix ``y_<t``, the per-token on-policy-distillation reward
evaluated on ``y_t`` between teacher and student ``pi_theta`` is

    r_opd^d(y_t | e, x, y_<t)
        = clip( sg( log( pi_teacher^(d,e)(y_t | x, y_<t) / pi_theta(y_t | x, y_<t) ) ),
                -R_max, R_max )

where ``sg`` is stop-gradient and ``R_max > 0`` clips extreme advantage signals,
which is what keeps RL stable when the student is far from a teacher.

The design's real advantage is that this is a *dense reward inside the existing
RL framework*, not a separate distillation loop: every infrastructure
optimisation -- partial rollout above all -- applies unchanged, which is what
makes distilling long-horizon behaviour affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from models.llm.kimi_k3.posttrain.rl.budget import EFFORTS

__all__ = ["MOPDConfig", "opd_reward", "multi_teacher_reward", "select_teacher", "token_logprobs"]


@dataclass
class MOPDConfig:
    """MOPD 配置：裁剪阈值和 effort 级别。"""
    #: Clipping threshold R_max on the log-ratio reward.
    r_max: float = 4.0
    #: Effort levels sampled during consolidation; the same set the budget
    #: controller anneals over, so a teacher exists for every one of them.
    efforts: tuple[str, ...] = EFFORTS


def opd_reward(
    teacher_logprobs: Tensor,
    student_logprobs: Tensor,
    *,
    r_max: float = 4.0,
) -> Tensor:
    """Eq. 15，per-token 裁剪 log-ratio 奖励。

    两个输入都是已实现 token 的 log-probability。结果被 detached：这是奖励，不是可微分 loss。
    """
    ratio = teacher_logprobs.detach() - student_logprobs.detach()
    return ratio.clamp(-r_max, r_max)


def select_teacher(teachers: dict[tuple[str, str], object], domain: str, effort: str):
    """为采样的 domain 和 effort 选择教师。"""
    key = (domain, effort)
    if key not in teachers:
        raise KeyError(f"no teacher for domain={domain!r} effort={effort!r}")
    return teachers[key]


def multi_teacher_reward(
    teacher_logprobs: dict[tuple[str, str], Tensor],
    student_logprobs: Tensor,
    domain: str,
    effort: str,
    *,
    cfg: MOPDConfig | None = None,
) -> Tensor:
    """一条轨迹的奖励，路由到正确的 domain/effort 教师。"""
    cfg = cfg or MOPDConfig()
    teacher = select_teacher(teacher_logprobs, domain, effort)  # type: ignore[arg-type]
    return opd_reward(teacher, student_logprobs, r_max=cfg.r_max)  # type: ignore[arg-type]


def token_logprobs(logits: Tensor, targets: Tensor) -> Tensor:
    """已实现 token 的 log-probability： ``(B, T-1)``。"""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return logp.gather(-1, targets[:, 1:].unsqueeze(-1)).squeeze(-1)
