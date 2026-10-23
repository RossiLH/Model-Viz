"""Agentic Generative Reward Model (paper section 4.1.2).

For non-verifiable general tasks there is no oracle, so K3 retains a
tournament-style group reward built from binary comparisons.  Beyond generic
agentic capability, the judge must follow a **mandatory protocol**:

1. read the outcome, product, or text output;
2. generate a rubric;
3. score each candidate against the rubric;
4. record the rubric-assigned scores in a scorepad.

Forcing the rubric to exist *before* the scores, and to be written down, is what
makes the judgment auditable rather than a vibe.

Reward hacking toward verbosity is mitigated by a budget-based verbosity control
analogous to the reasoning-effort control: given an initial verbosity ``l_0``
estimated from the cold-start model and a multiplier ``sigma``, a candidate whose
output length exceeds ``sigma * l_0`` automatically loses the binary comparison.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

__all__ = ["Rubric", "Candidate", "AgenticGRM", "tournament"]


@dataclass
class Rubric:
    """评分标准，在评分之前生成。"""

    criteria: list[str]
    weights: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("a rubric must contain at least one criterion")
        if self.weights is None:
            self.weights = [1.0 / len(self.criteria)] * len(self.criteria)
        if len(self.weights) != len(self.criteria):
            raise ValueError("weights and criteria must have the same length")

    def score(self, per_criterion: list[float]) -> float:
        """计算加权总分。"""
        assert self.weights is not None
        if len(per_criterion) != len(self.criteria):
            raise ValueError("one score per criterion is required")
        return sum(w * s for w, s in zip(self.weights, per_criterion))


@dataclass
class Candidate:
    """候选输出。"""
    ident: str
    text: str
    scores: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class ScorepadEntry:
    """评分卡记录。"""
    candidate: str
    rubric: list[str]
    per_criterion: list[float]
    total: float
    disqualified: bool = False
    reason: str = ""


@dataclass
class AgenticGRM:
    """执行四步协议和冗长惩罚的评判器。"""

    #: Verbosity multiplier; a candidate longer than ``sigma * l_0`` loses.
    sigma: float = 3.0
    #: Cold-start verbosity estimate per prompt.
    base_verbosity: dict[str, int] = field(default_factory=dict)
    scorepad: list[ScorepadEntry] = field(default_factory=list)

    def verbosity_limit(self, prompt_id: str) -> float:
        """获取问题的冗长上限。"""
        if prompt_id not in self.base_verbosity:
            raise KeyError(f"no cold-start verbosity estimated for {prompt_id!r}")
        return self.sigma * self.base_verbosity[prompt_id]

    def evaluate(
        self,
        prompt_id: str,
        rubric: Rubric,
        candidate: Candidate,
        per_criterion: list[float],
    ) -> ScorepadEntry:
        """步骤 3 和 4：按 rubric 评分并记录到 scorepad。"""
        limit = self.verbosity_limit(prompt_id)
        if candidate.length > limit:
            entry = ScorepadEntry(
                candidate.ident, rubric.criteria, per_criterion, float("-inf"),
                disqualified=True,
                reason=f"length {candidate.length} exceeds sigma*l0 = {limit:.0f}",
            )
        else:
            entry = ScorepadEntry(
                candidate.ident, rubric.criteria, per_criterion,
                rubric.score(per_criterion),
            )
        self.scorepad.append(entry)
        return entry

    def compare(
        self,
        prompt_id: str,
        rubric: Rubric,
        a: Candidate,
        b: Candidate,
        scores_a: list[float],
        scores_b: list[float],
    ) -> str:
        """一次二元比较，返回胜者 id。"""
        ea = self.evaluate(prompt_id, rubric, a, scores_a)
        eb = self.evaluate(prompt_id, rubric, b, scores_b)
        if ea.disqualified and not eb.disqualified:
            return b.ident
        if eb.disqualified and not ea.disqualified:
            return a.ident
        return a.ident if ea.total >= eb.total else b.ident


def tournament(
    grm: AgenticGRM,
    prompt_id: str,
    rubric: Rubric,
    candidates: list[Candidate],
    scores: dict[str, list[float]],
) -> dict[str, float]:
    """Round-robin 二元比较，返回胜率作为 group reward。"""
    wins = {c.ident: 0 for c in candidates}
    pairs = list(itertools.combinations(candidates, 2))
    for a, b in pairs:
        winner = grm.compare(prompt_id, rubric, a, b, scores[a.ident], scores[b.ident])
        wins[winner] += 1
    n = max(1, len(candidates) - 1)
    return {ident: w / n for ident, w in wins.items()}
