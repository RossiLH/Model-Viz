"""Demo 2: 生成式奖励模型 (Agentic GRM) — 如何对不可验证任务进行奖励判断。

演示 Rubric 协议、冗长惩罚、Tournament 二元比较。
"""

import sys, os
from pathlib import Path

# 本目录位于 experiments/models/llm/kimi_k3/posttrain, 向上 5 级即 Model-Viz 根
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.llm.kimi_k3.posttrain.rl.grm import Rubric, Candidate, AgenticGRM, tournament

def main():
    print("=" * 65)
    print("Demo 2: Agentic GRM (生成式奖励模型)")
    print("=" * 65)

    # ---------------------------------------------------------------
    # 场景 1: 天气查询任务
    # ---------------------------------------------------------------
    print("\n[1] 场景: 天气查询 (不可验证的通用任务)")
    print("-" * 40)

    # 定义 Rubric: 评判标准 + 权重
    rubric = Rubric(
        criteria=["正确性", "完整性", "简洁性"],
        weights=[0.5, 0.3, 0.2],
    )
    print(f"  Rubric 标准: {rubric.criteria}")
    print(f"  Rubric 权重: {rubric.weights}")

    # 3 个候选回答
    candidates = [
        Candidate(ident="short",  text="晴天"),
        Candidate(ident="medium", text="今天北京晴天，最高温度 25°C"),
        Candidate(ident="long",   text="今天天气非常好，阳光明媚，万里无云，最高温度 25 度，"
                  "最低温度 15 度，适合出门活动，建议做好防晒"),
    ]

    # 模拟 judge 对每个候选的评分 (per-criterion)
    scores = {
        "short":  [0.4, 0.2, 1.0],   # 正确但太简短
        "medium": [0.9, 0.8, 0.9],   # 正确、完整、简洁
        "long":   [0.9, 0.9, 0.2],   # 正确、完整、但太冗长
    }

    # 单次评估
    print("\n  [1a] 单次评估 (evaluate)")
    grm = AgenticGRM(sigma=3.0, base_verbosity={"weather": 30})
    for c in candidates:
        entry = grm.evaluate("weather", rubric, c, scores[c.ident])
        total = f"{entry.total:.2f}" if not entry.disqualified else "DISQUALIFIED"
        print(f"    {c.ident:8s} | len={c.length:>2d} | scores={scores[c.ident]} "
              f"| total={total} {f'({entry.reason})' if entry.disqualified else ''}")

    # 二元比较
    print("\n  [1b] 二元比较 (compare)")
    winner = grm.compare("weather", rubric, candidates[0], candidates[1],
                         scores["short"], scores["medium"])
    print(f"    short vs medium → winner: {winner}")

    winner = grm.compare("weather", rubric, candidates[1], candidates[2],
                         scores["medium"], scores["long"])
    print(f"    medium vs long  → winner: {winner}")

    # ---------------------------------------------------------------
    # 场景 2: Tournament (round-robin)
    # ---------------------------------------------------------------
    print("\n[2] Tournament (round-robin 竞技场)")
    print("-" * 40)

    grm2 = AgenticGRM(sigma=100.0, base_verbosity={"weather": 1000})  # 不触发冗长惩罚
    result = tournament(grm2, "weather", rubric, candidates, scores)
    print("  胜率排名:")
    for ident, rate in sorted(result.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"    {ident:8s} | {rate:.2f} | {bar}")
    print(f"  胜率之和 = {sum(result.values()):.2f} (应等于 {len(candidates)-1})")

    # ---------------------------------------------------------------
    # 场景 3: 冗长惩罚 (Verbosity Cap)
    # ---------------------------------------------------------------
    print("\n[3] 冗长惩罚 (Verbosity Cap)")
    print("-" * 40)

    grm3 = AgenticGRM(sigma=2.0, base_verbosity={"task": 50})
    limit = grm3.verbosity_limit("task")
    print(f"  sigma = {grm3.sigma}, l0 = 50 -> limit = sigma * l0 = {limit:.0f} chars")

    test_cases = [
        Candidate(ident="ok",   text="x" * 60),   # 60 < 100, 不触发
        Candidate(ident="long", text="x" * 150),   # 150 > 100, 触发
    ]
    rubric2 = Rubric(criteria=["质量"])
    for c in test_cases:
        entry = grm3.evaluate("task", rubric2, c, [0.9])
        status = "取消资格" if entry.disqualified else f"total={entry.total:.2f}"
        print(f"  {c.ident:6s} | len={c.length:>3d} | {status}")

    # ---------------------------------------------------------------
    # 场景 4: 编程任务
    # ---------------------------------------------------------------
    print("\n[4] 场景: 编程任务 (多维度评判)")
    print("-" * 40)

    code_rubric = Rubric(
        criteria=["正确性", "可读性", "性能", "边界处理"],
        weights=[0.4, 0.2, 0.2, 0.2],
    )
    code_candidates = [
        Candidate(ident="naive",  text="def f(n): return sum(range(n))"),
        Candidate(ident="better", text="def f(n): return n*(n-1)//2"),
        Candidate(ident="robust", text="def f(n):\n    if n < 0: raise ValueError\n    return n*(n-1)//2"),
    ]
    code_scores = {
        "naive":  [0.7, 0.9, 0.3, 0.1],
        "better": [0.7, 0.7, 0.9, 0.1],
        "robust": [0.9, 0.8, 0.9, 0.9],
    }

    grm4 = AgenticGRM(sigma=100.0, base_verbosity={"code": 1000})
    result = tournament(grm4, "code", code_rubric, code_candidates, code_scores)
    print("  胜率排名:")
    for ident, rate in sorted(result.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 20)
        print(f"    {ident:8s} | {rate:.2f} | {bar}")

    # ---------------------------------------------------------------
    # 场景 5: Scorepad 审计
    # ---------------------------------------------------------------
    print("\n[5] Scorepad (审计记录)")
    print("-" * 40)
    print(f"  grm4.scorepad 共 {len(grm4.scorepad)} 条记录")
    for entry in grm4.scorepad[:3]:
        print(f"    {entry.candidate:8s} | total={entry.total:.2f} | "
              f"criteria={entry.rubric}")

    print("\n" + "=" * 65)
    print("核心要点:")
    print("  1. Rubric: 先定标准后评分，可审计而非凭感觉")
    print("  2. Verbosity Cap: 防止模型学会'越长越好'的 reward hacking")
    print("  3. Tournament: round-robin 二元比较，胜率作为 group reward")
    print("  4. Scorepad: 每次评判都记录，支持事后审计")
    print("=" * 65)

if __name__ == "__main__":
    main()
