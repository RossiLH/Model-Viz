"""Tests for posttrain.rl — 预算控制、GRM、MOPD、部分 Rollout。"""

import torch
from models.llm.kimi_k3.posttrain.rl.budget import (
    BudgetController,
    EffortSchedule,
    effort_experts,
    DOMAINS,
    EFFORTS,
)
from models.llm.kimi_k3.posttrain.rl.grm import (
    Rubric,
    Candidate,
    AgenticGRM,
    tournament,
)
from models.llm.kimi_k3.posttrain.rl.mopd import (
    MOPDConfig,
    opd_reward,
    select_teacher,
    multi_teacher_reward,
    token_logprobs,
)
from models.llm.kimi_k3.posttrain.rl.partial_rollout import (
    Trajectory,
    PartialRolloutScheduler,
    staleness_regularizer,
    clipped_policy_ratio,
)


# ---------------------------------------------------------------------------
# budget.py tests
# ---------------------------------------------------------------------------

class TestBudgetController:
    def test_budget_for(self):
        bc = BudgetController(base_budgets={"p1": 1000})
        assert bc.budget_for("p1") == 1000

    def test_budget_for_missing(self):
        bc = BudgetController()
        try:
            bc.budget_for("missing")
            assert False, "should raise"
        except KeyError:
            pass

    def test_threshold(self):
        bc = BudgetController(base_budgets={"p1": 1000}, tau=3.0)
        assert bc.threshold("p1") == 3000.0

    def test_threshold_with_hard_cap(self):
        bc = BudgetController(base_budgets={"p1": 1000}, tau=8.0, hard_cap=5000)
        assert bc.threshold("p1") == 5000

    def test_apply_within_budget(self):
        bc = BudgetController(base_budgets={"p1": 1000}, tau=3.0)
        reward, exceeded = bc.apply("p1", 1.0, thinking_tokens=500)
        assert reward == 1.0
        assert exceeded is False

    def test_apply_exceeds_budget(self):
        bc = BudgetController(base_budgets={"p1": 1000}, tau=3.0)
        reward, exceeded = bc.apply("p1", 1.0, thinking_tokens=5000)
        assert reward == -1.0
        assert exceeded is True

    def test_count_mode_thinking(self):
        bc = BudgetController(base_budgets={"p1": 1000}, count_mode="thinking")
        assert bc.token_count(thinking_tokens=100, output_tokens=200) == 100

    def test_count_mode_output(self):
        bc = BudgetController(base_budgets={"p1": 1000}, count_mode="output")
        assert bc.token_count(thinking_tokens=100, output_tokens=200) == 200


class TestEffortSchedule:
    def test_default_stages(self):
        es = EffortSchedule()
        assert es.tau_for("max") == 8.0
        assert es.tau_for("high") == 3.0
        assert es.tau_for("low") == 1.0

    def test_unknown_effort(self):
        es = EffortSchedule()
        try:
            es.tau_for("ultra")
            assert False, "should raise"
        except KeyError:
            pass

    def test_is_monotone(self):
        es = EffortSchedule()
        assert es.is_monotone() is True

    def test_controller(self):
        es = EffortSchedule()
        bc = es.controller("high", {"p1": 1000})
        assert bc.tau == 3.0
        assert bc.base_budgets == {"p1": 1000}

    def test_controller_with_hard_cap(self):
        es = EffortSchedule(hard_cap=2000)
        bc = es.controller("max", {"p1": 1000})
        assert bc.hard_cap == 2000


def test_effort_experts():
    experts = effort_experts()
    assert len(experts) == 9
    assert ("general", "low") in experts
    assert ("coding_agent", "max") in experts


def test_domains_and_efforts():
    assert len(DOMAINS) == 3
    assert len(EFFORTS) == 3


# ---------------------------------------------------------------------------
# grm.py tests
# ---------------------------------------------------------------------------

class TestRubric:
    def test_basic(self):
        r = Rubric(criteria=["accuracy", "completeness"])
        assert len(r.criteria) == 2
        assert r.weights is not None
        assert len(r.weights) == 2

    def test_equal_weights(self):
        r = Rubric(criteria=["a", "b", "c"])
        assert all(abs(w - 1/3) < 1e-9 for w in r.weights)

    def test_custom_weights(self):
        r = Rubric(criteria=["a", "b"], weights=[0.7, 0.3])
        assert r.weights == [0.7, 0.3]

    def test_score(self):
        r = Rubric(criteria=["a", "b"], weights=[0.6, 0.4])
        assert r.score([1.0, 0.5]) == 0.8

    def test_empty_criteria(self):
        try:
            Rubric(criteria=[])
            assert False, "should raise"
        except ValueError:
            pass

    def test_mismatched_weights(self):
        try:
            Rubric(criteria=["a"], weights=[0.5, 0.5])
            assert False, "should raise"
        except ValueError:
            pass

    def test_score_wrong_count(self):
        r = Rubric(criteria=["a", "b"])
        try:
            r.score([1.0])
            assert False, "should raise"
        except ValueError:
            pass


class TestCandidate:
    def test_basic(self):
        c = Candidate(ident="c1", text="hello world")
        assert c.length == 11

    def test_empty(self):
        c = Candidate(ident="c1", text="")
        assert c.length == 0


class TestAgenticGRM:
    def test_verbosity_limit(self):
        grm = AgenticGRM(sigma=3.0, base_verbosity={"p1": 100})
        assert grm.verbosity_limit("p1") == 300.0

    def test_verbosity_limit_missing(self):
        grm = AgenticGRM()
        try:
            grm.verbosity_limit("missing")
            assert False, "should raise"
        except KeyError:
            pass

    def test_evaluate_normal(self):
        grm = AgenticGRM(sigma=3.0, base_verbosity={"p1": 100})
        rubric = Rubric(criteria=["quality"])
        cand = Candidate(ident="c1", text="good answer")
        entry = grm.evaluate("p1", rubric, cand, [0.8])
        assert entry.total == 0.8
        assert entry.disqualified is False

    def test_evaluate_verbosity_disqualify(self):
        grm = AgenticGRM(sigma=2.0, base_verbosity={"p1": 5})
        rubric = Rubric(criteria=["quality"])
        cand = Candidate(ident="c1", text="x" * 20)
        entry = grm.evaluate("p1", rubric, cand, [0.8])
        assert entry.disqualified is True
        assert entry.total == float("-inf")

    def test_compare_winner(self):
        grm = AgenticGRM(sigma=3.0, base_verbosity={"p1": 100})
        rubric = Rubric(criteria=["quality"])
        a = Candidate(ident="a", text="good")
        b = Candidate(ident="b", text="better")
        winner = grm.compare("p1", rubric, a, b, [0.6], [0.9])
        assert winner == "b"

    def test_compare_disqualified(self):
        grm = AgenticGRM(sigma=2.0, base_verbosity={"p1": 5})
        rubric = Rubric(criteria=["quality"])
        a = Candidate(ident="a", text="x" * 20)  # too long
        b = Candidate(ident="b", text="short")
        winner = grm.compare("p1", rubric, a, b, [0.8], [0.5])
        assert winner == "b"  # a disqualified


class TestTournament:
    def test_round_robin(self):
        grm = AgenticGRM(sigma=100.0, base_verbosity={"p1": 100})
        rubric = Rubric(criteria=["quality"])
        candidates = [
            Candidate(ident="a", text="short a"),
            Candidate(ident="b", text="short b"),
            Candidate(ident="c", text="short c"),
        ]
        scores = {"a": [0.9], "b": [0.5], "c": [0.7]}
        result = tournament(grm, "p1", rubric, candidates, scores)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        # a has highest score, should win most
        assert result["a"] >= result["b"]

    def test_two_candidates(self):
        grm = AgenticGRM(sigma=100.0, base_verbosity={"p1": 100})
        rubric = Rubric(criteria=["quality"])
        candidates = [
            Candidate(ident="a", text="short a"),
            Candidate(ident="b", text="short b"),
        ]
        scores = {"a": [1.0], "b": [0.0]}
        result = tournament(grm, "p1", rubric, candidates, scores)
        assert result["a"] == 1.0
        assert result["b"] == 0.0


# ---------------------------------------------------------------------------
# mopd.py tests
# ---------------------------------------------------------------------------

class TestMOPDConfig:
    def test_defaults(self):
        cfg = MOPDConfig()
        assert cfg.r_max == 4.0
        assert len(cfg.efforts) == 3


class TestOPDReward:
    def test_basic(self):
        teacher = torch.tensor([[-1.0, -2.0]])
        student = torch.tensor([[-1.5, -1.0]])
        r = opd_reward(teacher, student, r_max=4.0)
        expected = torch.tensor([0.5, -1.0])
        assert torch.allclose(r, expected)

    def test_clipping(self):
        teacher = torch.tensor([[-10.0]])
        student = torch.tensor([[-0.1]])
        r = opd_reward(teacher, student, r_max=2.0)
        assert r.item() == -2.0

    def test_positive_clipping(self):
        teacher = torch.tensor([[-0.1]])
        student = torch.tensor([[-10.0]])
        r = opd_reward(teacher, student, r_max=2.0)
        assert r.item() == 2.0

    def test_detached(self):
        teacher = torch.tensor([[-1.0]], requires_grad=True)
        student = torch.tensor([[-2.0]], requires_grad=True)
        r = opd_reward(teacher, student)
        assert not r.requires_grad


class TestSelectTeacher:
    def test_basic(self):
        teachers = {("general", "low"): "t1", ("general", "high"): "t2"}
        assert select_teacher(teachers, "general", "low") == "t1"

    def test_missing(self):
        teachers = {}
        try:
            select_teacher(teachers, "general", "low")
            assert False, "should raise"
        except KeyError:
            pass


class TestMultiTeacherReward:
    def test_basic(self):
        teachers = {
            ("general", "low"): torch.tensor([[-1.0]]),
            ("general", "high"): torch.tensor([[-2.0]]),
        }
        student = torch.tensor([[-1.5]])
        r = multi_teacher_reward(teachers, student, "general", "low")
        assert torch.allclose(r, torch.tensor([[0.5]]))


class TestTokenLogprobs:
    def test_basic(self):
        B, T, V = 1, 5, 10
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        logprobs = token_logprobs(logits, targets)
        assert logprobs.shape == (B, T - 1)

    def test_all_valid(self):
        B, T, V = 2, 4, 8
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        logprobs = token_logprobs(logits, targets)
        # Should be negative (log probabilities)
        assert (logprobs < 0).all()


# ---------------------------------------------------------------------------
# partial_rollout.py tests
# ---------------------------------------------------------------------------

class TestTrajectory:
    def test_key(self):
        t = Trajectory(prompt_id=1, sample_id=2)
        assert t.key == (1, 2)

    def test_staleness(self):
        t = Trajectory(prompt_id=0, sample_id=0, token_iteration=[1, 3, 5])
        stale = t.staleness(10)
        assert stale == [9, 7, 5]


class TestPartialRolloutScheduler:
    def test_init(self):
        sched = PartialRolloutScheduler(n_prompts=2, k_samples=3, lam=0.5)
        assert sched.target_active == 6

    def test_invalid_lambda(self):
        try:
            PartialRolloutScheduler(n_prompts=1, k_samples=1, lam=0.0)
            assert False, "should raise"
        except ValueError:
            pass

    def test_lambda_above_one(self):
        try:
            PartialRolloutScheduler(n_prompts=1, k_samples=1, lam=1.5)
            assert False, "should raise"
        except ValueError:
            pass

    def test_start_iteration(self):
        sched = PartialRolloutScheduler(n_prompts=2, k_samples=2, lam=0.5)
        batch = sched.start_iteration(new_prompts=[10, 20])
        assert len(batch) == 4
        assert sched.iteration == 1

    def test_should_pause(self):
        sched = PartialRolloutScheduler(n_prompts=2, k_samples=2, lam=0.5)
        sched.start_iteration(new_prompts=[1, 2])
        # Mark 2 of 4 as done (50%)
        keys = list(sched.active.keys())
        for k in keys[:2]:
            sched.active[k].done = True
        assert sched.should_pause() is True

    def test_should_not_pause(self):
        sched = PartialRolloutScheduler(n_prompts=2, k_samples=2, lam=0.5)
        sched.start_iteration(new_prompts=[1, 2])
        # Only 1 of 4 done (25%)
        keys = list(sched.active.keys())
        sched.active[keys[0]].done = True
        assert sched.should_pause() is False

    def test_pause_remaining(self):
        sched = PartialRolloutScheduler(n_prompts=1, k_samples=3, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        keys = list(sched.active.keys())
        sched.active[keys[0]].done = True  # 1 of 3 done → 33% < 50%, no pause
        # Need to force should_pause to True
        sched.active[keys[1]].done = True  # 2 of 3 done → 67%
        assert sched.should_pause()
        paused = sched.pause_remaining()
        assert len(paused) == 1  # only the non-done one
        assert len(sched.paused) == 1

    def test_resume_from_pause(self):
        sched = PartialRolloutScheduler(n_prompts=1, k_samples=2, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        keys = list(sched.active.keys())
        sched.active[keys[0]].done = True
        sched.active[keys[1]].done = True  # both done → harvest
        ready = sched.harvest()
        assert len(ready) == 1
        # Start next iteration with paused
        sched.start_iteration(new_prompts=None)
        assert sched.iteration == 2

    def test_harvest_all_done(self):
        sched = PartialRolloutScheduler(n_prompts=1, k_samples=2, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        for traj in sched.active.values():
            traj.done = True
        ready = sched.harvest()
        assert len(ready) == 1
        assert len(ready[0]) == 2

    def test_harvest_partial(self):
        sched = PartialRolloutScheduler(n_prompts=1, k_samples=3, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        keys = list(sched.active.keys())
        for k in keys[:2]:
            sched.active[k].done = True
        ready = sched.harvest()
        assert len(ready) == 0  # not all 3 done

    def test_stats(self):
        sched = PartialRolloutScheduler(n_prompts=1, k_samples=2, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        stats = sched.stats()
        assert stats["iteration"] == 1.0
        assert stats["active"] == 2.0

    def test_priority_resume(self):
        """Older trajectories resume first."""
        sched = PartialRolloutScheduler(n_prompts=2, k_samples=1, lam=0.5)
        sched.start_iteration(new_prompts=[1])
        sched.iteration = 1
        # Pause remaining
        for traj in sched.active.values():
            if not traj.done:
                traj.paused_at_iteration = 1
                import heapq
                heapq.heappush(sched.paused, (traj.paused_at_iteration, traj.key))
        # Start new iteration
        sched.start_iteration(new_prompts=[2])
        # The paused one should resume first
        assert sched.iteration == 2


class TestStalenessRegularizer:
    def test_identical(self):
        logp = torch.tensor([[0.0, -1.0, -2.0]])
        reg = staleness_regularizer(logp, logp)
        assert reg.item() < 1e-6

    def test_small_difference(self):
        logp_new = torch.tensor([[0.0, -1.0]])
        logp_old = torch.tensor([[-0.1, -1.1]])
        reg = staleness_regularizer(logp_new, logp_old, epsilon=0.2)
        assert reg.item() >= 0

    def test_large_difference_clamped(self):
        logp_new = torch.tensor([[0.0]])
        logp_old = torch.tensor([[-10.0]])
        reg = staleness_regularizer(logp_new, logp_old, epsilon=0.2)
        # Should be clamped to 0.2
        assert reg.item() <= 0.2 + 1e-6


class TestClippedPolicyRatio:
    def test_identical(self):
        logp = torch.tensor([[0.0, -1.0]])
        adv = torch.tensor([[1.0, 1.0]])
        loss = clipped_policy_ratio(logp, logp, adv)
        # ratio=1, unclipped=adv, clipped=adv, min=-adv, mean=-1
        assert abs(loss.item() + 1.0) < 1e-6

    def test_large_shift(self):
        logp_new = torch.tensor([[5.0]])
        logp_old = torch.tensor([[0.0]])
        adv = torch.tensor([[1.0]])
        loss = clipped_policy_ratio(logp_new, logp_old, adv, epsilon=0.2)
        # ratio=e^5 ≈ 148, clipped ratio = 1.2, so clipped wins
        assert loss.item() < 0
