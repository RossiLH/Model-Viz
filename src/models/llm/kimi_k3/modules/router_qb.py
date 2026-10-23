r"""Quantile Balancing: auxiliary-loss-free MoE routing (section 2.3.3, appendices C-D).

K3 routes with a per-expert bias added to the score used for Top-k selection but
*excluded* from the mixture weights::

    s_i = sigmoid(W_r x_i)
    T_i = argtopk_k(s_i + b)            selection sees the bias
    p_ij = s_ij / sum_{r in T_i} s_ir   gradients do not

so ``b`` steers dispatch without distorting the optimisation of the router.

The question is how to set ``b``.  The usual loss-free rule nudges it by a fixed
step in the direction of the load error, ``b <- b + gamma * sign(target - load)``,
whose step size is a hyperparameter and whose equilibration is slow -- and at 896
experts per layer, slow equilibration means persistently imbalanced expert-parallel
training and undertrained experts.

Quantile Balancing instead jumps straight to the bias that *makes* each expert's
load equal to its target, in one forward pass.  Routing uses Top-(k+1) rather than
Top-k on the biased score: the first ``k`` entries are the routes actually taken,
and the ``(k+1)``-th is the cutoff ``alpha_i`` an expert must exceed to enter token
``i``'s Top-k.  Taking the cutoff from the routing itself avoids a separate
token-side quantile.  With the cutoffs fixed, the count of tokens routed to expert
``j`` under a candidate bias is monotone in the threshold, so setting that count to
the target ``q = mk/n`` makes the bias the ``(q+1)``-th largest margin
``s_ij - alpha_i``.  That is Eq. 14::

    bhat_j <- -quantile_{1-k/n}(s_{:,j} - alpha)
    b      <- bhat - mean(bhat)

Appendix C derives this as the exact coordinate minimiser of the dual of the
maximum-score balanced-assignment LP, which is why it needs no learning rate.

The update is applied on the *next* step -- a batch is never routed with a bias
derived from itself -- and the bias is frozen at inference.

Histogram estimation (appendix D)
---------------------------------
At scale the quantile spans millions of margins sharded across data-parallel ranks
and gradient-accumulation steps, so gathering them exactly is not viable.  Instead
each rank histograms the *required bias* ``r_ij = alpha_i - s_ij`` -- the bias that
would place expert ``j`` exactly at token ``i``'s cutoff -- into per-expert bins.
Counts are additive, so a single integer all-reduce yields the histogram of the
pooled global batch, and the estimate is the true global quantile up to the bin
width, not an average of per-rank quantiles.  Cost is ``n * B`` integers per layer
per step, independent of the token count.
"""

# 【中文释义】本模块实现「Quantile Balancing（QB，分位数均衡）」：一种无辅助损失
# （auxiliary-loss-free）的 MoE 路由负载均衡方法（论文 §2.3.3，附录 C、D）。
#
# 问题：MoE 中不同专家天然热度不均，有的被过度选中、有的饿死（未训练充分）。
# 传统做法分两类：
#   - 辅助损失（auxiliary loss）：在总 loss 上加一项负载均衡惩罚，缺点是与主 loss
#     竞争、引入额外超参权重；
#   - 固定步长无损失规则：b ← b + γ·sign(target - load)（每步沿负载误差方向挪一小步），
#     缺点是有超参步长 γ、收敛慢——在 K3 每层 896 个专家的规模下，慢收敛意味着
#     EP 训练长期负载失衡 + 部分专家长期欠训练。
#
# QB 的思路：不再"挪小步"，而是**一步直接跳到让每个专家负载恰好等于目标值的偏置**。
# 关键机制：
#   1. 路由打分 sigmoid(W_r x)，Top-k 选择时用 s + b（带偏置），但混合权重只用
#      s 归一化（p_ij = s_ij / Σ s_ir），即偏置只影响"谁被选中"，不影响"选中后
#      权重有多大"→ 不扭曲路由器的优化目标。
#   2. 用 Top-(k+1) 而不是 Top-k：前 k 个是实际路由，第 (k+1) 个 biased score 是
#      cutoff α_i（token i 要进入 Top-k 所需超过的阈值）。cutoff 取自路由本身，
#      省去单独的 token 侧分位数。
#   3. 固定 cutoff 后，"候选偏置下路由到专家 j 的 token 数"关于阈值单调，把它设为
#      目标 q = mk/n，就得到偏置 = 第 (q+1) 大的 margin (s_ij - α_i)。即 Eq. 14：
#          bhat_j ← -quantile_{1-k/n}(s_{:,j} - α)
#          b      ← bhat - mean(bhat)
#   4. 附录 C 证明这是"最大打分均衡指派 LP 对偶问题"的精确坐标最小化，因此
#      不需要学习率。
#
# 工程注意点：
#   - 更新在**下一步**生效（本批从不用本批算出的偏置路由自己）；
#   - 推理时偏置冻结（forward 只路由，不 accumulate）；
#   - 大规模下收集全量 margin 不现实（跨数据并行 rank + 梯度累积步骤），改走
#     附录 D 的直方图：每个 rank 把"所需偏置" r_ij = α_i - s_ij 装进逐专家分箱，
#     计数可加，一次整数 all-reduce 就得到全局直方图 → 估出的分位数误差至多是
#     一个 bin 宽（不是各 rank 分位数的平均）。通信成本固定为 n·B 个整数/层/步，
#     与 token 数无关。

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor, nn

__all__ = ["QuantileBalancingRouter", "RouterOutput"]


class RouterOutput:
    """Routing decision plus everything the balancer needs for its next update."""

    __slots__ = ("indices", "weights", "scores", "cutoff", "n_tokens")

    def __init__(
        self,
        indices: Tensor,
        weights: Tensor,
        scores: Tensor,
        cutoff: Tensor,
        n_tokens: int,
    ):
        # 【中文释义】一次路由的全部产物：
        #   - indices：(tokens, k) —— 每个 token 实际选中的 top_k 个专家 id（int64）
        #   - weights：(tokens, k) —— 归一化混合权重（sigmoid score 在选中集内
        #     renormalize，与偏置无关）
        #   - scores： (tokens, n_experts) —— 原始 sigmoid 分数（未加偏置），
        #     QB 计算 margin 时需要它
        #   - cutoff： (tokens,)，第 (k+1) 个 biased score，即每个 token 的阈值 α
        #   - n_tokens：tokens 数（供直方图统计规模）
        # 注意区分：偏置 b 只进 Top-k 选择，不进 weights 的归一化，这是
        # "无损失"的关键（见模块 docstring）。
        self.indices = indices        # (tokens, k) long
        self.weights = weights        # (tokens, k) float
        self.scores = scores          # (tokens, n_experts) raw sigmoid scores
        self.cutoff = cutoff          # (tokens,) the (k+1)-th biased score
        self.n_tokens = n_tokens


class QuantileBalancingRouter(nn.Module):
    bias: Tensor
    histogram: Tensor
    hist_lo: Tensor
    hist_width: Tensor
    token_count: Tensor
    load: Tensor

    def __init__(
        self,
        dim: int,
        n_experts: int,
        top_k: int,
        *,
        enabled: bool = True,
        bins: int = 1000,
        ema: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        self.enabled = enabled
        self.bins = bins
        self.ema = ema
        self.weight = nn.Parameter(torch.empty(n_experts, dim))
        nn.init.normal_(self.weight, std=dim**-0.5)

        # Buffers, not parameters: the bias is set by the balancing rule, never by
        # a gradient, and it is frozen at inference.
        self.register_buffer("bias", torch.zeros(n_experts))
        self.register_buffer("histogram", torch.zeros(n_experts, bins, dtype=torch.long))
        self.register_buffer("hist_lo", torch.zeros(()))
        self.register_buffer("hist_width", torch.zeros(()))
        self.register_buffer("token_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("load", torch.zeros(n_experts, dtype=torch.long))

    # -- routing -------------------------------------------------------------

    def forward(self, x: Tensor) -> RouterOutput:
        """``x`` is ``(tokens, dim)``; the caller flattens batch and sequence."""
        # 【中文释义】路由打分 = sigmoid(W_r x)。注意与论文一致——训练与推理都是
        # sigmoid，而不是 softmax（softmax 会把所有专家的分数耦合在一起；sigmoid
        # 让每个专家独立响应，便于做逐专家分位数均衡）。
        scores = torch.sigmoid(torch.nn.functional.linear(x.float(), self.weight.float()))
        biased = scores + self.bias

        # Top-(k+1): k routes plus the cutoff, in one selection.
        # 【中文释义】一次 topk 同时拿到"实际路由（前 k 个）"与"阈值 cutoff（第 k+1
        # 个，biased score）"。它对应论文 Eq. 14 中的 α_i：专家分数 s_ij 必须超过
        # cutoff 才会进入 token i 的 Top-k。boundary case：当 k+1 > n_experts 时
        # （top_k 几乎等于全部专家），没有第 k+1 个，取 -inf 占位（即没有门槛）。
        k1 = min(self.top_k + 1, self.n_experts)
        top_val, top_idx = torch.topk(biased, k1, dim=-1)
        indices = top_idx[:, : self.top_k]
        cutoff = (
            top_val[:, self.top_k]
            if k1 > self.top_k
            else top_val[:, -1].new_full((x.shape[0],), float("-inf"))
        )

        # 【中文释义】混合权重只看原始 scores（不带偏置）= 被选中专家的 sigmoid
        # 分数，在选中的 k 个之间重新归一化（Eq. 11 的 p_ij）。clamp_min 防止
        # 全部为 0 时除零。
        chosen = torch.gather(scores, 1, indices)
        weights = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if self.training and self.enabled:
            self._accumulate(scores.detach(), cutoff.detach(), indices.detach())

        return RouterOutput(indices, weights.to(x.dtype), scores, cutoff, x.shape[0])

    # -- balancing -----------------------------------------------------------

    @torch.no_grad()
    def _accumulate(self, scores: Tensor, cutoff: Tensor, indices: Tensor) -> None:
        """Histogram the required bias ``r_ij = alpha_i - s_ij`` (appendix D)."""
        # 【中文释义】训练时每个前向把一路历史数据压进直方图（附录 D）。
        # 直接收集"每个 token-专家对的 margin"不可行：M×E 个浮点要跨 rank 聚合。
        # 改为统计"所需偏置" r_ij = α_i - s_ij（把专家 j 恰好推到 token i 的 cutoff
        # 所需的偏置值），按专家分成 bins 个桶存计数——计数可加，一次整数 all-reduce
        # 即得全局直方图。代价恒为 n·B 个整数/层/步，与 token 数无关。
        b_min = float(self.bias.min())
        b_max = float(self.bias.max())
        lo = b_min - 1.0
        width = (b_max - b_min + 2.0) / self.bins

        if int(self.token_count) == 0:
            self.hist_lo.fill_(lo)
            self.hist_width.fill_(width)
            self.histogram.zero_()
            self.load.zero_()
        lo = float(self.hist_lo)
        width = float(self.hist_width)

        required = cutoff.unsqueeze(1) - scores                  # (tokens, experts)
        idx = ((required - lo) / width).floor().long().clamp_(0, self.bins - 1)
        flat = idx + torch.arange(
            self.n_experts, device=idx.device
        ).unsqueeze(0) * self.bins
        self.histogram.view(-1).scatter_add_(
            0, flat.reshape(-1), torch.ones_like(flat.reshape(-1))
        )
        self.token_count += scores.shape[0]
        self.load += torch.bincount(indices.reshape(-1), minlength=self.n_experts)

    @torch.no_grad()
    def update_bias(self, group=None) -> Tensor:
        """Apply Eq. 14 from the pooled histogram.  Call once per optimizer step.

        Returns the new bias.  The single ``n * bins`` integer all-reduce here is
        the entire communication cost of load balancing.
        """
        # 【中文释义】按直方图估算出的分位数更新偏置（Eq. 14），通常每 optimize step
        # 调用一次。流程：
        #   1) all-reduce 直方图与 token 计数（如果多卡）：偏置只做负载均衡用，通信量
        #     就是 n*bins 个整数 / (层, 步)；
        #   2) 目标负载 q = m·k/n（m 个 token，每个选 k 个专家，平均到 n 个专家）；
        #   3) 沿直方图求第 q 大的分箱：对每个专家找"累计计数首次 ≥ q"的 bin，
        #      再在 bin 内按比例插值得到分位数估计：
        #         b_hat_j = hist_lo + (bin_idx + frac) * hist_width
        #      这里 frac 是 (q - before)/counts（bin 已排到那一格后的位置）；
        #   4) 减去均值再 EMA 平滑，写回 bias（Eq. 14 的 mean-centre 处理保证
        #      sum(b)=0，尺度可控）。
        if not self.enabled or int(self.token_count) == 0:
            return self.bias

        hist = self.histogram
        total = self.token_count.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(hist, group=group)
            dist.all_reduce(total, group=group)

        m = int(total)
        q = m * self.top_k / self.n_experts        # target load per expert
        cum = hist.cumsum(dim=1)
        target = torch.tensor(q, device=hist.device).ceil()

        # First bin whose cumulative count reaches the target load.
        reached = cum >= target
        bin_idx = torch.where(
            reached.any(dim=1),
            reached.float().argmax(dim=1),
            torch.full((self.n_experts,), self.bins - 1, device=hist.device, dtype=torch.long),
        )
        counts = hist.gather(1, bin_idx.unsqueeze(1)).squeeze(1).float()
        before = cum.gather(1, bin_idx.unsqueeze(1)).squeeze(1).float() - counts
        frac = torch.where(
            counts > 0,
            ((q - before) / counts.clamp_min(1.0)).clamp(0.0, 1.0),
            torch.zeros_like(counts),
        )
        b_hat = self.hist_lo + (bin_idx.float() + frac) * self.hist_width
        b_hat = b_hat - b_hat.mean()               # mean-centre, Eq. 14

        if self.ema > 0.0:
            b_hat = self.ema * self.bias + (1.0 - self.ema) * b_hat

        self.bias.copy_(b_hat)
        self.histogram.zero_()
        self.token_count.zero_()
        return self.bias

    @torch.no_grad()
    def update_bias_exact(self, scores: Tensor, cutoff: Tensor) -> Tensor:
        """Exact Eq. 14 on a materialised batch.  Test oracle for the histogram."""
        # 【中文释义】直接用真分位数 torch.quantile 在一整批显式张量上算 Eq. 14：
        #   b_hat_j = -quantile_{1-k/n}(... ) = quantile 在 margin 上取 top_k/n 分位，
        #   再 mean-centre。由于直方图法只是它的降精度近似（误差 ≤ 一个 bin 宽），
        #   update_bias_exact 专门作为直方图的测试 oracle（test 对照）。
        required = cutoff.unsqueeze(1) - scores
        qq = self.top_k / self.n_experts
        b_hat = torch.quantile(required.float(), qq, dim=0)
        b_hat = b_hat - b_hat.mean()
        self.bias.copy_(b_hat)
        return self.bias

    @torch.no_grad()
    def load_balance_stats(self) -> dict[str, float]:
        """Max/min load ratio and coefficient of variation over the last window."""
        # 【中文释义】负载均衡的量化健康指标，供 demo / 监控观察偏置更新前后的变化：
        #   - max_min_ratio：最忙专家 / 最闲专家 的负载比（理想 ≈1）
        #   - cv：负载的变异系数（std / mean，理想 → 0）
        #   - max_over_mean：峰值负载与均值的比值（理想 ≈1）
        load = self.load.float()
        if float(load.sum()) == 0:
            return {"max_min_ratio": 1.0, "cv": 0.0, "max_over_mean": 1.0}
        mean = load.mean()
        return {
            "max_min_ratio": float(load.max() / load.min().clamp_min(1.0)),
            "cv": float(load.std() / mean.clamp_min(1e-9)),
            "max_over_mean": float(load.max() / mean.clamp_min(1e-9)),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, experts={self.n_experts}, top_k={self.top_k}, "
            f"qb={self.enabled}, bins={self.bins}"
        )
