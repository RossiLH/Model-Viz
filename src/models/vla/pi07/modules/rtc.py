"""RTC（Real-Time Chunking，实时分块）—— 训练期动作前缀条件化

================================================================================
归属考证（重要）：本机制**不属于 π0.5**，故正宅在 pi07/ 下
================================================================================

一手论文证据：

1. **π0.5 论文（2504.16054）完全没有这个机制。**
   全文检索 `real-time chunking` / `delay` / `frozen` / `inpaint` —— **零命中**。
   π0.5 只在 §IV 提到"以 50Hz 直接下发目标位姿（with action chunking）"，
   谈的是"动作分块"这个 2023 年就有的旧概念（ACT/Zhao et al.），
   不是"如何在推理延迟下拼接相邻 chunk"。

2. **RTC 是一篇独立论文的贡献（2506.07339，Black / Galliker / Levine）。**
   该文反过来把 π0.5 当作**被加速的基座策略**：
       "we evaluate RTC on 6 challenging bimanual manipulation tasks
        using the π0.5 VLA [24] as the **base policy**"        (p2)
   附录 A.3 的延迟表里还专门列了一行 "Vanilla π0.5 76ms" 作为对照基线 (p24)。
   —— 若 RTC 是 π0.5 自带的能力，它不会成为"外挂在 π0.5 上的方法"。
   该文的 RTC 是**推理期 inpainting**（inference-time），带 guidance 权重，
   有额外计算开销（RTC 97ms vs Vanilla π0.5 76ms）。

3. **本模块实现的是"训练期 RTC"，出自 2512.05964，首发于 π0.6。**
       "we propose a simple alternative: **simulating inference delay at
        training time** and conditioning on action prefixes directly,
        eliminating any inference-time overhead."               (p1 摘要)
       "In real-world experiments ... with the **π0.6** VLA"     (p1)
   这正是本文件 sample_delay + make_prefix_mask + masked_mse 三件套的语义。

4. **π0.7（2604.15483）正式采纳它，故本项目归到 pi07。**
       "π0.7 **also employs the training-time version of real-time action
        chunking (RTC)** [107, 108] for generating smooth action trajectories
        in the presence of inference delay. During training, we simulate
        delays of **0 to 12 timesteps**, corresponding to a maximum inference
        latency of **240ms on a 50Hz robot**."                   (p7)
   [107] = 2506.07339（推理期 RTC）；[108] = 2512.05964（训练期 RTC）。

结论：π0.5 → 无；RTC 推理期 → 2506.07339；RTC 训练期 → 2512.05964（π0.6）；
      π0.7 → 采纳训练期版本，`max_delay=12`。本项目 `Pi07RTCConfig.max_delay=12`
      即照抄论文这一句。

--------------------------------------------------------------------------------
机制本身：为什么需要它？
--------------------------------------------------------------------------------

前提：VLA 推理要花时间（几十到几百毫秒），而机器人以 50Hz（20ms/步）执行。
困难：等模型算完新 chunk，机器人**已经往前走了 d 步**。这 d 步的动作是
      "既定事实"（已经下发给 PD 控制器，无法撤回）。如果新 chunk 不认这
      d 步，第 d 步与第 d+1 步之间就会出现跳变 —— 论文原话是
      "out-of-distribution jerky movements at chunk boundaries"，
      表现为机械臂在 chunk 边界抽一下，或者干脆停顿。

解法（训练期版本，本模块）：**训练时就把这个既定事实喂给模型**。
      随机采一个延迟 d，把 chunk 前 d 步标成"已提交"，
      令这些位置 time=0（即 x_t = 真实动作，无噪声、干净），
      模型于是学到"看到前 d 步是干净动作时，要生成与之连续的后续动作"。
      损失只在未冻结步上计算 —— 冻结步没有"待预测"的语义，
      算进去只是在惩罚模型复现一个它已经被告知的答案，纯属噪声。

推理时：把真实的已提交动作填进前 d 步，每个去噪步都重新注入（防止被
      Euler 积分改写），末尾再 clamp 一次。相比推理期 inpainting 方案，
      这里**没有任何额外推理开销** —— 这正是训练期方案的卖点。

--------------------------------------------------------------------------------
时间轴约定：本项目 time=0 表示 clean，论文 time=1 表示 clean（方向相反）
--------------------------------------------------------------------------------

这一点极易踩坑，必须写清楚。2512.05964 的 Algorithm 1（p6）原样代码：

    time = jax.random.uniform(time_rng, (b,))
    prefix_mask = jnp.arange(ah)[None, :] < delay[:, None]
    time = jnp.where(prefix_mask, 1.0, time[:, None])       # prefix 置 1.0
    x_t = time[:, :, None] * action_chunk + (1 - time[:, :, None]) * noise
    loss = (pred_v_t - (action_chunk - noise)) ** 2

论文约定：``x_t = t·action + (1−t)·noise``
    → t=1 是干净动作，t=0 是纯噪声；采样时 τ 从 0 积到 1，``dt = +1/num_steps``；
      速度场目标 ``u = action − noise``。

本项目约定（见 core/diffusion.py 的 flow_matching_scheduler）：
``x_t = t·noise + (1−t)·action``
    → t=0 是干净动作，t=1 是纯噪声；采样时 τ 从 1 走到 1/num_steps，
      ``dt = −1/num_steps``；速度场目标 ``u = noise − action``。

两者是同一条直线的**反向参数化**：``t_本项目 = 1 − t_论文``。三处符号同时翻转
（插值系数、积分方向、速度场目标），因此数学上完全等价。

结论：本项目把冻结前缀置 ``time=0``，与论文置 ``time=1.0`` 是**同一件事**
（都表示"这一步已是干净动作"）。这不是实现偏差。之所以不改成论文写法，
是因为 π0/π0.5 沿用 OpenTau/openpi 的 t=0-clean 约定，全仓统一才不会串味。
阅读论文公式时记得做一次 ``t → 1−t`` 的心算替换即可。

--------------------------------------------------------------------------------
与论文 Algorithm 1 的两处刻意偏差（列入遗留差异）
--------------------------------------------------------------------------------

1. **delay 采样区间**：论文注释写 ``Unif[0, max_delay)``（右开，取不到
   max_delay）；本实现用 ``randint(0, max_delay + 1)``（右闭，能取到）。
   理由：π0.7 正文说"simulate delays of **0 to 12** timesteps"，中文语感与
   英文 "0 to 12" 都含 12；且右闭让 ``max_delay`` 这个配置名字面成立。
   影响：仅改变延迟分布的一个端点权重，量级 1/(d_max+1)，不影响机制。

2. **损失归一化粒度**：论文是**全批全局**归一化
   ``sum(loss·postfix) / sum(postfix)``；本项目 compute_loss 走**逐样本**
   归一化（先按各自 postfix 步数求均值，再对 batch 求均值）。
   理由：逐样本标量是 RL token 重加权（``mse * rl_token_weight``）的挂点，
   必须先得到"每样本一个损失值"才能加权。副作用是让 delay 大的样本
   （postfix 步少）不被静默降权 —— 见下方 masked_mse 的详细论证。
   注：本模块的 masked_mse 保留论文的全局归一化写法，作为对照参考实现。

--------------------------------------------------------------------------------
接口语义
--------------------------------------------------------------------------------

训练（在 compute_loss 内）:
    delay       = sample_delay(max_delay, B)          # 每样本独立随机
    prefix_mask = make_prefix_mask(H, delay)          # (B, H) True=已提交
    time        = where(prefix_mask, 0, time)         # 冻结步 time=0（干净动作）
    loss        = masked_mse(v_t, u_t, prefix_mask)   # 只算未冻结步

推理（在 sample_actions 内）:
    每去噪步: x_t = where(prefix_mask, action_prefix, x_t)   # 重注入
    末尾:     x_t = where(prefix_mask, action_prefix, x_t)   # clamp

`max_delay=0` 时：无采样、无 mask、无重注入，行为与"不带 RTC"完全一致。
这个退化路径是刻意保留的 —— 它让 RTC 成为一个可完全关闭的正交特性。

--------------------------------------------------------------------------------
第三块拼图：逐 token timestep 调制（不在本模块，但缺了它 RTC 就是空壳）
--------------------------------------------------------------------------------

2512.05964 把改造拆成三步（p2），本模块只负责第 2、3 步：

    ① "Modify the model architecture to allow for a different flow matching
        timestep for each action timestep. For a transformer that uses
        adaLN-zero conditioning for the flow matching timestep, simply allow
        the scale, shift, and gate to differ between tokens. This does not
        change the number of learnable parameters."
    ② "Use ground-truth, non-noisy actions for the prefix, and set the
        corresponding flow matching timesteps to 1."
    ③ "Mask the loss function so that loss is only computed on outputs
        corresponding to the postfix."

第 ① 步落在 AdaRMSNorm 的 ``per_token_cond`` 开关上（由 Pi07RTCConfig 的
``per_token_timestep`` 打开）。它是**前提**而非锦上添花：若 AdaRMSNorm 把
条件在序列轴均值池化，prefix 步的 time 与 postfix 步的 time 会被压成同一组
scale/shift/gate，模型只能感知"有多少步被冻结"、感知不到"冻结在哪"——
此时 ②③ 退化为"仅遮蔽损失若干列"，RTC 名存实亡。本项目对这一点有回归测试
钉死（延迟位置敏感性：开启时输出差异约 1e-1，关闭时约 1e-7 即数值噪声）。
"""
import torch


def sample_delay(max_delay, batch_size, device=None):
    """采样每样本的随机部署延迟 d。

    为什么每样本独立采样、而非整批用同一个 d：部署时的延迟是随硬件负载
    波动的随机量，训练分布应覆盖整个 [0, max_delay] 区间。若整批同 d，
    等价于每个 step 只见到一种延迟，梯度方差大、覆盖慢。

    Args:
        max_delay: 最大延迟步数（0 时返回全 0，即关闭 RTC）
        batch_size: 批大小
        device: 输出设备

    Returns:
        (batch_size,) long，取值 [0, max_delay]
    """
    if max_delay <= 0:
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    return torch.randint(0, max_delay + 1, (batch_size,), device=device)


def make_prefix_mask(chunk_size, delay, device=None):
    """构造冻结前缀掩码：前 delay 步为 True（已提交、不可改）。

    Args:
        chunk_size: 动作块长度 H
        delay: (B,) long，每样本已提交步数
        device: 输出设备（未使用，保持与 delay.device 一致）

    Returns:
        (B, chunk_size) bool，True = 冻结（已提交）
    """
    arange = torch.arange(chunk_size, device=delay.device)
    return arange.unsqueeze(0) < delay.unsqueeze(1)      # (B, H)


def masked_mse(v_t, u_t, prefix_mask):
    """掩码 Flow Matching MSE：只算未冻结步，按未冻结槽位数归一化。

    为什么必须重新归一化、而不能直接 `mean()`：不同样本的 delay 不同，
    未冻结步数也不同。若用全张量 mean，delay 大的样本（未冻结步少）在
    分母里仍按 H 计，等价于被静默降权 —— 延迟越大的样本学得越少，
    而这恰恰是我们最想让模型学好的困难情形。按实际未冻结槽位数归一化，
    才能让每个样本的"有效监督强度"一致。

    Args:
        v_t: (B, H, A) 预测速度场
        u_t: (B, H, A) 目标速度场
        prefix_mask: (B, H) bool，True = 冻结步（不计入损失）

    Returns:
        标量损失
    """
    keep = ~prefix_mask                                 # (B, H) 未冻结步
    diff = (v_t - u_t) ** 2
    diff = diff * keep[:, :, None]                      # 冻结步清零
    denom = keep.float().sum() * v_t.shape[-1]          # 未冻结槽位总数
    if denom.item() == 0:
        # 极端情形：整批全冻结（delay==H）。返回 0 而非 NaN，保持可反向。
        return (diff * 0).sum()
    return diff.sum() / denom
