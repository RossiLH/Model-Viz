"""episode 元数据 —— 把"这次任务是怎么干的"喂给模型

================================================================================
这是什么：给次优数据"贴价签"的一整套机制
================================================================================

π0.7 敢吃别人不敢吃的**次优数据**（失败示范、失误严重的成功示范、旧模型
自主采集的数据），靠的是一句话（§VI-A）：
    "we need to appropriately label these diverse trajectories with information
     about *how* the task was performed so that the model can correctly
     contextualize them."                                   (§V-C, p5)

"贴价签" = episode metadata。它把"这条轨迹快不快 / 干得好不好 / 有没有失误"
这些**执行质量信号**写进上下文，让模型学会把这些信号和动作关联起来。
推理时把信号拨到"高质、无误"档，模型就照着它见过的最好示范的样子干活。

本模块实现三件事（各对应论文一个句子，出处见函数 docstring）：
1. ``bin_episode_length`` —— speed 的分箱（500 步一箱，§V-C）
2. ``sample_prompt_dropout`` —— 训练时的两级 prompt dropout（§V-E）
3. ``make_metadata_identity`` —— 元数据在 batch 内的样例批内扩散（推理用）

CFG（classifier-free guidance）不在这里，见 config.Pi07CFGConfig + model 的
``_cfg_denoise``。

================================================================================
为什么是"分箱"而不是"直接给数字"：稀疏性 vs 可泛化性
================================================================================

一个 episode 的长度是连续值（比如 1873 步、2000 步、8124 步）。如果直接把
原始步数喂给模型，会有两个麻烦：
- 取值空间太稀疏：训练集里见过的长度是有限几个离散值，模型对没见过的长度
  （比如 2600 步）只能外推，而外推是最不可靠的；
- 语义无意义：1873 步和 2000 步对动作生成没有本质差别，但把 1873 和 2000
  当成两个不同类别，等于逼模型区分无意义的东西。

分箱（binning）把连续值压到"语义上等价"的粗糙粒度：500 步以内的差异被抹平，
模型只学到"这任务大概多快"这个层级的信息。论文原话（§V-C）：
    "We discretize the values in an interval of 500 steps, i.e., values between
     1750 and 2250 are binned to '2000 steps'."

注意"between 1750 and 2250"这个区间——它不是 [1750, 2250) 这种平铺直叙，
而是把 2000 当中心、半径 250 的对称邻域。这就是"四舍五入到最近的 500 倍数"
的意思：|x − 2000| ≤ 250 才归到 2000。本函数按这个语义实现。

================================================================================
两级 dropout：先整块、再逐字段 —— 为什么不是只整块
================================================================================

训练时每类 prompt 组件都要 dropout（论文："we train the model with dropout
for each component, and then prompt the model flexibly combining modalities"）。
对 episode metadata 具体是（§V-E）：
    "For episode metadata, we drop it entirely 15% of the time, and additionally
     each component (overall speed, overall quality, and mistake label) is
     dropped with 5% probability individually."

两级的目的：
- **整块 15%**：让模型学会"完全没有元数据也能干活"（对应推理时某些设备/任务
  压根不提供速度/质量标签的冷启动场景）；
- **逐字段 5%**：让模型学会"缺个别字段也能干活"，并且每个字段独立可弃 ——
  这样推理时可以任意组合注入，而不会因为某字段缺失就崩。

两级的组合概率：一个字段最终保留的概率是 (1 − 0.15) × (1 − 0.05) = 0.8075。
注意论文说 **control mode 不做 dropout**（"We do not apply dropout for the
control mode"）——控制模式是硬约束（joint 还是 ee 动作），没有它模型连动作
空间都分不清，所以必须恒在。

================================================================================
推理时固定档位：把"最好"写成输入
================================================================================

论文 §VII 把元数据在运行时拨到固定档位（p7）：
    "Overall speed: set per-task to the 15th percentile of the episode length
     from the task."
    "Overall quality: always set to 5, which is the highest score."
    "Mistake: always set to false, meaning no mistake."

三个档位各自的道：
- **speed 取 15 百分位**：不是取最快，而是取"比 85% 的示范都快"。取极值
  （最快）会让模型进入它训练分布之外的外推区，动作容易失真；15 百分位是
  "快且仍在分布内"的折中。
- **quality 恒取 5**：让模型永远照最高质量示范的样子生成。
- **mistake 恒 false**：明确要求"别犯错"。
"""
from __future__ import annotations
import torch


def bin_episode_length(length_steps: torch.Tensor, bin_size: int = 500) -> torch.Tensor:
    """把 episode 长度（步数）分箱到 ``bin_size`` 步的粒度（默认 500，论文值）。

    语义：**四舍五入到最近的 bin_size 倍数**。论文 §V-C（p5）：
        "We discretize the values in an interval of 500 steps, i.e., values
         between 1750 and 2250 are binned to '2000 steps'."
    "between 1750 and 2250" 是 2000 的对称邻域（半径 250 = bin_size/2），
    即 round(x / bin_size) * bin_size。

    Args:
        length_steps: ``(...)`` 任意形状的 episode 长度（步数）。
        bin_size: 箱宽。论文固定 500。

    Returns:
        与输入同形状的分箱后长度（仍然是步数，只是被压到箱粒度）。
    """
    return (torch.round(length_steps / bin_size) * bin_size).to(length_steps.dtype)


def sample_prompt_dropout(
    n_samples: int,
    *,
    metadata_drop_prob: float = 0.15,
    per_field_drop_prob: float = 0.05,
    n_metadata_fields: int = 3,
    subgoal_keep_prob: float = 0.25,
    subtask_drop_given_subgoal: float = 0.30,
    control_mode_drop_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """采样每个训练样本的 prompt 组件保留/丢弃掩码（论文 §V-E 的两级 dropout）。

    返回的掩码都表示"**保留**"（True = 该组件参与该样本）。
    布局约定（与 model.embed_prefix 的参数一一对应）：
    - ``metadata``: ``(B,)`` 元数据整块是否保留（1 − 0.15）
    - ``metadata_fields``: ``(B, 3)`` 逐字段保留（speed/quality/mistake，
      ``1 − 0.05``）——**仅在整块保留时才有意义**，整块丢弃时全 False
    - ``subgoal``: ``(B,)`` 子目标图像是否加入（论文 25% 加入，故 keep=0.25）
    - ``subtask``: ``(B,)`` 子任务指令是否保留（``1 − 0.30``，且**只在有
      子目标图像的样本里**才掷这枚骰子；无子目标图像时必须保留子任务，
      否则样本既没图像又没指令，彻底不可学）
    - ``control_mode``: ``(B,)`` 恒 True（论文明确 "do not apply dropout
      for the control mode"）

    注意字段 drop 与整块 drop 的先后关系：论文是先掷整块（15% 整块消失），
    再在**剩余**的样本里逐字段掷（5%）。本实现里两个掩码是独立采样的，
    整块丢弃的样本其字段掩码也被强制清零，组合语义一致：
    ``P(字段最终保留) = (1 − 0.15)(1 − 0.05) = 0.8075``。

    为什么 subgoal 是 "keep=0.25" 而非 "drop=0.75"：论文原话是 "we **only add**
    the visual subgoal images to **25%** of the examples" —— 它是加入率不是
    丢弃率。用 keep 命名能避免把方向搞反（这是本模块最容易写反的地方）。
    """
    def _rand(shape) -> torch.Tensor:
        return torch.rand(shape, generator=generator)

    keep_metadata = _rand((n_samples,)) >= metadata_drop_prob
    fields = _rand((n_samples, n_metadata_fields)) >= per_field_drop_prob
    fields = fields & keep_metadata.unsqueeze(-1)          # 整块丢弃则字段全弃

    keep_subgoal = _rand((n_samples,)) < subgoal_keep_prob
    # 有子目标图像的样本里，子任务指令以 30% 概率被图像"顶替"；
    # 没有子目标图像的样本，子任务指令必须保留（兜底监督）。
    subtask = torch.ones(n_samples, dtype=torch.bool)
    drop_subtask = (keep_subgoal & (_rand((n_samples,)) < subtask_drop_given_subgoal))
    subtask = subtask & ~drop_subtask

    return {
        "metadata": keep_metadata,
        "metadata_fields": fields,
        "subgoal": keep_subgoal,
        "subtask": subtask,
        "control_mode": torch.ones(n_samples, dtype=torch.bool),
    }


def make_metadata_identity(
    speed_steps: torch.Tensor,
    quality: torch.Tensor,
    mistake: torch.Tensor,
    *,
    speed_bin_size: int = 500,
) -> torch.Tensor:
    """把推理时给定的元数据组装成模型输入张量（分箱 + 排列）。

    π0.7 元数据在模型里是 ``(B, n_metadata)`` 的连续张量（每字段一个 token，
    与 embed_prefix 的 ``metadata_proj`` 对齐），字段顺序固定为
    [speed, quality, mistake]（论文 §V-E prompt 示例 "Speed: 8000. Quality: 5.
    Mistake: false." 的顺序）。本函数负责把外部给的原始值（长度步数、质量分、
    布尔失误）规范化到这个约定格式。

    推理时论文把 speed 取该任务 episode 长度的**第 15 百分位**、quality 恒 5、
    mistake 恒 false（§VII）。调用方负责算百分位，本函数只做分箱与拼装。
    """
    speed_binned = bin_episode_length(speed_steps, speed_bin_size)
    return torch.stack(
        [
            speed_binned.to(torch.float32),
            quality.to(torch.float32),
            mistake.to(torch.float32),
        ],
        dim=-1,
    )


def guidance_interp(v_cond: torch.Tensor, v_uncond: torch.Tensor, beta: float) -> torch.Tensor:
    """CFG 速度场外插（π0.7 §VII 公式的 flow-matching 形态）。

    论文公式（p7，C = 完整上下文，C^uncond = 无条件上下文）：
        ∇_a log π_θ(a|o, C) + β·(∇_a log π_θ(a|o, C) − ∇_a log π_θ(a|o, C^uncond))
    在 flow matching 里把"评分/梯度"换成速度场：
        v_guided = v_cond + β·(v_cond − v_uncond) = (1+β)·v_cond − β·v_uncond

    直觉：v_cond 把动作推向"带条件"的分布，v_uncond 把动作推向"无条件"的分布，
    两者之差 Δ = v_cond − v_uncond 就是"**条件往哪个方向拽动作**"。CFG 把这个
    拽的方向再放大 β 倍并加上去，条件的效果被强化（β=0 时退化为无条件、
    β=1 时是普通条件、β>1 是外插增强）。论文建议中等值 β ∈ {1.3, 1.7, 2.2}，
    β 过大会让生成动作失真。

    为什么能外插：训练时每类 prompt 组件都做了 dropout，模型同时学会了
    有/无该条件的两个分布（详见 sample_prompt_dropout），两个前向是同一套
    权重、同一批去噪轨迹上算的，差向量 Δ 才是有意义的"条件效应方向"。
    代价：每个去噪步跑两次前向，推理开销翻倍 —— 故默认关闭。
    """
    return v_cond + beta * (v_cond - v_uncond)
