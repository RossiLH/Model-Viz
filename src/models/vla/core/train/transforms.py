# -*- coding: utf-8 -*-
"""数据变换管线（复刻官方 openpi.transforms 的核心子集）

每个 transform 是一个可调用对象：接收一个 dict，返回变换后的 dict。
通过 compose() 可以把多个 transform 按顺序串联起来。

与官方的对应关系：
- Normalize       ←  transforms.Normalize
- Unnormalize     ←  transforms.Unnormalize
- PadStatesAndActions ← transforms.PadStatesAndActions
- DeltaActions    ←  transforms.DeltaActions
- AbsoluteActions ←  transforms.AbsoluteActions
- ResizeImages    ←  transforms.ResizeImages
- compose         ←  transforms.compose / CompositeTransform
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Any

import numpy as np
from PIL import Image

from models.vla.core.train.norm_stats import NormStats


def compose(transforms: Sequence[Callable[[dict], dict]]) -> Callable[[dict], dict]:
    """把多个变换组合成一个可调用对象，按顺序执行。

    Args:
        transforms: 变换列表，按列表顺序依次应用

    Returns:
        一个接收 dict 返回 dict 的函数
    """
    transforms = list(transforms)

    def _apply(data: dict) -> dict:
        for t in transforms:
            data = t(data)
        return data

    return _apply


@dataclass(frozen=True)
class Normalize:
    """分位数归一化或 z-score 归一化（复刻官方 transforms.Normalize）。

    【为什么需要这个变换】
    训练时模型输入的数值范围差异极大：关节角（弧度，~[-π, π]）和夹爪开合
    （[0, 1]）如果直接堆在一起送进网络，前几层梯度会被关节维度主导，夹爪
    信号被淹没。归一化到统一分布后，每个维度对 loss 的贡献数量级相当。

    PI0.5 走分位数路线（夹爪长尾分布对 z-score 不友好，0/1 处的尖峰会
    把标准差撑到 0.3 这种没意义的值）；PI0 走 z-score（见官方
    `config.py:187`：`use_quantile_norm = model_type != ModelType.PI0`）。

    【为什么图像类数据要跳过】
    图像在归一化前都是 uint8 [0, 255]。如果把它们也塞进 stats 字典做归一化
    （1）q01/q99 几乎不变（0 和 255），做了等于没做；（2）下面
    `PadStatesAndActions` 之后会被错误地塞进 `state` 统计里。
    实际上图像归一化是视觉编码器（SigLIP/ViT）自己做的，
    不归这条管线管。

    Args:
        norm_stats: {key: NormStats} 字典。key 通常是 "state"、"actions"，
                    也可能是某种特定数据集字段。
        use_quantiles: True=分位数归一化 (q01/q99 -> [-1, 1])，False=z-score。
                       训练时与 norm_stats 的计算方式必须严格匹配。
        strict: True=缺 key 时报错（生产/调试模式）；
                False=静默跳过（数据管线容错模式）。
    """

    norm_stats: Dict[str, NormStats]
    use_quantiles: bool = True
    strict: bool = False

    def __call__(self, data: dict) -> dict:
        for key, stats in self.norm_stats.items():
            if key not in data:
                if self.strict:
                    # 严格模式下缺关键字段直接抛错，训练时一秒钟能发现配置错误
                    raise KeyError(f"Normalize: 数据中缺少 key '{key}'")
                continue

            # 必须先用 view 拿原始 dtype，再 asarray 默认复制为 float64 做计算。
            # 之前版本的 bug 是先转 float64 再判断 arr.dtype == np.uint8 —— 永远为假，
            # 图像类数据被错误地归一化，`.astype(arr.dtype)` 又写回 float64，
            # 训练时图像被破坏成小数张量。
            orig = data[key]
            orig_dtype = getattr(orig, "dtype", None)
            if orig_dtype is not None and orig_dtype == np.uint8:
                # 图像类数据透传，不做归一化
                continue

            arr = np.asarray(orig, dtype=np.float64)

            if self.use_quantiles:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(
                        f"分位数归一化需要 q01/q99，但 '{key}' 的统计量没有")
                # stats 可能比数据宽（因为 stats 是按 model_action_dim 算的），
                # 所以只取数据实际需要的最后一维范围
                q01 = stats.q01[..., :arr.shape[-1]]
                q99 = stats.q99[..., :arr.shape[-1]]
                # +1e-6 防止 q99==q01 时除零；*2-1 把 [0,1] 区间映射到 [-1, 1]
                normalized = (arr - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                # z-score 路径（PI0 用）：mean/std 同样按数据最后一维裁剪
                mean = stats.mean[..., :arr.shape[-1]]
                std = stats.std[..., :arr.shape[-1]]
                normalized = (arr - mean) / (std + 1e-6)

            # 用原 dtype 还原（如果是 float32 输入就不要硬升成 float64，
            # 会让数据加载器多一倍的内存与带宽）。
            if orig_dtype is not None and np.issubdtype(orig_dtype, np.floating):
                data[key] = normalized.astype(orig_dtype)
            else:
                data[key] = normalized

        return data


@dataclass(frozen=True)
class Unnormalize:
    """Normalize 的逆变换（推理时把模型输出从 [-1, 1] 转回真实物理单位）。

    【为什么必须有 Unnormalize】
    模型训练时学的是归一化空间里的动作分布；下游控制器（机械臂 SDK、
    MuJoCo 环境等）要的是物理单位（弧度、米），不接受 [-1, 1]。
    Unnormalize 就是那一道桥。

    【为什么会有 `arr.shape[-1] > q01.shape[-1]` 分支】
    训练时可能对 state/actions 做了零填充（`PadStatesAndActions`），
    把维度撑到 `model_action_dim`。统计量是按原始维度算的，
    反归一化时只对真实维度做变换，多出来的 padding 维保持原样不动。
    这个分支在 PI0/PI0.5 的夹爪 + 主臂独立 action_dim 场景里经常触发。

    Args:
        norm_stats: 必须与训练时的 Normalize 使用完全相同的 stats。
        use_quantiles: 必须与训练时一致，否则反归一化结果会错。
    """

    norm_stats: Dict[str, NormStats]
    use_quantiles: bool = True

    def __call__(self, data: dict) -> dict:
        for key, stats in self.norm_stats.items():
            if key not in data:
                continue

            orig = data[key]
            orig_dtype = getattr(orig, "dtype", None)
            # 同样先做图像跳过判断
            if orig_dtype is not None and orig_dtype == np.uint8:
                continue

            arr = np.asarray(orig, dtype=np.float64)

            if self.use_quantiles:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(
                        f"分位数反归一化需要 q01/q99，但 '{key}' 没有")
                q01 = stats.q01[..., :arr.shape[-1]]
                q99 = stats.q99[..., :arr.shape[-1]]
                if arr.shape[-1] > q01.shape[-1]:
                    # 拆分：主体维度反归一化、padding 维度透传
                    extra = arr[..., q01.shape[-1]:]
                    # 严格按 Normalize 的公式反推：+1 → /2 → * (q99-q01+eps) → +q01
                    main = (arr[..., :q01.shape[-1]] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                    unnormalized = np.concatenate([main, extra], axis=-1)
                else:
                    unnormalized = (arr + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            else:
                mean = stats.mean[..., :arr.shape[-1]]
                std = stats.std[..., :arr.shape[-1]]
                if arr.shape[-1] > mean.shape[-1]:
                    extra = arr[..., mean.shape[-1]:]
                    main = arr[..., :mean.shape[-1]] * (std + 1e-6) + mean
                    unnormalized = np.concatenate([main, extra], axis=-1)
                else:
                    unnormalized = arr * (std + 1e-6) + mean

            if orig_dtype is not None and np.issubdtype(orig_dtype, np.floating):
                data[key] = unnormalized.astype(orig_dtype)
            else:
                data[key] = unnormalized

        return data


@dataclass(frozen=True)
class PadStatesAndActions:
    """把 state 和 actions 零填充到模型的 action_dim。

    【为什么需要 pad】
    真实数据集的 action 维度千差万别：Libero 7 维（6 关节 + 1 夹爪）、
    Aloha 单臂 14 维、双臂 28 维、推块 2 维。但模型只在训练时定义了
    一个固定的 `model_action_dim`（通常等于数据集最大维度的圆整值）。
    不 pad 会导致 batch 内不同 episode 维度对不齐；只 pad 不后续识别
    又会让 loss 把 padding 维当真信号学习。

    约定：
        - state:   从 (..., D) pad 到 (..., model_action_dim)
        - actions: 从 (..., D) pad 到 (..., model_action_dim)
        - 只 pad 最后一维
        - 如果 D >= model_action_dim 则截断（保守起见保前面维度）

    注：下游 `_pad_last_dim` 还有一个反向能力：当数据维数已经超过目标时
    截断（实际工程中也会发生，例如不同数据源混训）。这是安全网，不是常规路径。
    """

    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            data["state"] = _pad_last_dim(data["state"], self.model_action_dim)
        if "actions" in data:
            data["actions"] = _pad_last_dim(data["actions"], self.model_action_dim)
        return data


def _pad_last_dim(arr, target_dim: int, value: float = 0.0):
    """把数组最后一维填充到 target_dim。

    行为：
        - 已达到 target_dim：原样返回
        - 超过 target_dim：截断到 target_dim
        - 不足 target_dim：用 value（默认 0）在末尾补齐

    仅作用于最后一维 —— 因为 state / actions 的语义是"每一行是一个
    状态或动作向量"，padding 在末尾（高位）能保证前几维语义不变。

    Args:
        arr: 任意 numpy 兼容输入（list / tuple / ndarray 都可以）
        target_dim: 目标最后一维长度
        value: 填充值，默认为 0.0
    """
    arr = np.asarray(arr)
    if arr.shape[-1] >= target_dim:
        return arr[..., :target_dim]
    # pad_width 格式: 每个轴 (before, after)，
    # 这里只对最后一维做后填充，前面的轴 (0, 0) 不动
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - arr.shape[-1])]
    return np.pad(arr, pad_width, mode="constant", constant_values=value)


@dataclass(frozen=True)
class DeltaActions:
    """绝对动作转 delta（增量）动作 —— 以当前 state 为参考原点。

    【为什么需要这个变换】
    机器人关节的绝对角度范围很大，且与初始构型强相关：同一个"把手臂抬高
    10 度"的动作，在不同初始姿态下绝对目标值完全不同。模型要从图像里
    推断出绝对角度非常难。改成增量后，"抬高 10 度"永远是同一个数字，
    分布集中、方差小，训练稳定得多。

    【为什么不是所有维度都做 delta】
    夹爪只有开/闭两个语义状态（0/1 附近），绝对值本身就是有意义的指令。
    做成增量反而丢失了"目标是开还是闭"的信息。所以用 mask 区分：
    关节维度做 delta，夹爪维度保持绝对。

    Args:
        mask: 布尔序列，True=该维做 delta，False=保持绝对。
              长度可以小于实际动作维度（超出部分视为 False，即保持绝对）。
              None 表示整个变换是空操作（no-op）。

    形状约定：
        actions 支持 (A,) / (H, A) / (B, H, A)；
        state 支持 (D,) / (B, D)。
        当 state 只有一个向量时，广播给所有动作步（chunk 内共用同一参考点）。
    """

    mask: Sequence[bool]

    def __call__(self, data: dict) -> dict:
        # 缺字段或 mask 为 None 时直接返回，保证变换可安全串在任意管线里
        if "actions" not in data or "state" not in data:
            return data
        if self.mask is None:
            return data

        mask = np.asarray(self.mask, dtype=bool)
        dims = int(mask.shape[-1])  # mask 覆盖的维度数
        actions = np.asarray(data["actions"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)

        # 记录原始形状，最后要还原回去
        orig_shape = actions.shape
        orig_dtype = np.asarray(data["actions"]).dtype

        # 统一展平成 2 维 (N, A) 处理，避免为每种维度写一套分支
        actions_flat = actions.reshape(-1, actions.shape[-1])
        state_flat = state.reshape(-1, state.shape[-1])

        # 实际能做 delta 的维度 = mask 覆盖数 与 动作宽度 的较小值
        # （动作可能比 mask 短，例如 mask 给了 7 维但动作只有 2 维）
        eff = min(dims, actions_flat.shape[-1])
        eff_mask = mask[:eff]

        # 取参考点：state 的前 eff 维
        if state_flat.shape[0] == actions_flat.shape[0]:
            # 每个动作样本对应一个 state（批量场景）
            ref = state_flat[:, :eff]
        else:
            # state 只有一个向量（或数量不匹配）→ 广播给所有动作步。
            # 这是 chunk 场景的常态：一个观测对应未来 H 步动作，
            # H 步都以这同一个观测时刻的 state 为参考原点。
            ref = state_flat[0, :eff]

        # np.where 按位选择：mask 为 True 的维度减去参考点，False 的维度原样保留
        converted = np.where(eff_mask,
                             actions_flat[:, :eff] - ref,
                             actions_flat[:, :eff])

        # 拼回 mask 覆盖范围之外的维度（这些维度不受影响）
        if actions_flat.shape[-1] > eff:
            result = np.concatenate([converted, actions_flat[:, eff:]], axis=-1)
        else:
            result = converted

        data["actions"] = result.reshape(orig_shape).astype(orig_dtype)
        return data


@dataclass(frozen=True)
class AbsoluteActions:
    """DeltaActions 的逆变换：delta（增量）动作转回绝对动作。

    【什么时候用】
    推理阶段。模型学的是增量空间的动作，但机器人控制器要的是绝对目标值，
    所以模型输出后必须加回参考点。这个变换与 DeltaActions 必须用
    **同一个 mask**，否则会出现"某维度加了两次参考点"或"该加没加"的错位。

    数学上就是 DeltaActions 的严格逆运算（减号变加号），
    因此 AbsoluteActions(DeltaActions(x)) == x 是一条必须成立的不变式，
    测试里有对应的往返验证。

    Args:
        mask: 与 DeltaActions 相同的布尔序列。None 表示空操作。
    """

    mask: Sequence[bool]

    def __call__(self, data: dict) -> dict:
        # 与 DeltaActions 对称：缺字段或 mask 为 None 时空操作
        if "actions" not in data or "state" not in data:
            return data
        if self.mask is None:
            return data

        mask = np.asarray(self.mask, dtype=bool)
        dims = int(mask.shape[-1])
        actions = np.asarray(data["actions"], dtype=np.float64)
        state = np.asarray(data["state"], dtype=np.float64)

        orig_shape = actions.shape
        orig_dtype = np.asarray(data["actions"]).dtype

        # 同样展平成 (N, A) 统一处理
        actions_flat = actions.reshape(-1, actions.shape[-1])
        state_flat = state.reshape(-1, state.shape[-1])

        # 有效维度取 mask 长度与动作宽度的较小值（与 DeltaActions 保持一致）
        eff = min(dims, actions_flat.shape[-1])
        eff_mask = mask[:eff]

        if state_flat.shape[0] == actions_flat.shape[0]:
            ref = state_flat[:, :eff]
        else:
            # 单个 state 广播给所有动作步，与 DeltaActions 的广播规则严格对称，
            # 这是往返一致性成立的前提
            ref = state_flat[0, :eff]

        # 唯一与 DeltaActions 不同的地方：这里是加号
        converted = np.where(eff_mask,
                             actions_flat[:, :eff] + ref,
                             actions_flat[:, :eff])

        if actions_flat.shape[-1] > eff:
            result = np.concatenate([converted, actions_flat[:, eff:]], axis=-1)
        else:
            result = converted

        data["actions"] = result.reshape(orig_shape).astype(orig_dtype)
        return data


@dataclass(frozen=True)
class ResizeImages:
    """把 data["image"] 里的所有相机图像 resize 到统一尺寸。

    【为什么需要】
    不同数据集的相机分辨率各不相同（Libero 256x256、Aloha 480x640、
    本项目推块环境 32x32 或 64x64），但模型的视觉编码器要求固定输入尺寸
    （由 patch_size 和位置编码长度决定）。这一层负责统一。

    【为什么用 PIL 而不是 torch.nn.functional.interpolate】
    这一层跑在数据管线里，处理的是 uint8 的 numpy 数组。
    用 PIL 可以避免"numpy → tensor → resize → tensor → numpy"的来回转换，
    也不需要在数据层引入 torch 依赖。

    Args:
        height: 目标高度
        width:  目标宽度

    输入约定：
        data["image"] 是 {相机名: ndarray (H, W, 3) uint8} 字典
    """

    height: int
    width: int

    def __call__(self, data: dict) -> dict:
        if "image" not in data:
            return data

        images = data["image"]
        resized = {}
        for key, img in images.items():
            arr = np.asarray(img)
            # 非 (H, W, 3) 形状的数据原样透传（例如 mask 或其他辅助字段）
            if arr.ndim != 3 or arr.shape[2] != 3:
                resized[key] = img
                continue
            # 尺寸已经正确就跳过，省掉一次无意义的重采样
            if arr.shape[0] == self.height and arr.shape[1] == self.width:
                resized[key] = img
                continue
            # 注意 PIL 的 resize 参数顺序是 (width, height)，与 numpy 的
            # (height, width) 相反 —— 这是个高频踩坑点
            pil_img = Image.fromarray(arr.astype(np.uint8))
            pil_img = pil_img.resize((self.width, self.height), Image.BILINEAR)
            resized[key] = np.array(pil_img, dtype=np.uint8)
        data["image"] = resized
        return data
